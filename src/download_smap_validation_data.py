"""Download point-sampled SMAP soil moisture for matched study locations.

Authentication is prompted interactively and held only in memory. Task state,
coordinate metadata, and downloaded source files are resumable and auditable.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_COORDINATES = (
    PROJECT_ROOT.parent / "SoilEnvironmentUpdates" / "figures" / "spatial_match_diagnostics_points.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external_validation" / "smap_spl3smp_v9"
API_ROOT = "https://appeears.earthdatacloud.nasa.gov/api"
PRODUCT = "SPL3SMP.009"
LAYERS = (
    "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm",
    "Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm",
)
START_DATE = "01-01-2015"
END_DATE = "12-31-2024"


def load_locations(path: Path) -> pd.DataFrame:
    points = pd.read_csv(path)
    required = {"LAT", "LON", "matched"}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"Coordinate file is missing columns: {sorted(missing)}")

    is_matched = points["matched"].astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
    locations = points.loc[is_matched, ["LAT", "LON"]].copy()
    locations["LAT"] = pd.to_numeric(locations["LAT"], errors="raise")
    locations["LON"] = pd.to_numeric(locations["LON"], errors="raise")
    locations = locations.drop_duplicates().sort_values(["LAT", "LON"]).reset_index(drop=True)
    locations.insert(
        0,
        "point_id",
        [f"P{index:05d}" for index in range(1, len(locations) + 1)],
    )
    if locations.empty:
        raise ValueError("No matched coordinates were found in the input file")
    return locations


def signature_for(locations: pd.DataFrame) -> str:
    canonical = locations.to_csv(index=False, float_format="%.8f")
    payload = f"{START_DATE}|{END_DATE}|{PRODUCT}|{LAYERS}|{canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def login(session: requests.Session) -> str:
    username = os.getenv("NASA_EARTHDATA_USERNAME", "").strip()
    password = os.getenv("NASA_EARTHDATA_PASSWORD", "")
    if not username:
        username = input("NASA Earthdata username: ").strip()
    if not password:
        password = getpass.getpass("NASA Earthdata password: ")
    if not username or not password:
        raise ValueError("Earthdata username and password are required")
    response = session.post(f"{API_ROOT}/login", auth=(username, password), timeout=60)
    response.raise_for_status()
    token = response.json().get("token")
    if not token:
        raise RuntimeError("AppEEARS login response did not contain an access token")
    session.headers.update({"Authorization": f"Bearer {token}"})
    return token


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    attempts: int = 6,
    **kwargs: Any,
) -> requests.Response:
    for attempt in range(attempts):
        response = session.request(method, url, timeout=180, **kwargs)
        if response.status_code in {401, 403} and attempt == 0:
            login(session)
            continue
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response
        if attempt == attempts - 1:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        delay = int(retry_after) if retry_after and retry_after.isdigit() else min(30 * (2**attempt), 300)
        print(f"Temporary API response {response.status_code}; retrying in {delay}s", flush=True)
        time.sleep(delay)
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}")


def build_task_payload(task_name: str, chunk: pd.DataFrame) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "task_type": "point",
        "params": {
            "dates": [{"startDate": START_DATE, "endDate": END_DATE}],
            "layers": [{"product": PRODUCT, "layer": layer} for layer in LAYERS],
            "coordinates": [
                {
                    "id": str(row.point_id),
                    "category": "SOTWIS_matched",
                    "latitude": float(row.LAT),
                    "longitude": float(row.LON),
                }
                for row in chunk.itertuples(index=False)
            ],
        },
    }


def task_status(session: requests.Session, task_id: str) -> str:
    response = request_with_retry(session, "GET", f"{API_ROOT}/task/{task_id}")
    status = response.json()
    if isinstance(status, list):
        status = status[0] if status else {}
    task_state = status.get("status") if isinstance(status, dict) else None
    if not task_state:
        raise RuntimeError(f"AppEEARS task response did not contain a status for {task_id}")
    return str(task_state).lower()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "download"


def download_bundle(session: requests.Session, task_id: str, output_dir: Path) -> list[dict[str, Any]]:
    bundle = request_with_retry(session, "GET", f"{API_ROOT}/bundle/{task_id}").json()
    downloaded = []
    for item in bundle.get("files", []):
        filename = safe_filename(item.get("file_name", item["file_id"]))
        destination = output_dir / filename
        if destination.exists() and destination.stat().st_size == int(item.get("file_size", -1)):
            downloaded.append({"file_name": filename, "sha256": item.get("sha256"), "status": "already_present"})
            continue
        response = request_with_retry(
            session,
            "GET",
            f"{API_ROOT}/bundle/{task_id}/{item['file_id']}",
            stream=True,
        )
        digest = hashlib.sha256()
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    output.write(block)
                    digest.update(block)
        actual_hash = digest.hexdigest()
        expected_hash = item.get("sha256")
        if expected_hash and actual_hash.lower() != expected_hash.lower():
            temporary.unlink(missing_ok=True)
            raise IOError(f"SHA-256 mismatch for {filename}")
        temporary.replace(destination)
        downloaded.append({"file_name": filename, "sha256": actual_hash, "status": "downloaded"})
        print(f"Downloaded {filename}", flush=True)
    return downloaded


def run(args: argparse.Namespace) -> None:
    locations = load_locations(args.coordinates)
    output_dir = args.output.resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    locations.to_csv(output_dir / "matched_sampling_locations.csv", index=False)

    batch_size = args.batch_size
    chunks = [locations.iloc[start : start + batch_size] for start in range(0, len(locations), batch_size)]
    manifest_path = output_dir / "appeears_manifest.json"
    signature = signature_for(locations)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_signature") != signature:
            raise ValueError(
                f"{manifest_path} belongs to different coordinates/settings; use a new output directory"
            )
    else:
        manifest = {
            "input_signature": signature,
            "product": PRODUCT,
            "layers": list(LAYERS),
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
            "point_count": len(locations),
            "batch_size": batch_size,
            "tasks": {},
        }

    print(
        f"Matched locations: {len(locations)}; batches: {len(chunks)}; "
        f"period: 2015-01-01 through 2024-12-31",
        flush=True,
    )
    if args.prepare_only:
        print(f"Wrote location list and request plan to {output_dir}")
        return

    session = requests.Session()
    login(session)
    for batch_index, chunk in enumerate(chunks, start=1):
        task_name = f"SMAP-SOTWIS-{batch_index:03d}"
        task_record = manifest["tasks"].get(task_name, {})
        if task_record.get("downloaded"):
            print(f"Skipping completed batch {task_name}", flush=True)
            continue

        task_id = task_record.get("task_id")
        if not task_id:
            payload = build_task_payload(task_name, chunk)
            response = request_with_retry(
                session,
                "POST",
                f"{API_ROOT}/task",
                json=payload,
            )
            task_id = response.json().get("task_id")
            if not task_id:
                raise RuntimeError(f"AppEEARS did not return a task ID for {task_name}")
            task_record = {"task_id": task_id, "point_count": len(chunk), "downloaded": False}
            manifest["tasks"][task_name] = task_record
            write_json(manifest_path, manifest)
            print(f"Submitted {task_name}: task {task_id}", flush=True)

        while True:
            status = task_status(session, task_id)
            task_record["status"] = status
            write_json(manifest_path, manifest)
            print(f"{task_name}: {status}", flush=True)
            if status == "done":
                break
            if status in {"error", "failed", "canceled"}:
                raise RuntimeError(f"AppEEARS task {task_name} ended with status {status}")
            time.sleep(args.poll_seconds)

        task_record["files"] = download_bundle(session, task_id, raw_dir)
        task_record["downloaded"] = True
        write_json(manifest_path, manifest)

    manifest["complete"] = True
    write_json(manifest_path, manifest)
    print(f"SMAP point-sample download complete: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 500:
        parser.error("--batch-size must be between 1 and 500")
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    try:
        run(args)
    except requests.HTTPError as error:
        response = error.response
        detail = response.text[:1000] if response is not None else str(error)
        print(f"AppEEARS HTTP error: {detail}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
