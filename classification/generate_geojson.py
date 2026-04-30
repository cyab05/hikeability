"""
Reads the latest predictions from hikes-model-output, pulls coordinates
from wta-hikes metadata, and uploads a GeoJSON to hikeability-public-map-data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from google.cloud import storage

# ── Config ───────────────────────────────────────────────────────────────────

GCS_BUCKET_OUTPUT  = "hikes-model-output"
GCS_BUCKET_RAW     = "wta-hikes"
GCS_BUCKET_MAP     = "hikeability-public-map-data"
GCS_PRED_PREFIX    = "predictions"
GCS_INPUT_PREFIX   = "output/hikes"
GCS_MAP_FILE       = "example.geojson"

LABEL_SCORES = {
    "hikeable":   8,
    "modest":     5,
    "unhikeable": 1,
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_latest_predictions(client: storage.Client) -> list[dict]:
    """Read all batch JSON files from the most recent predictions date folder."""
    bucket = client.bucket(GCS_BUCKET_OUTPUT)
    blobs = list(bucket.list_blobs(prefix=GCS_PRED_PREFIX + "/"))
    
    dates = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) >= 3:
            dates.add(parts[1])
    
    if not dates:
        print("No predictions found.")
        return []
    
    latest_date = sorted(dates)[-1]
    print(f"Using predictions from: {latest_date}")
    
    all_predictions = []
    for blob in blobs:
        if f"/{latest_date}/" in blob.name and blob.name.endswith(".json"):
            content = json.loads(blob.download_as_text())
            all_predictions.extend(content)
            print(f"Loaded {len(content)} predictions from {blob.name}")
    
    return all_predictions


def get_metadata(client: storage.Client, hike_id: str) -> dict | None:
    """Pull metadata.json for a hike from wta-hikes bucket."""
    bucket = client.bucket(GCS_BUCKET_RAW)
    blob = bucket.blob(f"{GCS_INPUT_PREFIX}/{hike_id}/metadata.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def predictions_to_geojson(predictions: list[dict], client: storage.Client) -> dict:
    """Convert predictions to GeoJSON, pulling coordinates from metadata."""
    features = []
    seen_hike_ids = set()

    for pred in predictions:
        hike_id = pred.get("hike_id")
        if not hike_id or hike_id in seen_hike_ids:
            continue
        seen_hike_ids.add(hike_id)

        metadata = get_metadata(client, hike_id)
        if not metadata:
            print(f"  -> No metadata for {hike_id}, skipping")
            continue

        lat = metadata.get("latitude")
        lng = metadata.get("longitude")
        if not lat or not lng:
            print(f"  -> No coordinates for {hike_id}, skipping")
            continue

        label = pred.get("predicted_label", "unknown")
        score = LABEL_SCORES.get(label, 5)

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lng), float(lat)]
            },
            "properties": {
                "name": pred.get("hike_name", hike_id),
                "url": metadata.get("url", ""),
                "region": pred.get("hike_region", ""),
                "elevation_gain": pred.get("elevation_gain", ""),
                "highest_point": pred.get("highest_point", ""),
                "rating": metadata.get("rating", ""),
                "hikeability_score": score,
                "reasoning": pred.get("label_explanation", ""),
            }
        }
        features.append(feature)

    print(f"\nTotal features: {len(features)}")
    return {"type": "FeatureCollection", "features": features}


def upload_geojson(client: storage.Client, geojson: dict):
    """Upload GeoJSON to the public map bucket."""
    bucket = client.bucket(GCS_BUCKET_MAP)
    blob = bucket.blob(GCS_MAP_FILE)
    blob.upload_from_string(
        json.dumps(geojson, ensure_ascii=False),
        content_type="application/geo+json"
    )
    print(f"Uploaded -> gs://{GCS_BUCKET_MAP}/{GCS_MAP_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    client = storage.Client()
    
    print("Fetching latest predictions...")
    predictions = get_latest_predictions(client)
    if not predictions:
        return
    
    print(f"\nConverting {len(predictions)} predictions to GeoJSON...")
    geojson = predictions_to_geojson(predictions, client)
    
    print("\nUploading to hikeability-public-map-data...")
    upload_geojson(client, geojson)
    
    print("\nDone! Map data updated.")


if __name__ == "__main__":
    main()
