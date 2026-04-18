# Weather scraper

Grab hourly forecast throghout the next 24 hrs for all scraped trails and saves to GCS.

Variables:
- apparent temperature
- snowfall
- snow depth
- US AQI

## Setup

```bash
pip install -r weather_forecasts/requirements.txt
gcloud auth application-default login
```

## Run it

```bash
python trail_forecasts_gcs.py --workers 20
```

# Testing

```bash
pytest tests/test_weather_forecasts.py
```