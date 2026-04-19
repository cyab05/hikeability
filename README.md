# hikeability
USF ML Ops project: a website that allows you to search for hikes based on their likelihood of being hikeable


## Setup

### Create Conda Environment
```bash
conda env create -f environment.yml
conda activate hikeability
gcloud auth application-default login
```

### Or venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gcloud auth application-default login
```

## Tests

To run a specific test file:

```bash
.venv/bin/python -m pytest tests/test_classification.py
.venv/bin/python -m pytest tests/test_weather_forecasts.py
```