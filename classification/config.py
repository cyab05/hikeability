"""
Shared configuration for trail condition classification pipeline.
"""

# ── NVIDIA / LLM ─────────────────────────────────────────────────────────────

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL_ID        = "nvidia/llama-3.1-nemotron-nano-8b-v1"

# ── Labels ───────────────────────────────────────────────────────────────────

LABELS = ["hikeable", "modest", "unhikeable"]

# Map training CSV labels -> LLM labels
CSV_LABEL_MAP = {
    "hikeable":         "hikeable",
    "modest_conditions": "modest",
    "not_hikeable":     "unhikeable",
}

# ── Strategy (winner from eval) ──────────────────────────────────────────────

STRATEGY    = "one_shot"
PER_LABEL   = 1  # one-shot = 1 example per label

# ── MLflow ───────────────────────────────────────────────────────────────────

MLFLOW_TRACKING   = "http://35.202.68.106:5000/"
MLFLOW_EXPERIMENT = "trail-condition-llm"

# ── GCS Buckets & Prefixes ───────────────────────────────────────────────────

GCS_BUCKET_RAW     = "wta-hikes"
GCS_BUCKET_OUTPUT  = "hikes-model-output"
GCS_BUCKET_WEATHER = "weather-conditions"

GCS_INPUT_PREFIX   = "output/hikes"
GCS_PRED_PREFIX    = "predictions"
GCS_WEATHER_PREFIX = "weather-scraped"
GCS_ARTIF_PREFIX   = "artifacts/mlflow"

# ── Report Freshness ─────────────────────────────────────────────────────────

REPORT_MAX_AGE_DAYS = 30

# ── Batch Size ───────────────────────────────────────────────────────────────

DEFAULT_BATCH_SIZE = 500

# ── API Rate Limiting ────────────────────────────────────────────────────────

API_DELAY_SECONDS = 1.5

# ── Training Data ────────────────────────────────────────────────────────────

TRAINING_CSV_PATH = "model_sandbox/wta_training_data.csv"

# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert trail conditions analyst. You read hike trip reports and 
classify the overall trail conditions into exactly one of three categories:

  - hikeable    — Trail is in good condition. Safe and enjoyable for most hikers.
  - modest      — Trail is passable but has notable challenges (e.g. patchy snow, 
    mud, downed trees, some route-finding). Experienced hikers can manage with 
    proper gear.
  - unhikeable  — Trail is closed, unsafe, or impassable. Hazards include deep 
    snow, washed-out sections, closed roads, extreme weather, or explicit warnings.

You will receive some combination of: a hiker's trip report, current weather 
conditions at the trailhead, trail region, and trail elevation data. Not all data will always 
be present.

Important Considerations:
  - Current weather can override older reports. A clear report from 2 weeks ago 
    is unreliable if current weather shows heavy snowfall or storms.
  - Trail elevation is critical in the Pacific Northwest. Trails above 4,000 feet 
    often retain snow into June or July. A sunny day at the trailhead does NOT 
    mean the summit is clear. If a trail gains 2,000+ feet and tops out above 
    5,000 feet, assume snow is likely from October through June unless a recent 
    report says otherwise.
  - Consider the region. Alpine areas (North Cascades, Mount Rainier, Olympic 
    high country) hold snow longer than lowland trails (Puget Sound, San Juan 
    Islands, urban parks).
  - Your explanation will be shown to hikers on a website.
  - Plain language. Don't use "we..."

Respond ONLY with a JSON object — no markdown, no extra text:
{"label": "<hikeable | modest | unhikeable>", "explanation": "<1-2 sentences why>"}
"""