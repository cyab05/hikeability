FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scrapers/wta_daily_scraper.py .

CMD ["python", "wta_daily_scraper.py"]