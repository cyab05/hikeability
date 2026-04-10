import os
import json
import time
import random
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from google.cloud import storage
from collections import defaultdict

class WTADailyScraper:
    def __init__(self, bucket_name, output_prefix="hikes"):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self.base_url = "https://www.wta.org"
        
        # GCS Initialization
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.output_prefix = output_prefix

    def get_recent_report_links(self, max_pages=1):
        """Fetches the most recent trip report links from the main feed."""
        report_links = []
        for i in range(max_pages):
            start_index = i * 50
            search_url = f"{self.base_url}/@@search_tripreport_listing?b_size=50&b_start:int={start_index}"
            print(f"Fetching search results page: {search_url}")
            
            try:
                response = self.session.get(search_url, timeout=10)
                if response.status_code != 200:
                    print(f"  -> Failed to fetch. Status Code: {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Cloudflare Bot Check
                page_title = soup.title.get_text(strip=True) if soup.title else ''
                if "Just a moment" in page_title or "Cloudflare" in page_title:
                    print("  -> ERROR: Blocked by Cloudflare.")
                    break

                link_tags = soup.select('.listitem-title a')
                for a_tag in link_tags:
                    link = a_tag.get('href')
                    if link:
                        if link.startswith('/'):
                            link = f"{self.base_url}{link}"
                        if "trip_report" in link and link not in report_links:
                            report_links.append(link)
                
                time.sleep(1) # Be polite between pages
            except requests.exceptions.RequestException as e:
                print(f"  -> Connection Error: {e}")
                break
                
        return report_links

    def scrape_report_details(self, report_url):
        """Scrapes the individual report and extracts associated hikes."""
        try:
            response = self.session.get(report_url, timeout=10)
            if response.status_code != 200: return None
        except requests.exceptions.RequestException:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        
        report_data = {
            'report_url': report_url,
            'title': None,
            'author': None,
            'date_hiked': None,
            'type_of_hike': None,
            'trail_conditions': None,
            'road_conditions': None,
            'bugs': None,
            'snow': None,
            'report_text': None,
            'associated_hikes': [] 
        }

        # 1. Title & Date
        title_tag = soup.find('h1', class_='documentFirstHeading')
        if title_tag:
            raw_title = title_tag.get_text(strip=True)
            if '—' in raw_title:
                title_parts = raw_title.rsplit('—', 1) 
                report_data['title'] = title_parts[0].strip()
                report_data['date_hiked'] = title_parts[1].strip()
            else:
                report_data['title'] = raw_title

        # 2. Author
        author_tag = soup.find(class_=re.compile(r'Creator|author', re.IGNORECASE))
        if author_tag:
            report_data['author'] = author_tag.get_text(strip=True).replace('By', '').strip()

        # 3. Conditions
        for div in soup.find_all('div', class_='trip-condition'):
            h4, span = div.find('h4'), div.find('span')
            if h4 and span:
                label = h4.get_text(strip=True).lower()
                value = span.get_text(strip=True)
                if 'type of hike' in label: report_data['type_of_hike'] = value
                elif 'trail' in label: report_data['trail_conditions'] = value
                elif 'road' in label: report_data['road_conditions'] = value
                elif 'bugs' in label: report_data['bugs'] = value
                elif 'snow' in label: report_data['snow'] = value
                    
        # 4. Report Text
        report_body = soup.find('div', id='tripreport-body')
        if report_body:
            text_blocks = [p.get_text(separator=' ', strip=True) for p in report_body.find_all('p')]
            report_data['report_text'] = '\n\n'.join(filter(None, text_blocks))

        # 5. Extract Associated Hikes
        seen_urls = set()
        for a_tag in soup.find_all('a', href=re.compile(r'/go-hiking/hikes/')):
            href = a_tag['href']
            if 'hike_search' not in href and href not in seen_urls:
                hike_name = a_tag.get_text(strip=True)
                if hike_name:
                    report_data['associated_hikes'].append({
                        'hike_name': hike_name,
                        'hike_slug': href.rstrip('/').split('/')[-1] # Extract the slug for routing
                    })
                    seen_urls.add(href)

        return report_data

    def _parallel_scrape_wrapper(self, url):
        time.sleep(random.uniform(0.5, 1.5))
        return self.scrape_report_details(url)

    def sort_reports_chronologically(self, reports):
        def parse_date(report):
            date_str = report.get('date_hiked')
            if not date_str: return datetime.min
            try:
                return datetime.strptime(date_str, '%b %d, %Y')
            except ValueError:
                return datetime.min
        return sorted(reports, key=parse_date)

    def update_hike_in_gcs(self, hike_slug, new_reports):
        """Reads existing reports from GCS, appends new ones, sorts, and re-uploads."""
        reports_path = f"{self.output_prefix}/{hike_slug}/reports.jsonl"
        blob = self.bucket.blob(reports_path)
        
        existing_reports = []
        
        # 1. Download existing records if they exist
        if blob.exists():
            content = blob.download_as_text()
            for line in content.strip().split('\n'):
                if line.strip():
                    try:
                        existing_reports.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"      -> New hike detected: {hike_slug}. Creating new directory path.")
            # Note: You may want to trigger metadata scraping here if it's a completely new hike

        # 2. Deduplicate
        existing_urls = {r['report_url'] for r in existing_reports}
        added_count = 0
        
        for report in new_reports:
            # Clean out the 'associated_hikes' key so we don't store redundant data in the JSONL
            clean_report = {k: v for k, v in report.items() if k != 'associated_hikes'}
            
            if clean_report['report_url'] not in existing_urls:
                existing_reports.append(clean_report)
                existing_urls.add(clean_report['report_url'])
                added_count += 1

        # 3. Upload back to GCS if new data was added
        if added_count > 0:
            sorted_reports = self.sort_reports_chronologically(existing_reports)
            jsonl_str = '\n'.join([json.dumps(r, ensure_ascii=False) for r in sorted_reports]) + '\n'
            blob.upload_from_string(jsonl_str, content_type='application/jsonl')
            print(f"      -> {hike_slug}: Added {added_count} new report(s). Total: {len(sorted_reports)}")
        else:
            print(f"      -> {hike_slug}: No new reports (already up to date).")

    def run(self, max_pages=1, max_workers=5):
        print(f"Starting Daily Run. Fetching top {max_pages * 50} recent reports...")
        
        # 1. Gather Links
        report_links = self.get_recent_report_links(max_pages=max_pages)
        print(f"Found {len(report_links)} recent trip reports. Initiating parallel scrape...")

        # 2. Scrape Reports
        all_reports_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._parallel_scrape_wrapper, url): url for url in report_links}
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    data = future.result()
                    if data: all_reports_data.append(data)
                except Exception as exc:
                    print(f"  -> ERROR during parallel scrape: {exc}")

        # 3. Group Reports by Hike
        print("\nGrouping reports by associated hike...")
        reports_by_hike = defaultdict(list)
        
        for report in all_reports_data:
            if not report.get('associated_hikes'):
                print(f"  -> Skipping report (no associated hikes found): {report['report_url']}")
                continue
                
            for hike in report['associated_hikes']:
                reports_by_hike[hike['hike_slug']].append(report)

        print(f"Found updates for {len(reports_by_hike)} distinct hikes. Syncing to GCS...")

        # 4. Route and Upload to GCS
        for hike_slug, reports in reports_by_hike.items():
            self.update_hike_in_gcs(hike_slug, reports)
            
        print("\nDaily update complete.")


if __name__ == "__main__":
    TARGET_BUCKET = "wta-hikes" 
    
    scraper = WTADailyScraper(bucket_name=TARGET_BUCKET, output_prefix="output/hikes")
    # Setting max_pages to 2 will fetch the 100 most recent trip reports on WTA
    scraper.run(max_pages=2, max_workers=5)