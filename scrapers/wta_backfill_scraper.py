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

# Validators for scraped stat fields. Mirror the patterns used in app/gcs.py.
# We apply them at scrape-time so junk (e.g. the hike description) can't make
# it into metadata.json when WTA's stats block is missing/replaced by a closure note.
_VALID_FEET     = re.compile(r"^[\d,]+(\s*(feet|ft))?\.?$", re.IGNORECASE)
_VALID_DISTANCE = re.compile(r"^[\d.,]+\s*miles?(\s*,?\s*(roundtrip|one-way|of trails))?\.?$", re.IGNORECASE)


def _clean_stat(value, validator):
    """Return the value if it matches the validator regex, else None."""
    if value is None:
        return None
    s = str(value).strip()
    return s if validator.match(s) else None


class WTABackfillScraper:
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

    def get_hike_links(self, start_index=0):
        search_url = f"{self.base_url}/go-outside/hikes/hike_search?b_start:int={start_index}"
        try:
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            page_title = soup.title.get_text(strip=True) if soup.title else ''
            
            if "Just a moment" in page_title or "Cloudflare" in page_title:
                print("  -> ERROR: Blocked by Cloudflare.")
                return []

            hike_links = []
            for item in soup.select('.search-result-item'):
                a_tag = item.select_one('a.listitem-title') or item.select_one('h2 a') or item.select_one('h3 a')
                if a_tag and a_tag.get('href') not in hike_links:
                    hike_links.append(a_tag.get('href'))
                    
            return hike_links
        except requests.exceptions.RequestException:
            return []

    def scrape_hike_details(self, hike_url):
        try:
            response = self.session.get(hike_url, timeout=10)
            if response.status_code != 200:
                return None
        except requests.exceptions.RequestException:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        hike_data = {
            'url': hike_url,
            'name': None,
            'latitude': None,
            'longitude': None,
            'region': None,
            'distance': None,
            'elevation_gain': None,
            'highest_point': None,
            'rating': None,
            'image_url': None,
            'difficulty': None,
            'parking_pass': None,
            'closure_warning': [],
        }

        def get_stat(label_word):
            dt_tag = soup.find('dt', string=re.compile(label_word, re.IGNORECASE))
            if dt_tag and dt_tag.find_next_sibling('dd'):
                return dt_tag.find_next_sibling('dd').get_text(strip=True)

            label_node = soup.find(string=re.compile(label_word, re.IGNORECASE))
            if label_node and label_node.parent:
                parent_text = label_node.parent.get_text(separator=' ', strip=True)
                clean_text = re.sub(f'{label_word}[:\s]*', '', parent_text, flags=re.IGNORECASE).strip()
                if not clean_text:
                    sibling = label_node.parent.find_next_sibling()
                    if sibling: return sibling.get_text(strip=True)
                return clean_text
            return None

        title_tag = soup.find('h1', class_='documentFirstHeading')
        if title_tag: hike_data['name'] = title_tag.get_text(strip=True)

        rating_tag = soup.find('div', class_='current-rating')
        if rating_tag: hike_data['rating'] = rating_tag.get_text(strip=True)

        # Validate each stat against its expected shape so the loose page-wide
        # regex fallback in get_stat() can't smuggle in description prose.
        hike_data['elevation_gain'] = _clean_stat(
            get_stat(r'Elevation Gain') or get_stat(r'Gain'), _VALID_FEET
        )
        hike_data['highest_point'] = _clean_stat(get_stat(r'Highest Point'), _VALID_FEET)

        dist = get_stat(r'Length') or get_stat(r'Distance')
        if dist:
            cleaned = re.sub(r'(roundtrip|one-way|of trails).*', r'\1', dist, flags=re.IGNORECASE).strip()
            hike_data['distance'] = _clean_stat(cleaned, _VALID_DISTANCE)

        modern_region = soup.select_one('span.wta-icon-headline.h3 .wta-icon-headline__text')
        if modern_region: hike_data['region'] = modern_region.get_text(strip=True)

        coord_container = soup.select_one('.wta-icon-headline__text .h4')
        if coord_container:
            coord_spans = coord_container.find_all('span')
            if len(coord_spans) >= 2:
                hike_data['latitude'] = coord_spans[0].get_text(strip=True)
                hike_data['longitude'] = coord_spans[1].get_text(strip=True)

        # Image URL — prefer Open Graph tag (stable across template versions)
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            raw_url = og_image['content']
            image_url = raw_url if raw_url.startswith('http') else f"{self.base_url}{raw_url}"
            # Strip cache-busting query strings (e.g. ?width=...)
            hike_data['image_url'] = image_url.split('?')[0]
        else:
            img = soup.select_one('.hero-photo img, #hero-photo img, .hike-hero img')
            if img and img.get('src'):
                src = img['src']
                src = src if src.startswith('http') else f"{self.base_url}{src}"
                hike_data['image_url'] = src.split('?')[0]

        # Calculated difficulty — WTA pill inside the last stats row
        difficulty_pill = soup.select_one('.hike-stats__stat--last-row .wta-pill')
        if difficulty_pill:
            hike_data['difficulty'] = difficulty_pill.get_text(strip=True)

        # Parking pass / entry fee — inside an .alert block
        for alert in soup.select('div.alert'):
            h4 = alert.find('h4')
            if not h4 or 'Parking Pass' not in h4.get_text():
                continue
            a = alert.find('a')
            if a:
                href = a.get('href', '')
                if href.startswith('/'):
                    href = f"{self.base_url}{href}"
                hike_data['parking_pass'] = {'name': a.get_text(strip=True), 'url': href}
            else:
                text = alert.get_text(separator=' ', strip=True).replace(h4.get_text(strip=True), '').strip()
                hike_data['parking_pass'] = {'name': text or None, 'url': None}
            break

        # Closure / advisory notes — div.wta-note variants. Capture every
        # severity (red, orange, yellow, etc.) so the frontend can color them.
        hike_data['closure_warning'] = self._extract_notes(soup)

        return hike_data

    # WTA uses .wta-note--small as a size variant for non-warning widgets
    # (e.g. trip-report banners). We only treat color modifiers as real
    # advisory severities so unrelated notes don't pollute the metadata.
    _NOTE_SEVERITIES = {'red', 'orange', 'yellow', 'blue', 'green'}

    def _extract_notes(self, soup):
        """Extract wta-note advisories whose modifier class is a recognized
        color severity, as a list of {severity, message} dicts. Deduplicated
        by message. Returns [] when none are present so we can distinguish
        'checked, no warnings' from 'never checked' in metadata."""
        notes = []
        seen = set()
        for note_div in soup.select('div[class*="wta-note"]'):
            text_el = note_div.select_one('.wta-note__text')
            if not text_el:
                continue
            msg = text_el.get_text(separator=' ', strip=True).strip('"').strip()
            if not msg or msg in seen:
                continue
            severity = None
            for cls in note_div.get('class', []):
                if cls.startswith('wta-note--'):
                    severity = cls.replace('wta-note--', '')
                    break
            if severity not in self._NOTE_SEVERITIES:
                continue
            seen.add(msg)
            notes.append({'severity': severity, 'message': msg})
        return notes

    def get_trip_reports_for_hike(self, hike_url, max_reports=None):
        report_links = []
        start_index = 0
        clean_hike_url = hike_url.rstrip('/')
        
        while True:
            search_url = f"{clean_hike_url}/@@related_tripreport_listing?b_start:int={start_index}"
            print(f"    -> Fetching reports from: {search_url}")
            try:
                response = self.session.get(search_url, timeout=10)
                if response.status_code != 200:
                    print(f"      -> Failed to fetch. Status Code: {response.status_code}")
                    break
                soup = BeautifulSoup(response.text, 'html.parser')
                
                page_title = soup.title.get_text(strip=True) if soup.title else ''
                if "Just a moment" in page_title or "Cloudflare" in page_title:
                    print("      -> ERROR: Blocked by Cloudflare anti-bot protection.")
                    break
                
                link_tags = soup.select('.listitem-title a')
                if not link_tags:
                    print("      -> No more report links found. Ending pagination.")
                    break
                new_links_found = False
                for a_tag in link_tags:
                    link = a_tag.get('href')
                    if link:
                        if link.startswith('/'):
                            link = f"{self.base_url}{link}"
                        
                        if "trip_report" in link and link not in report_links:
                            report_links.append(link)
                            new_links_found = True
                
                if not new_links_found:
                    print("      -> No new unique links found. Ending pagination.")
                    break
                    
                print(f"      -> Added {len(link_tags)} links. Total so far: {len(report_links)}")
                
                if max_reports and len(report_links) >= max_reports:
                    break
                
                start_index += 5
                time.sleep(random.uniform(0.5, 1.0))
                
            except requests.exceptions.RequestException as e:
                print(f"      -> Connection Error: {e}")
                break
                
        return report_links[:max_reports] if max_reports else report_links

    def scrape_report_details(self, report_url):
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
            'author_badges': [],
            'date_hiked': None,
            'likes': 0,
            'type_of_hike': None,
            'trail_conditions': None,
            'road_conditions': None,
            'bugs': None,
            'snow': None,
            'report_text': None
        }

        title_tag = soup.find('h1', class_='documentFirstHeading')
        if title_tag:
            raw_title = title_tag.get_text(strip=True)
            if '—' in raw_title:
                title_parts = raw_title.rsplit('—', 1) 
                report_data['title'] = title_parts[0].strip()
                report_data['date_hiked'] = title_parts[1].strip()
            else:
                report_data['title'] = raw_title

        author_span = soup.find('span', itemprop='author')
        if author_span:
            report_data['author'] = author_span.get_text(strip=True)

        badges_container = soup.find('div', class_='Badges')
        if badges_container:
            for badge in badges_container.find_all('div', class_=re.compile(r'Badge')):
                badge_title = badge.get('data-title')
                if badge_title:
                    report_data['author_badges'].append(badge_title)

        for div in soup.find_all('div', class_='trip-condition'):
            h4, span = div.find('h4'), div.find('span')
            if h4 and span:
                label, value = h4.get_text(strip=True).lower(), span.get_text(strip=True)
                if 'type of hike' in label: report_data['type_of_hike'] = value
                elif 'trail' in label: report_data['trail_conditions'] = value
                elif 'road' in label: report_data['road_conditions'] = value
                elif 'bugs' in label: report_data['bugs'] = value
                elif 'snow' in label: report_data['snow'] = value
        
        likes_tag = soup.select_one('.total-thumbs-up .tally-total')
        if likes_tag:
            raw_likes = likes_tag.get_text(strip=True)
            try:
                report_data['likes'] = int(raw_likes)
            except ValueError:
                pass
                    
        report_body = soup.find('div', id='tripreport-body')
        if report_body:
            text_blocks = [p.get_text(separator=' ', strip=True) for p in report_body.find_all('p')]
            report_data['report_text'] = '\n\n'.join(filter(None, text_blocks))

        return report_data

    def _parallel_report_scraper(self, url):
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

    def process_single_hike(self, hike_url, max_reports_per_hike=None, max_workers=5):
        hike_slug = hike_url.strip('/').split('/')[-1]
        
        # Define GCS Paths
        metadata_path = f"{self.output_prefix}/{hike_slug}/metadata.json"
        reports_path = f"{self.output_prefix}/{hike_slug}/reports.jsonl"
        
        metadata_blob = self.bucket.blob(metadata_path)
        reports_blob = self.bucket.blob(reports_path)
        
        # Check if already processed
        if metadata_blob.exists() and reports_blob.exists():
            print(f"  -> Skipping '{hike_slug}': Already processed in GCS bucket.")
            return

        print(f"\nProcessing Hike: {hike_slug}")
        
        # 1. Scrape Metadata
        metadata = self.scrape_hike_details(hike_url)
        if not metadata:
            print(f"  -> ERROR: Could not fetch metadata for {hike_slug}")
            return
            
        # 2. Scrape Reports
        report_links = self.get_trip_reports_for_hike(hike_url, max_reports=max_reports_per_hike)
        print(f"  -> Found {len(report_links)} trip reports. Initiating parallel download...")

        reports_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._parallel_report_scraper, url): url for url in report_links}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    if data: 
                        reports_data.append(data)
                    else:
                        print(f"      -> Warning: Scraper returned None for {url}")
                except Exception as exc:
                    print(f"      -> ERROR: Thread crashed on {url} - {exc}")

        sorted_reports = self.sort_reports_chronologically(reports_data)
        
        # 3. Upload to GCS
        try:
            # Upload Metadata
            metadata_str = json.dumps(metadata, indent=4, ensure_ascii=False)
            metadata_blob.upload_from_string(metadata_str, content_type='application/json')
            
            # Upload Reports as JSONL
            if sorted_reports:
                jsonl_str = '\n'.join([json.dumps(report, ensure_ascii=False) for report in sorted_reports]) + '\n'
                reports_blob.upload_from_string(jsonl_str, content_type='application/jsonl')
            else:
                reports_blob.upload_from_string("", content_type='application/jsonl')
                
            print(f"  -> Success: Uploaded metadata and {len(sorted_reports)} reports to gs://{self.bucket.name}/{self.output_prefix}/{hike_slug}/")
        
        except Exception as e:
            print(f"  -> ERROR uploading to GCS: {e}")

    def backfill_metadata_fields(self, max_workers=10):
        """One-off pass to populate image_url, difficulty, parking_pass, and
        closure_warning on existing metadata.json files, AND to clean up
        bad elevation_gain/distance/highest_point values that the old scraper
        wrote when a closure note replaced the structured stats block.

        Skip predicate (returns True = skip): all expected keys are present,
        and any stat values that exist match their validator regex.
        """
        print("Listing all metadata.json blobs in GCS...")
        blobs = list(self.bucket.list_blobs(prefix=f"{self.output_prefix}/"))
        meta_blobs = [b for b in blobs if b.name.endswith('/metadata.json')]
        print(f"Found {len(meta_blobs)} metadata.json files to inspect.")

        def _needs_recheck(meta):
            if not meta.get('image_url'):              return True
            if not meta.get('difficulty'):             return True
            if 'parking_pass'    not in meta:          return True
            if 'closure_warning' not in meta:          return True
            for field, validator in (('elevation_gain', _VALID_FEET),
                                     ('distance',       _VALID_DISTANCE),
                                     ('highest_point',  _VALID_FEET)):
                v = meta.get(field)
                if v and not validator.match(str(v).strip()):
                    return True
            return False

        def _process_blob(blob):
            try:
                meta = json.loads(blob.download_as_text())
            except Exception as e:
                print(f"  -> ERROR reading {blob.name}: {e}")
                return

            if not _needs_recheck(meta):
                return

            hike_url = meta.get('url')
            if not hike_url:
                print(f"  -> SKIP {blob.name}: no url in metadata")
                return

            time.sleep(random.uniform(0.5, 1.5))
            try:
                response = self.session.get(hike_url, timeout=10)
                if response.status_code != 200:
                    print(f"  -> SKIP {blob.name}: HTTP {response.status_code}")
                    return
            except Exception as e:
                print(f"  -> ERROR fetching {hike_url}: {e}")
                return

            soup = BeautifulSoup(response.text, 'html.parser')
            page_title = soup.title.get_text(strip=True) if soup.title else ''
            if 'Just a moment' in page_title or 'Cloudflare' in page_title:
                print(f"  -> BLOCKED by Cloudflare for {hike_url}")
                return

            updated = False

            if not meta.get('image_url'):
                og_image = soup.find('meta', property='og:image')
                if og_image and og_image.get('content'):
                    raw = og_image['content']
                    url = raw if raw.startswith('http') else f"{self.base_url}{raw}"
                    meta['image_url'] = url.split('?')[0]
                    updated = True
                else:
                    img = soup.select_one('.hero-photo img, #hero-photo img, .hike-hero img')
                    if img and img.get('src'):
                        src = img['src']
                        src = src if src.startswith('http') else f"{self.base_url}{src}"
                        meta['image_url'] = src.split('?')[0]
                        updated = True

            if not meta.get('difficulty'):
                pill = soup.select_one('.hike-stats__stat--last-row .wta-pill')
                if pill:
                    meta['difficulty'] = pill.get_text(strip=True)
                    updated = True

            if meta.get('parking_pass') is None:
                for alert in soup.select('div.alert'):
                    h4 = alert.find('h4')
                    if not h4 or 'Parking Pass' not in h4.get_text():
                        continue
                    a = alert.find('a')
                    if a:
                        href = a.get('href', '')
                        if href.startswith('/'):
                            href = f"{self.base_url}{href}"
                        meta['parking_pass'] = {'name': a.get_text(strip=True), 'url': href}
                    else:
                        text = alert.get_text(separator=' ', strip=True).replace(h4.get_text(strip=True), '').strip()
                        meta['parking_pass'] = {'name': text or None, 'url': None}
                    updated = True
                    break

            # Closure warnings — write whenever the field is missing OR the
            # set of notes on the live page differs from what we have stored.
            fresh_notes = self._extract_notes(soup)
            if 'closure_warning' not in meta or meta.get('closure_warning') != fresh_notes:
                meta['closure_warning'] = fresh_notes
                updated = True

            # Overwrite stale bad stat values left by the old scraper bug.
            for field, validator in (('elevation_gain', _VALID_FEET),
                                     ('highest_point',  _VALID_FEET)):
                v = meta.get(field)
                if v and not validator.match(str(v).strip()):
                    meta[field] = None
                    updated = True
            v = meta.get('distance')
            if v and not _VALID_DISTANCE.match(str(v).strip()):
                meta['distance'] = None
                updated = True

            if updated:
                try:
                    blob.upload_from_string(
                        json.dumps(meta, indent=4, ensure_ascii=False),
                        content_type='application/json'
                    )
                    print(f"  -> Updated {blob.name}")
                except Exception as e:
                    print(f"  -> ERROR uploading {blob.name}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_process_blob, meta_blobs))

        print("Backfill complete.")

    def run(self, max_hikes=None, max_reports_per_hike=20, max_workers=5):
        """Coordinates the batch run."""
        print("Gathering initial hike links from directory...")
        hike_links = []
        start_index = 0
        
        while True:
            print(f"  -> Fetching search directory page at index {start_index}...")
            links = self.get_hike_links(start_index=start_index)
            
            # End of site
            if not links:
                print("  -> No more hikes found. Reached the end of the directory.")
                break
                
            hike_links.extend(links)
            
            if max_hikes and len(hike_links) >= max_hikes:
                hike_links = hike_links[:max_hikes]
                break
                
            # Increment by 30 for the next page
            start_index += 30
            time.sleep(1) # Be polite to the server between page requests
            
        print(f"\nFound {len(hike_links)} hikes to process.")
        print("-" * 40)

        for idx, hike_url in enumerate(hike_links, 1):
            print(f"[{idx}/{len(hike_links)}] Status Update")
            self.process_single_hike(hike_url, max_reports_per_hike, max_workers)
            time.sleep(0.5)

if __name__ == "__main__":
    import sys
    # set GOOGLE_APPLICATION_CREDENTIALS to service account file in environment before running
    TARGET_BUCKET = "wta-hikes"

    scraper = WTABackfillScraper(bucket_name=TARGET_BUCKET, output_prefix="output/hikes")

    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        scraper.backfill_metadata_fields(max_workers=5)
    else:
        scraper.run(max_hikes=None, max_reports_per_hike=10, max_workers=3)