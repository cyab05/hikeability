import requests
from bs4 import BeautifulSoup
import time
import json
import re
import concurrent.futures
import random

class WTATripReportScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self.base_url = "https://www.wta.org"

    def get_report_links(self, start_index=0):
        search_url = f"{self.base_url}/@@search_tripreport_listing?b_size=50&b_start:int={start_index}"
        print(f"Fetching search results page: {search_url}")
        
        try:
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                print(f"  -> Failed to fetch {search_url}. Status Code: {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Bot check
            page_title = soup.title.get_text(strip=True) if soup.title else 'No title'
            if "Just a moment" in page_title or "Cloudflare" in page_title:
                print("  -> ERROR: The scraper was blocked by Cloudflare anti-bot protection.")
                return []

            report_links = []
            
            link_tags = soup.select('.listitem-title a')

            if not link_tags:
                 print("  -> ERROR: Could not find the trip report links.")
                 return []

            for a_tag in link_tags:
                link = a_tag.get('href')
                if link:
                    if link.startswith('/'):
                        link = f"https://www.wta.org{link}"
                    
                    if "trip_report" in link and link not in report_links:
                        report_links.append(link)
                    
            print(f"  -> Found {len(report_links)} links on this page.")
            return report_links

        except requests.exceptions.RequestException as e:
            print(f"  -> Connection Error: {e}")
            return []
    
    def scrape_report_details(self, report_url):
        response = self.session.get(report_url)
        if response.status_code != 200:
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

        title_tag = soup.find('h1', class_='documentFirstHeading')
        if title_tag:
            raw_title = title_tag.get_text(strip=True)
            
            # Extract date from the end of title
            if '—' in raw_title:
                title_parts = raw_title.rsplit('—', 1) 
                report_data['title'] = title_parts[0].strip()
                report_data['date_hiked'] = title_parts[1].strip()
            else:
                report_data['title'] = raw_title
                report_data['date_hiked'] = None

        author_tag = soup.find(class_=re.compile(r'Creator|author', re.IGNORECASE))
        if author_tag:
            report_data['author'] = author_tag.get_text(strip=True).replace('By', '').strip()

        condition_divs = soup.find_all('div', class_='trip-condition')
        for div in condition_divs:
            h4 = div.find('h4')
            span = div.find('span')
            
            if h4 and span:
                label = h4.get_text(strip=True).lower()
                value = span.get_text(strip=True)
                
                if 'type of hike' in label:
                    report_data['type_of_hike'] = value
                elif 'trail' in label:
                    report_data['trail_conditions'] = value
                elif 'road' in label:
                    report_data['road_conditions'] = value
                elif 'bugs' in label:
                    report_data['bugs'] = value
                elif 'snow' in label:
                    report_data['snow'] = value
                    
        # Extract the main report text
        report_body = soup.find('div', id='tripreport-body')
        if report_body:
            paragraphs = report_body.find_all('p')
            text_blocks = [p.get_text(separator=' ', strip=True) for p in paragraphs]
            report_data['report_text'] = '\n\n'.join(filter(None, text_blocks))

        seen_urls = set()
        
        for a_tag in soup.find_all('a', href=re.compile(r'/go-hiking/hikes/')):
            href = a_tag['href']
            if 'hike_search' not in href and href not in seen_urls:
                hike_name = a_tag.get_text(strip=True)
                if hike_name:
                    report_data['associated_hikes'].append({
                        'hike_name': hike_name,
                        'hike_url': href
                    })
                    seen_urls.add(href)

        return report_data

    def _parallel_scrape_wrapper(self, url):
        time.sleep(random.uniform(0.5, 1.5))
        return self.scrape_report_details(url)

    def run(self, max_pages=1, max_workers=5, max_reports=None):
        all_reports_data = []
        all_links = []
        
        print("Gathering recent trip report links...")
        for i in range(max_pages):
            start_index = i * 30
            links = self.get_report_links(start_index)
            
            for link in links:
                if link not in all_links:
                    all_links.append(link)
            
            if max_reports and len(all_links) >= max_reports:
                print(f"Reached the maximum limit of {max_reports} reports during link gathering.")
                all_links = all_links[:max_reports]
                break
                
            time.sleep(1) 
            
        print(f"\nFound {len(all_links)} reports. Starting parallel scrape with {max_workers} workers...\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._parallel_scrape_wrapper, url): url for url in all_links}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    if data:
                        all_reports_data.append(data)
                        print(f"  -> Scraped: {str(data['title'])[:30]}... ({len(all_reports_data)}/{len(all_links)})")
                except Exception as exc:
                    print(f"  -> ERROR: {url} generated an exception: {exc}")

        with open('wta_recent_reports.json', 'w', encoding='utf-8') as f:
            json.dump(all_reports_data, f, indent=4, ensure_ascii=False)
            
        print(f"\nSuccess! Saved {len(all_reports_data)} recent reports to wta_recent_reports.json")

if __name__ == "__main__":
    scraper = WTATripReportScraper()
    scraper.run(max_pages=1, max_workers=2, max_reports=5)