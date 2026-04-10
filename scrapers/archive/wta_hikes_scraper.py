import requests
from bs4 import BeautifulSoup
import concurrent.futures
import random
import time
import json
import re

class WTAScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        })
        self.base_url = "https://www.wta.org/go-outside/hikes"
        
    def get_hike_links(self, start_index=0):
        search_url = f"{self.base_url}/hike_search?b_start:int={start_index}"
        print(f"Fetching search results page: {search_url}")
        
        response = self.session.get(search_url)
        if response.status_code != 200:
            print(f"Failed to fetch {search_url}. Status Code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        
        page_title = soup.title.get_text(strip=True) if soup.title else 'No title'
        print(f"  -> Page Title detected: '{page_title}'")
        if "Just a moment" in page_title or "Cloudflare" in page_title:
            print("  -> ERROR: The scraper was blocked by Cloudflare anti-bot protection.")
            return []

        hike_links = []
        
        result_items = soup.select('.search-result-item')
        
        if not result_items:
             print("  -> ERROR: Could not find '.search-result-item'. The website HTML layout has changed.")
             return []

        for item in result_items:
            # Look for the title link inside the item
            a_tag = item.select_one('a.listitem-title') or item.select_one('h2 a') or item.select_one('h3 a')
            if a_tag:
                link = a_tag.get('href')
                if link and link not in hike_links:
                    hike_links.append(link)
                
        print(f"  -> Found {len(hike_links)} links on this page.")
        return hike_links

    def scrape_hike_details(self, hike_url):
            print(f"  Scraping details for: {hike_url}")
            response = self.session.get(hike_url)
            if response.status_code != 200:
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
                'trip_reports': []
            }

            def get_stat(label_word):
                # 1. Check for the modern <dt> / <dd> structure you found in the Inspector
                dt_tag = soup.find('dt', string=re.compile(label_word, re.IGNORECASE))
                if dt_tag:
                    dd_tag = dt_tag.find_next_sibling('dd')
                    if dd_tag:
                        return dd_tag.get_text(strip=True)

                # 2. Fallback: Check for the old plain-text structure
                label_node = soup.find(string=re.compile(label_word, re.IGNORECASE))
                if label_node and label_node.parent:
                    parent_text = label_node.parent.get_text(separator=' ', strip=True)
                    clean_text = re.sub(f'{label_word}[:\s]*', '', parent_text, flags=re.IGNORECASE).strip()
                    if not clean_text or clean_text == "":
                        sibling = label_node.parent.find_next_sibling()
                        if sibling:
                            return sibling.get_text(strip=True)
                    return clean_text
                
                return None

            # 1. Hike Name
            title_tag = soup.find('h1', class_='documentFirstHeading')
            if title_tag:
                hike_data['name'] = title_tag.get_text(strip=True)

            # 2. Rating
            rating_tag = soup.find('div', class_='current-rating')
            if rating_tag:
                hike_data['rating'] = rating_tag.get_text(strip=True)

            # 3. Stats (Using the new helper function)
            hike_data['elevation_gain'] = get_stat(r'Elevation Gain') or get_stat(r'Gain')
            hike_data['highest_point'] = get_stat(r'Highest Point')
            
            dist = get_stat(r'Length') or get_stat(r'Distance')
            if dist:
                hike_data['distance'] = re.sub(r'(roundtrip|one-way|of trails).*', r'\1', dist, flags=re.IGNORECASE).strip()

            # 4. Region
            modern_region = soup.select_one('span.wta-icon-headline.h3 .wta-icon-headline__text')
            if modern_region:
                hike_data['region'] = modern_region.get_text(strip=True)
            else:
                old_region_h2 = soup.find('h2', class_='documentFirstHeading')
                if old_region_h2:
                    hike_data['region'] = old_region_h2.get_text(strip=True)
                else:
                    old_region_div = soup.find('div', id='hike-region')
                    if old_region_div: 
                        hike_data['region'] = old_region_div.get_text(separator=' ', strip=True).replace('Region:', '').strip()

            # 5. Latitude / Longitude (Your exact working code)
            coord_container = soup.select_one('.wta-icon-headline__text .h4')
            if coord_container:
                coord_spans = coord_container.find_all('span')
                if len(coord_spans) >= 2:
                    hike_data['latitude'] = coord_spans[0].get_text(strip=True)
                    hike_data['longitude'] = coord_spans[1].get_text(strip=True)

            return hike_data

    def run(self, max_pages=1, max_hikes=None, max_workers=5):
        """
        Runs the scraper using a ThreadPool to process multiple hikes at once.
        """
        all_hikes_data = []
        all_links = []
        
        # Step 1: Gather all the links first
        print("Gathering hike links...")
        for i in range(max_pages):
            start_index = i * 30
            links = self.get_hike_links(start_index)
            all_links.extend(links)
            
            # Stop gathering if we hit our requested limit
            if max_hikes and len(all_links) >= max_hikes:
                all_links = all_links[:max_hikes]
                break
                
            time.sleep(1) # polite scraping
            
        print(f"\nFound {len(all_links)} links. Starting parallel scrape with {max_workers} workers...\n")

        # Scrape the details in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._parallel_scrape_wrapper, url): url for url in all_links}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    if data:
                        all_hikes_data.append(data)
                        
                        if len(all_hikes_data) % 10 == 0:
                            print(f"  -> Progress: {len(all_hikes_data)}/{len(all_links)} hikes completed.")
                except Exception as exc:
                    print(f"  -> ERROR: {url} generated an exception: {exc}")

        # Save to JSON
        with open('wta_hikes.json', 'w', encoding='utf-8') as f:
            json.dump(all_hikes_data, f, indent=4, ensure_ascii=False)
            
        print(f"\nScraping complete! Saved {len(all_hikes_data)} hikes to wta_hikes.json")

    def _parallel_scrape_wrapper(self, url):
        """
        A small wrapper to add a randomized delay before scraping.
        This staggers the parallel requests so you don't look like a bot attack.
        """
        time.sleep(random.uniform(0.5, 2.0))
        return self.scrape_hike_details(url)

if __name__ == "__main__":
    scraper = WTAScraper()
    # Scrape just the first page (30 hikes) for testing. 
    # Change max_pages to a higher number to scrape more of the database.
    scraper.run(max_hikes=10, max_workers=2)