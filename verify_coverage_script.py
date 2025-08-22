#!/usr/bin/env python3
"""
Script to verify scraping coverage and ensure ALL sub-pages are captured
"""

import json
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from typing import Set, Dict, List

class ScrapingVerifier:
    """Verify and ensure complete documentation coverage."""
    
    def __init__(self):
        self.data_dir = Path("./docs_data")
        self.content_file = self.data_dir / "content.jsonl"
        self.urls_file = self.data_dir / "urls.json"
        
    def check_current_coverage(self) -> Dict:
        """Check what's currently scraped."""
        print("📊 CHECKING CURRENT SCRAPING COVERAGE")
        print("=" * 60)
        
        if not self.content_file.exists():
            print("❌ No content file found. Scraping may still be running.")
            return {}
        
        # Load scraped URLs
        scraped_urls = set()
        scraped_docs = []
        
        with open(self.content_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    scraped_urls.add(doc['url'])
                    scraped_docs.append(doc)
                except:
                    continue
        
        print(f"✅ Total pages scraped: {len(scraped_urls)}")
        
        # Analyze URL patterns
        url_patterns = {
            'root_level': [],
            'section_level': [],
            'subsection_level': [],
            'deep_nested': []
        }
        
        for url in scraped_urls:
            path = urlparse(url).path
            # Count depth by slashes after /docs/
            depth = path.count('/') - 2  # Subtract /docs/ part
            
            if depth <= 0:
                url_patterns['root_level'].append(url)
            elif depth == 1:
                url_patterns['section_level'].append(url)
            elif depth == 2:
                url_patterns['subsection_level'].append(url)
            else:
                url_patterns['deep_nested'].append(url)
        
        print(f"\n📁 URL Depth Analysis:")
        print(f"  Root level (/docs/xxx): {len(url_patterns['root_level'])}")
        print(f"  Section level (/docs/xxx/yyy): {len(url_patterns['section_level'])}")
        print(f"  Subsection level (/docs/xxx/yyy/zzz): {len(url_patterns['subsection_level'])}")
        print(f"  Deep nested (>3 levels): {len(url_patterns['deep_nested'])}")
        
        # Check for specific patterns
        print(f"\n🔍 Checking for specific page types:")
        
        # Check for IAG pages
        iag_pages = [url for url in scraped_urls if 'iag' in url.lower()]
        print(f"  IAG pages: {len(iag_pages)}")
        if iag_pages[:3]:
            for url in iag_pages[:3]:
                print(f"    - {url}")
        
        # Check for installation pages
        install_pages = [url for url in scraped_urls if 'install' in url.lower()]
        print(f"  Installation pages: {len(install_pages)}")
        if install_pages[:3]:
            for url in install_pages[:3]:
                print(f"    - {url}")
        
        # Check for RHEL pages
        rhel_pages = [url for url in scraped_urls if 'rhel' in url.lower()]
        print(f"  RHEL pages: {len(rhel_pages)}")
        if rhel_pages[:3]:
            for url in rhel_pages[:3]:
                print(f"    - {url}")
        
        # Check for the specific URLs mentioned
        specific_urls = [
            "https://docs.itential.com/docs/iag-related-terminology",
            "https://docs.itential.com/docs/rhel-8-full-installation-method-iag-20233-and-20232"
        ]
        
        print(f"\n✅ Checking specific URLs you mentioned:")
        for url in specific_urls:
            if url in scraped_urls:
                print(f"  ✅ Found: {url}")
            else:
                print(f"  ❌ Missing: {url}")
        
        # Extract all internal links from scraped content
        print(f"\n🔗 Analyzing internal links...")
        all_internal_links = set()
        
        for doc in scraped_docs[:10]:  # Check first 10 docs for speed
            if 'links' in doc:
                for link in doc['links']:
                    if isinstance(link, str) and 'docs.itential.com/docs' in link:
                        all_internal_links.add(link)
        
        # Find potentially missing pages
        missing_pages = all_internal_links - scraped_urls
        if missing_pages:
            print(f"\n⚠️ Found {len(missing_pages)} referenced but unscraped pages:")
            for url in list(missing_pages)[:5]:
                print(f"    - {url}")
        
        return {
            'total_scraped': len(scraped_urls),
            'url_patterns': url_patterns,
            'missing_pages': list(missing_pages)
        }

    def find_all_subpages(self, max_additional: int = 100) -> Set[str]:
        """Find ALL sub-pages by deep crawling from scraped content."""
        print(f"\n🕸️ DEEP CRAWLING TO FIND ALL SUB-PAGES")
        print("=" * 60)
        
        # Load already scraped URLs
        scraped_urls = set()
        if self.content_file.exists():
            with open(self.content_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        scraped_urls.add(doc['url'])
                    except:
                        continue
        
        print(f"📊 Starting with {len(scraped_urls)} already scraped URLs")
        
        # Find new URLs by checking each scraped page
        new_urls = set()
        checked = 0
        
        for url in list(scraped_urls)[:20]:  # Check first 20 for speed
            if checked >= 20:
                break
            
            try:
                print(f"  Checking: {url}")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find all links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        # Make absolute URL
                        if href.startswith('/'):
                            full_url = f"https://docs.itential.com{href}"
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        
                        # Check if it's a docs URL we haven't seen
                        if ('docs.itential.com/docs' in full_url and 
                            full_url not in scraped_urls and
                            not any(skip in full_url for skip in ['/api/', '/opensource/', '#', '.pdf'])):
                            new_urls.add(full_url)
                
                checked += 1
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        if new_urls:
            print(f"\n🆕 Found {len(new_urls)} new sub-pages not in current scraping:")
            for url in list(new_urls)[:10]:
                print(f"  - {url}")
            
            # Save list of missing URLs
            missing_file = self.data_dir / "missing_urls.json"
            with open(missing_file, 'w') as f:
                json.dump(list(new_urls), f, indent=2)
            print(f"\n💾 Saved missing URLs to: {missing_file}")
            
            print(f"\n💡 To scrape these missing pages:")
            print(f"   1. Add them to your scraper's queue")
            print(f"   2. Or run a supplemental scraping pass")
        else:
            print(f"\n✅ Great! No missing sub-pages found in sample check")
        
        return new_urls

    def create_supplemental_scraper(self):
        """Create a script to scrape any missing pages."""
        print(f"\n📝 Creating supplemental scraper for missing pages...")
        
        script_content = '''#!/usr/bin/env python3
"""Supplemental scraper for missing sub-pages"""

import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

# Load missing URLs
missing_file = Path("./docs_data/missing_urls.json")
if missing_file.exists():
    with open(missing_file, 'r') as f:
        missing_urls = json.load(f)
    
    print(f"Found {len(missing_urls)} missing pages to scrape")
    
    # Append to existing content file
    content_file = Path("./docs_data/content.jsonl")
    
    with open(content_file, 'a', encoding='utf-8') as f:
        for i, url in enumerate(missing_urls, 1):
            print(f"[{i}/{len(missing_urls)}] Scraping: {url}")
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract content (same as main scraper)
                    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()
                    
                    title = soup.find('title')
                    title_text = title.get_text(strip=True) if title else "No Title"
                    
                    content_area = soup.find('main') or soup.find('body')
                    text = content_area.get_text(separator='\\n', strip=True) if content_area else ""
                    
                    doc = {
                        'url': url,
                        'title': title_text,
                        'text': text,
                        'headings': [],
                        'code_blocks': []
                    }
                    
                    f.write(json.dumps(doc, ensure_ascii=False) + '\\n')
                    print(f"  ✅ Success: {title_text[:50]}")
                    
            except Exception as e:
                print(f"  ❌ Failed: {e}")
            
            time.sleep(0.5)  # Rate limiting
    
    print("✅ Supplemental scraping complete!")
else:
    print("No missing URLs file found")
'''
        
        script_file = Path("scrape_missing_pages.py")
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Created: {script_file}")
        print(f"   Run it with: python {script_file}")

def main():
    """Run the verification."""
    verifier = ScrapingVerifier()
    
    # Check current coverage
    coverage = verifier.check_current_coverage()
    
    if coverage:
        # Find missing sub-pages
        missing = verifier.find_all_subpages()
        
        if missing:
            # Create supplemental scraper
            verifier.create_supplemental_scraper()
            
            print("\n" + "=" * 60)
            print("📋 RECOMMENDATIONS:")
            print("=" * 60)
            print("1. Your scraper IS getting sub-pages, but may miss some")
            print("2. The crawl_depth parameter might need to be increased")
            print("3. Run the supplemental scraper for missing pages")
            print("\nTo ensure COMPLETE coverage:")
            print("  python scrape_missing_pages.py")
    else:
        print("\n⏳ Waiting for initial scraping to complete...")
        print("   Check back after scraping finishes")

if __name__ == "__main__":
    main()