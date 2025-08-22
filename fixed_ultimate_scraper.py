#!/usr/bin/env python3
"""
Fixed Ultimate Documentation Scraper with Better URL Discovery
Includes debugging and fallback mechanisms
"""

import asyncio
import aiohttp
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Set, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from datetime import datetime
import re
from collections import defaultdict

class FixedDocumentationScraper:
    """Fixed scraper with better URL discovery and debugging."""
    
    def __init__(self, base_url: str = "https://docs.itential.com/docs/"):
        self.base_url = base_url
        self.data_dir = Path("./docs_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage
        self.content_db = self.data_dir / "content.jsonl"
        self.url_list = self.data_dir / "urls.json"
        
        # Tracking
        self.discovered_urls: Set[str] = set()
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        
        # Debug mode
        self.debug = True

    def test_connection(self) -> bool:
        """Test if we can connect to the documentation site."""
        print("🔌 Testing connection to docs.itential.com...")
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Connection successful! Status: {response.status_code}")
                print(f"📄 Page size: {len(response.content)} bytes")
                return True
            else:
                print(f"⚠️ Unexpected status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def discover_urls_simple(self) -> Set[str]:
        """Simple synchronous URL discovery with debugging."""
        print("\n🔍 Starting simple URL discovery...")
        urls = set()
        
        # Start with base URL
        print(f"📍 Starting from: {self.base_url}")
        
        try:
            # Get the main docs page
            response = requests.get(self.base_url, timeout=10)
            if response.status_code != 200:
                print(f"❌ Failed to fetch base URL: {response.status_code}")
                return urls
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Show page title
            title = soup.find('title')
            if title:
                print(f"📄 Page title: {title.get_text().strip()}")
            
            # Method 1: Find all links
            all_links = soup.find_all('a', href=True)
            print(f"🔗 Found {len(all_links)} total links on page")
            
            for link in all_links:
                href = link['href']
                # Make absolute URL
                full_url = urljoin(self.base_url, href)
                
                # Check if it's a docs URL
                if 'docs.itential.com/docs' in full_url:
                    # Clean the URL
                    parsed = urlparse(full_url)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    
                    # Skip API and opensource
                    if not any(skip in clean_url for skip in ['/api/', '/opensource/', '#', '.pdf', '.zip']):
                        urls.add(clean_url)
                        if self.debug and len(urls) <= 10:
                            print(f"  ✅ Found: {clean_url}")
            
            print(f"📊 Discovered {len(urls)} documentation URLs from base page")
            
            # Method 2: Try common documentation paths
            common_paths = [
                "getting-started",
                "installation",
                "configuration",
                "administration",
                "troubleshooting",
                "platform-6",
                "platform-6-install",
                "platform-6-configuration",
                "iap-2023-1",
                "iap-2023-2", 
                "iap-2024-1",
                "iag-2023-1",
                "iag-2023-2",
                "iag-2024-1",
                "release-notes",
                "cli",
                "api-reference",
                "integrations",
                "adapters",
                "migration",
                "upgrade",
                "requirements",
                "prerequisites"
            ]
            
            print(f"\n🎯 Checking {len(common_paths)} common documentation paths...")
            for path in common_paths:
                test_url = f"{self.base_url}{path}"
                try:
                    resp = requests.head(test_url, timeout=5, allow_redirects=True)
                    if resp.status_code == 200:
                        urls.add(test_url)
                        print(f"  ✅ Valid: {path}")
                    elif self.debug:
                        print(f"  ❌ Not found: {path} ({resp.status_code})")
                except:
                    pass
            
            print(f"\n📊 Total URLs discovered: {len(urls)}")
            
        except Exception as e:
            print(f"❌ Error during URL discovery: {e}")
        
        return urls

    def crawl_depth(self, start_urls: Set[str], max_depth: int = 3) -> Set[str]:
        """Crawl to discover more URLs with depth limit."""
        print(f"\n🕸️ Deep crawling from {len(start_urls)} seed URLs (max depth: {max_depth})...")
        
        all_urls = set(start_urls)
        visited = set()
        queue = [(url, 0) for url in start_urls]
        
        while queue:
            url, depth = queue.pop(0)
            
            if url in visited or depth > max_depth:
                continue
            
            visited.add(url)
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    
                    # Clean URL
                    parsed = urlparse(full_url)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    
                    # Check if valid docs URL
                    if ('docs.itential.com/docs' in clean_url and 
                        clean_url not in all_urls and
                        not any(skip in clean_url for skip in ['/api/', '/opensource/', '#', '.pdf'])):
                        
                        all_urls.add(clean_url)
                        if depth < max_depth:
                            queue.append((clean_url, depth + 1))
                        
                        if len(all_urls) % 10 == 0:
                            print(f"  📈 Discovered {len(all_urls)} URLs so far...")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ Error crawling {url}: {e}")
                continue
        
        print(f"✅ Deep crawl complete. Total URLs: {len(all_urls)}")
        return all_urls

    def scrape_content(self, url: str) -> Optional[Dict]:
        """Scrape content from a single URL."""
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove non-content
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            # Get title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "No Title"
            
            # Find main content
            content_area = None
            for selector in ['main', '.main-content', '.content', 'article', '#content', '.doc-content']:
                content_area = soup.select_one(selector)
                if content_area:
                    break
            
            if not content_area:
                content_area = soup.find('body')
            
            if not content_area:
                return None
            
            # Extract content
            content = {
                'url': url,
                'title': title_text,
                'text': content_area.get_text(separator='\n', strip=True),
                'headings': [],
                'code_blocks': [],
                'tables': [],
                'links': []
            }
            
            # Extract headings
            for h in content_area.find_all(['h1', 'h2', 'h3', 'h4']):
                heading_text = h.get_text(strip=True)
                if heading_text:
                    content['headings'].append({
                        'level': h.name,
                        'text': heading_text
                    })
            
            # Extract code blocks
            for code in content_area.find_all(['pre', 'code']):
                code_text = code.get_text(strip=True)
                if code_text and len(code_text) > 20:
                    content['code_blocks'].append(code_text[:500])  # Limit size
            
            # Extract internal links
            for a in content_area.find_all('a', href=True):
                href = a['href']
                if '/docs/' in href or href.startswith('/'):
                    full_url = urljoin(url, href)
                    content['links'].append(full_url)
            
            return content
            
        except Exception as e:
            print(f"  ❌ Error scraping {url}: {e}")
            return None

    def run_complete_scraping(self):
        """Run the complete scraping process."""
        print("🚀 Starting Fixed Documentation Scraping")
        print("=" * 60)
        
        # Test connection first
        if not self.test_connection():
            print("❌ Cannot connect to documentation site. Please check your internet connection.")
            return
        
        # Discover URLs
        print("\n📍 Phase 1: URL Discovery")
        initial_urls = self.discover_urls_simple()
        
        if not initial_urls:
            print("❌ No URLs discovered from initial page. Site structure may have changed.")
            print("\n🔧 Trying alternative approach...")
            
            # Fallback: Manual seed URLs
            initial_urls = {
                f"{self.base_url}",
                f"{self.base_url}getting-started",
                f"{self.base_url}platform-6",
                f"{self.base_url}installation"
            }
            print(f"📌 Using {len(initial_urls)} seed URLs")
        
        # Deep crawl for more URLs
        print("\n📍 Phase 2: Deep Crawling")
        all_urls = self.crawl_depth(initial_urls, max_depth=5)
        
        # Save URL list
        with open(self.url_list, 'w') as f:
            json.dump(list(all_urls), f, indent=2)
        print(f"💾 Saved {len(all_urls)} URLs to {self.url_list}")
        
        # Scrape content
        print(f"\n📍 Phase 3: Content Scraping")
        print(f"📥 Scraping {len(all_urls)} pages...")
        
        scraped_count = 0
        failed_count = 0
        
        with open(self.content_db, 'w', encoding='utf-8') as f:
            for i, url in enumerate(all_urls, 1):
                print(f"\n[{i}/{len(all_urls)}] Scraping: {url}")
                
                content = self.scrape_content(url)
                
                if content:
                    f.write(json.dumps(content, ensure_ascii=False) + '\n')
                    scraped_count += 1
                    print(f"  ✅ Success: {content['title'][:50]}...")
                else:
                    failed_count += 1
                    print(f"  ❌ Failed")
                
                # Progress update
                if i % 10 == 0:
                    print(f"\n📊 Progress: {scraped_count} succeeded, {failed_count} failed")
                
                # Rate limiting
                time.sleep(0.5)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 SCRAPING COMPLETE")
        print("=" * 60)
        print(f"✅ Successfully scraped: {scraped_count} pages")
        print(f"❌ Failed: {failed_count} pages")
        print(f"📁 Data saved to: {self.data_dir}")
        print(f"📄 Content file: {self.content_db}")
        print(f"🔗 URL list: {self.url_list}")

def test_scraped_data():
    """Test the scraped data."""
    data_dir = Path("./docs_data")
    content_file = data_dir / "content.jsonl"
    
    if not content_file.exists():
        print("❌ No content file found. Run scraping first.")
        return
    
    print("\n🧪 Testing scraped data...")
    
    # Load and analyze
    documents = []
    with open(content_file, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    
    print(f"📄 Loaded {len(documents)} documents")
    
    # Show sample
    if documents:
        print("\n📋 Sample documents:")
        for doc in documents[:5]:
            print(f"  - {doc['title'][:60]}...")
            print(f"    URL: {doc['url']}")
            print(f"    Text length: {len(doc['text'])} chars")
            print(f"    Headings: {len(doc['headings'])}")
            print(f"    Code blocks: {len(doc['code_blocks'])}")

def main():
    """Run the fixed scraper."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_scraped_data()
    else:
        scraper = FixedDocumentationScraper()
        scraper.run_complete_scraping()

if __name__ == "__main__":
    main()