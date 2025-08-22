#!/usr/bin/env python3
"""
Ultimate Documentation AI System for Itential Internal Engineering
Combines: Sitemap crawling, complete scraping, live search fallback, and intelligent caching
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import pickle
from collections import defaultdict

class UltimateDocumentationSystem:
    """
    Complete documentation system with multiple strategies:
    1. Sitemap-based comprehensive crawling
    2. Complete content extraction
    3. Intelligent indexing
    4. Live search fallback
    5. Smart caching and updates
    """
    
    def __init__(self, base_url: str = "https://docs.itential.com/docs/"):
        self.base_url = base_url
        self.data_dir = Path("./ultimate_docs_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage paths
        self.sitemap_file = self.data_dir / "sitemap.json"
        self.content_db = self.data_dir / "content_database.jsonl"
        self.index_db = self.data_dir / "search_index.pkl"
        self.url_map = self.data_dir / "url_mapping.json"
        self.metadata_db = self.data_dir / "metadata.json"
        
        # Tracking
        self.all_urls: Set[str] = set()
        self.scraped_urls: Set[str] = set()
        self.excluded_urls: Set[str] = set()
        
        # Documentation structure
        self.doc_tree = {
            'products': defaultdict(list),
            'versions': defaultdict(list),
            'categories': defaultdict(list),
            'search_index': defaultdict(set)
        }

    async def discover_all_urls(self) -> Set[str]:
        """
        Discover ALL documentation URLs using multiple methods:
        1. Sitemap parsing
        2. Homepage crawling
        3. Navigation menu extraction
        4. Deep link following
        """
        print("🔍 Discovering all documentation URLs...")
        urls = set()
        
        # Method 1: Try to find sitemap
        sitemap_urls = await self._parse_sitemap()
        urls.update(sitemap_urls)
        print(f"  📄 Found {len(sitemap_urls)} URLs from sitemap")
        
        # Method 2: Crawl navigation and main pages
        nav_urls = await self._crawl_navigation()
        urls.update(nav_urls)
        print(f"  🧭 Found {len(nav_urls)} URLs from navigation")
        
        # Method 3: Deep crawl from known entry points
        entry_points = [
            self.base_url,
            f"{self.base_url}getting-started",
            f"{self.base_url}installation",
            f"{self.base_url}configuration", 
            f"{self.base_url}administration",
            f"{self.base_url}troubleshooting",
            f"{self.base_url}release-notes",
            f"{self.base_url}platform-6",
            f"{self.base_url}iap-2023-1",
            f"{self.base_url}iap-2023-2",
            f"{self.base_url}iag-2023-1",
            f"{self.base_url}iag-2023-2"
        ]
        
        deep_urls = await self._deep_crawl(entry_points)
        urls.update(deep_urls)
        print(f"  🕸️ Found {len(deep_urls)} URLs from deep crawl")
        
        # Filter to only /docs/ URLs
        filtered_urls = self._filter_docs_only(urls)
        print(f"  ✅ Total unique /docs/ URLs: {len(filtered_urls)}")
        
        return filtered_urls

    async def _parse_sitemap(self) -> Set[str]:
        """Parse sitemap.xml if available."""
        urls = set()
        sitemap_urls = [
            "https://docs.itential.com/sitemap.xml",
            "https://docs.itential.com/docs/sitemap.xml",
            "https://docs.itential.com/sitemap_index.xml"
        ]
        
        async with aiohttp.ClientSession() as session:
            for sitemap_url in sitemap_urls:
                try:
                    async with session.get(sitemap_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            # Parse XML
                            root = ET.fromstring(content)
                            # Handle different sitemap formats
                            for elem in root.iter():
                                if 'loc' in elem.tag:
                                    url = elem.text
                                    if url and '/docs/' in url:
                                        urls.add(url)
                except:
                    continue
        
        return urls

    async def _crawl_navigation(self) -> Set[str]:
        """Crawl navigation menus and sidebars."""
        urls = set()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find all navigation links
                        nav_selectors = [
                            'nav a',
                            '.sidebar a',
                            '.navigation a',
                            '.menu a',
                            '.toc a',  # Table of contents
                            'aside a'
                        ]
                        
                        for selector in nav_selectors:
                            for link in soup.select(selector):
                                href = link.get('href')
                                if href:
                                    full_url = urljoin(self.base_url, href)
                                    if '/docs/' in full_url:
                                        urls.add(full_url)
            except:
                pass
        
        return urls

    async def _deep_crawl(self, entry_points: List[str], max_depth: int = 5) -> Set[str]:
        """Deep crawl from entry points to discover all linked pages."""
        urls = set()
        visited = set()
        queue = [(url, 0) for url in entry_points]
        
        async with aiohttp.ClientSession() as session:
            while queue:
                url, depth = queue.pop(0)
                
                if url in visited or depth > max_depth:
                    continue
                
                visited.add(url)
                
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Find all links
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                full_url = urljoin(url, href)
                                
                                # Clean URL
                                parsed = urlparse(full_url)
                                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                                
                                if '/docs/' in clean_url and clean_url not in visited:
                                    urls.add(clean_url)
                                    if depth < max_depth:
                                        queue.append((clean_url, depth + 1))
                except:
                    continue
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        return urls

    def _filter_docs_only(self, urls: Set[str]) -> Set[str]:
        """Filter to only /docs/ URLs, excluding API and opensource."""
        filtered = set()
        
        for url in urls:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Must be docs.itential.com
            if 'docs.itential.com' not in parsed.netloc:
                continue
            
            # Must be in /docs/
            if not path.startswith('/docs'):
                continue
            
            # Exclude unwanted sections
            excluded = ['/api/', '/opensource/', '/swagger/', '/openapi/', '/graphql/']
            if any(exc in path for exc in excluded):
                self.excluded_urls.add(url)
                continue
            
            # Skip non-content
            if path.endswith(('.pdf', '.zip', '.png', '.jpg', '.svg')):
                continue
            
            filtered.add(url)
        
        return filtered

    async def scrape_all_content(self, urls: Set[str]) -> Dict[str, Dict]:
        """Scrape complete content from all URLs."""
        print(f"\n📥 Scraping {len(urls)} documentation pages...")
        
        content_db = {}
        semaphore = asyncio.Semaphore(5)  # Concurrent limit
        
        async def scrape_url(session, url):
            async with semaphore:
                try:
                    await asyncio.sleep(0.5)  # Rate limiting
                    async with session.get(url, timeout=30) as response:
                        if response.status != 200:
                            return None
                        
                        html = await response.text()
                        content = self._extract_complete_content(html, url)
                        
                        if content:
                            print(f"  ✅ Scraped: {content['title'][:50]}...")
                            return (url, content)
                        
                except Exception as e:
                    print(f"  ❌ Failed: {url} - {str(e)[:50]}")
                    return None

        async with aiohttp.ClientSession() as session:
            tasks = [scrape_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if result:
                    url, content = result
                    content_db[url] = content
                    self.scraped_urls.add(url)
        
        print(f"✅ Successfully scraped {len(content_db)} pages")
        return content_db

    def _extract_complete_content(self, html: str, url: str) -> Dict:
        """Extract complete structured content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove non-content
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else ""
        
        # Find main content
        content_area = None
        for selector in ['main', '.main-content', '.content', 'article', '#content']:
            content_area = soup.select_one(selector)
            if content_area:
                break
        
        if not content_area:
            content_area = soup.find('body')
        
        if not content_area:
            return None
        
        # Extract everything
        content = {
            'url': url,
            'title': title_text,
            'raw_text': content_area.get_text(separator='\n', strip=True),
            'headings': [],
            'paragraphs': [],
            'code_blocks': [],
            'tables': [],
            'lists': [],
            'links': [],
            'metadata': self._extract_metadata(soup, url)
        }
        
        # Extract headings with hierarchy
        for h in content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = h.get_text(strip=True)
            if text:
                content['headings'].append({
                    'level': int(h.name[1]),
                    'text': text,
                    'id': h.get('id', '')
                })
        
        # Extract paragraphs
        for p in content_area.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 30:
                content['paragraphs'].append(text)
        
        # Extract code blocks
        for code in content_area.find_all(['pre', 'code']):
            code_text = code.get_text(strip=True)
            if code_text and len(code_text) > 20:
                content['code_blocks'].append({
                    'code': code_text,
                    'language': self._detect_language(code_text)
                })
        
        # Extract tables
        for table in content_area.find_all('table'):
            table_data = self._parse_table(table)
            if table_data:
                content['tables'].append(table_data)
        
        # Extract lists
        for lst in content_area.find_all(['ul', 'ol']):
            items = [li.get_text(strip=True) for li in lst.find_all('li')]
            if items:
                content['lists'].append({
                    'type': lst.name,
                    'items': items
                })
        
        # Extract internal links
        for a in content_area.find_all('a', href=True):
            href = a['href']
            if href.startswith('/') or 'itential.com' in href:
                content['links'].append({
                    'url': urljoin(url, href),
                    'text': a.get_text(strip=True)
                })
        
        return content

    def _extract_metadata(self, soup, url: str) -> Dict:
        """Extract metadata from page."""
        metadata = {
            'url': url,
            'scraped_at': datetime.now().isoformat()
        }
        
        # Parse URL for metadata
        path = urlparse(url).path.lower()
        
        # Detect product
        if 'iap' in path:
            metadata['product'] = 'IAP'
        elif 'iag' in path:
            metadata['product'] = 'IAG'
        elif 'platform' in path:
            metadata['product'] = 'Platform'
        else:
            metadata['product'] = 'General'
        
        # Detect version
        version_match = re.search(r'(\d{4}\.\d+(?:\.\d+)?)', path)
        if version_match:
            metadata['version'] = version_match.group(1)
        
        # Detect category
        categories = {
            'install': 'Installation',
            'config': 'Configuration',
            'troubleshoot': 'Troubleshooting',
            'admin': 'Administration',
            'migration': 'Migration',
            'release': 'Release Notes',
            'api': 'API',
            'cli': 'CLI'
        }
        
        for key, value in categories.items():
            if key in path:
                metadata['category'] = value
                break
        
        return metadata

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        if 'import ' in code or 'def ' in code:
            return 'python'
        elif 'const ' in code or 'function ' in code:
            return 'javascript'
        elif 'SELECT' in code.upper():
            return 'sql'
        elif '#!/bin/bash' in code:
            return 'bash'
        elif '{' in code and '}' in code:
            return 'json'
        else:
            return 'text'

    def _parse_table(self, table) -> Optional[Dict]:
        """Parse HTML table."""
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = []
        
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all('td')]
            if cells:
                rows.append(cells)
        
        if headers or rows:
            return {
                'headers': headers,
                'rows': rows
            }
        return None

    def build_search_index(self, content_db: Dict[str, Dict]):
        """Build comprehensive search index."""
        print("\n🔨 Building search index...")
        
        # Create inverted index
        inverted_index = defaultdict(set)
        
        # Create multiple indices for different search strategies
        title_index = defaultdict(set)
        heading_index = defaultdict(set)
        code_index = defaultdict(set)
        exact_phrase_index = defaultdict(set)
        
        for url, content in content_db.items():
            # Index title
            title_tokens = self._tokenize(content['title'])
            for token in title_tokens:
                title_index[token].add(url)
                inverted_index[token].add(url)
            
            # Index headings
            for heading in content['headings']:
                heading_tokens = self._tokenize(heading['text'])
                for token in heading_tokens:
                    heading_index[token].add(url)
                    inverted_index[token].add(url)
            
            # Index code
            for code_block in content['code_blocks']:
                code_tokens = self._tokenize(code_block['code'])
                for token in code_tokens[:50]:  # Limit tokens per code block
                    code_index[token].add(url)
            
            # Index paragraphs
            for para in content['paragraphs']:
                para_tokens = self._tokenize(para)
                for token in para_tokens:
                    inverted_index[token].add(url)
                
                # Index 2-3 word phrases
                words = para.lower().split()
                for i in range(len(words) - 1):
                    phrase2 = f"{words[i]} {words[i+1]}"
                    exact_phrase_index[phrase2].add(url)
                    
                    if i < len(words) - 2:
                        phrase3 = f"{words[i]} {words[i+1]} {words[i+2]}"
                        exact_phrase_index[phrase3].add(url)
        
        # Save indices
        search_index = {
            'inverted': dict(inverted_index),
            'title': dict(title_index),
            'heading': dict(heading_index),
            'code': dict(code_index),
            'phrase': dict(exact_phrase_index),
            'metadata': {
                'total_docs': len(content_db),
                'total_terms': len(inverted_index),
                'created_at': datetime.now().isoformat()
            }
        }
        
        with open(self.index_db, 'wb') as f:
            pickle.dump(search_index, f)
        
        print(f"✅ Index built with {len(inverted_index)} unique terms")
        return search_index

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text for indexing."""
        # Convert to lowercase and split
        tokens = set()
        
        # Clean text
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        words = text.split()
        
        for word in words:
            if len(word) > 2:  # Skip very short words
                tokens.add(word)
                
                # Add stem variations
                if word.endswith('ing'):
                    tokens.add(word[:-3])
                elif word.endswith('ed'):
                    tokens.add(word[:-2])
                elif word.endswith('s') and len(word) > 3:
                    tokens.add(word[:-1])
        
        return tokens

    def save_everything(self, content_db: Dict[str, Dict]):
        """Save all data to disk."""
        print("\n💾 Saving data...")
        
        # Save content database
        with open(self.content_db, 'w', encoding='utf-8') as f:
            for url, content in content_db.items():
                f.write(json.dumps({
                    'url': url,
                    'content': content
                }, ensure_ascii=False) + '\n')
        
        # Save URL mapping
        url_map = {
            'all_urls': list(self.all_urls),
            'scraped_urls': list(self.scraped_urls),
            'excluded_urls': list(self.excluded_urls),
            'total_scraped': len(self.scraped_urls),
            'total_excluded': len(self.excluded_urls)
        }
        
        with open(self.url_map, 'w') as f:
            json.dump(url_map, f, indent=2)
        
        # Save metadata
        metadata = {
            'base_url': self.base_url,
            'total_pages': len(content_db),
            'scraped_at': datetime.now().isoformat(),
            'products': list(self.doc_tree['products'].keys()),
            'versions': list(self.doc_tree['versions'].keys()),
            'categories': list(self.doc_tree['categories'].keys())
        }
        
        with open(self.metadata_db, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ All data saved")

    async def run_complete_scraping(self):
        """Run the complete scraping process."""
        print("🚀 Starting Ultimate Documentation Scraping")
        print("=" * 60)
        
        # Step 1: Discover all URLs
        all_urls = await self.discover_all_urls()
        self.all_urls = all_urls
        
        # Step 2: Scrape all content
        content_db = await self.scrape_all_content(all_urls)
        
        # Step 3: Build search index
        search_index = self.build_search_index(content_db)
        
        # Step 4: Save everything
        self.save_everything(content_db)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 SCRAPING COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"✅ Total URLs discovered: {len(self.all_urls)}")
        print(f"✅ Successfully scraped: {len(self.scraped_urls)}")
        print(f"❌ Excluded (API/opensource): {len(self.excluded_urls)}")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🔍 Search terms indexed: {len(search_index['inverted'])}")
        
        # Show category breakdown
        categories = defaultdict(int)
        for url, content in content_db.items():
            cat = content['metadata'].get('category', 'General')
            categories[cat] += 1
        
        print("\n📚 Content by Category:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} pages")

class FastSearchEngine:
    """Ultra-fast search engine using the pre-built index."""
    
    def __init__(self, data_dir: str = "./ultimate_docs_data"):
        self.data_dir = Path(data_dir)
        
        # Load search index
        with open(self.data_dir / "search_index.pkl", 'rb') as f:
            self.search_index = pickle.load(f)
        
        # Load content database
        self.content_db = {}
        with open(self.data_dir / "content_database.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line)
                self.content_db[data['url']] = data['content']
        
        print(f"✅ Loaded {len(self.content_db)} documents")
        print(f"✅ Search index has {len(self.search_index['inverted'])} terms")

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Ultra-fast search using multiple strategies:
        1. Exact phrase matching
        2. Title matching (highest weight)
        3. Heading matching (high weight)
        4. Content matching
        5. Code matching
        """
        query_lower = query.lower()
        results = defaultdict(float)
        
        # Strategy 1: Exact phrase match (highest score)
        if query_lower in self.search_index['phrase']:
            for url in self.search_index['phrase'][query_lower]:
                results[url] += 10.0
        
        # Strategy 2: Title match (very high score)
        query_tokens = self._tokenize(query_lower)
        for token in query_tokens:
            if token in self.search_index['title']:
                for url in self.search_index['title'][token]:
                    results[url] += 5.0
        
        # Strategy 3: Heading match (high score)
        for token in query_tokens:
            if token in self.search_index['heading']:
                for url in self.search_index['heading'][token]:
                    results[url] += 3.0
        
        # Strategy 4: Content match
        for token in query_tokens:
            if token in self.search_index['inverted']:
                for url in self.search_index['inverted'][token]:
                    results[url] += 1.0
        
        # Strategy 5: Code match (for technical queries)
        if any(tech in query_lower for tech in ['code', 'example', 'script', 'command']):
            for token in query_tokens:
                if token in self.search_index['code']:
                    for url in self.search_index['code'][token]:
                        results[url] += 2.0
        
        # Sort by score
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Format results
        formatted_results = []
        for url, score in sorted_results:
            if url in self.content_db:
                content = self.content_db[url]
                formatted_results.append({
                    'url': url,
                    'title': content['title'],
                    'score': score,
                    'preview': content['raw_text'][:300] + '...',
                    'metadata': content['metadata']
                })
        
        return formatted_results

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize query."""
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        return set(word for word in text.split() if len(word) > 2)

# Main execution
async def main():
    """Run the complete system."""
    
    # Step 1: Complete scraping
    scraper = UltimateDocumentationSystem()
    await scraper.run_complete_scraping()
    
    # Step 2: Test search
    print("\n🧪 Testing search engine...")
    search = FastSearchEngine()
    
    test_queries = [
        "Python version requirements IAP 2023.1",
        "Install Platform 6",
        "MongoDB configuration",
        "CLI troubleshooting duplicate data",
        "Event deduplication"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = search.search(query, limit=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title'][:60]}... (score: {result['score']:.2f})")
            print(f"     {result['url']}")

if __name__ == "__main__":
    asyncio.run(main())