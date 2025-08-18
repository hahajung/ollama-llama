#!/usr/bin/env python3
"""
Comprehensive Technical Documentation Scraper for docs.itential.com
Focuses on extracting ALL version information, dependencies, and technical requirements.
Windows-compatible version with no emoji characters.
"""

import os
import re
import json
import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque, defaultdict
from typing import Set, Deque, Dict, Optional, List, Tuple, Union, Any
import time
from pathlib import Path

class TechnicalDocumentationScraper:
    """Comprehensive scraper specifically designed for technical documentation."""
    
    def __init__(self, 
                 base_url: str = "https://docs.itential.com/", 
                 output_file: str = "complete_technical_docs.jsonl",
                 max_concurrent: int = 3,
                 rate_limit: float = 0.8):
        self.base_url = base_url
        self.output_file = output_file
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.visited_urls: Set[str] = set()
        self.queue: Deque[Tuple[str, float, int]] = deque()
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.scraped_count = 0
        self.output_lock = asyncio.Lock()
        self.last_request_time = 0
        
        # Technical content tracking
        self.found_technical_pages: Set[str] = set()
        self.version_registry: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_registry: Dict[str, Dict] = {}
        
        # Priority technical keywords for IAP/IAG systems
        self.technical_keywords = {
            # Product versions
            'iap_versions': ['iap', 'itential automation platform', 'automation platform'],
            'iag_versions': ['iag', 'itential automation gateway', 'automation gateway'],
            'platform_versions': ['platform 6', 'platform 7', 'platform 8'],
            
            # Dependencies
            'runtime_deps': ['python', 'node.js', 'nodejs', 'java', 'npm', 'pip'],
            'database_deps': ['mongodb', 'redis', 'elasticsearch', 'postgresql'],
            'messaging_deps': ['rabbitmq', 'bullmq', 'kafka', 'mqtt'],
            'infrastructure_deps': ['docker', 'kubernetes', 'helm', 'nginx'],
            
            # Technical concepts
            'technical_terms': [
                'dependencies', 'requirements', 'prerequisites', 'installation', 
                'configuration', 'setup', 'deployment', 'upgrade', 'migration',
                'compatibility', 'support matrix', 'version matrix'
            ]
        }
        
        # Version patterns to extract
        self.version_patterns = [
            r'(?:IAP|iap)[\s]*([0-9]{4}\.[0-9]+(?:\.[0-9]+)?)',  # IAP 2023.1, IAP 2023.2.1
            r'(?:IAG|iag)[\s]*([0-9]{4}\.[0-9]+(?:\.[0-9]+)?)',  # IAG 2023.1
            r'Platform[\s]*([0-9]+(?:\.[0-9]+)?)',               # Platform 6, Platform 7.1
            r'MongoDB[\s]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)',        # MongoDB 5.0, 6.0.1
            r'Python[\s]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)',         # Python 3.9.5
            r'Node\.js[\s]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)',       # Node.js 18.15.0
            r'Redis[\s]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)',          # Redis 6.2.7
            r'RabbitMQ[\s]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)',       # RabbitMQ 3.9.0
        ]

    async def _init_session(self) -> None:
        """Initialize HTTP session with proper headers."""
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            },
            timeout=aiohttp.ClientTimeout(total=90),
            connector=aiohttp.TCPConnector(limit_per_host=4)
        )

    def is_technical_url(self, url: str) -> bool:
        """Determine if URL contains technical documentation."""
        parsed = urlparse(url)
        invalid_exts = {'.pdf', '.zip', '.jpg', '.png', '.gif', '.svg', '.mp4', '.css', '.js'}
        invalid_patterns = {'login', 'search', 'edit', 'admin', 'api/', '_static/', 'blog/', 'news/'}
        
        if (parsed.netloc != 'docs.itential.com' or
            any(url.endswith(ext) for ext in invalid_exts) or
            any(pattern in url.lower() for pattern in invalid_patterns) or
            '#' in url or len(url) > 250):
            return False
        
        # Prioritize technical documentation paths
        technical_paths = [
            'installation', 'dependencies', 'requirements', 'prerequisites',
            'configuration', 'setup', 'deployment', 'upgrade', 'migration',
            'version', 'release', 'compatibility', 'support',
            'adapter', 'platform', 'gateway', 'automation'
        ]
        
        url_lower = url.lower()
        return any(path in url_lower for path in technical_paths) or '/docs/' in url_lower

    def calculate_technical_priority(self, url: str, title: str = "", content: str = "") -> float:
        """Calculate priority based on technical relevance."""
        priority = 1.0
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # High priority: Critical technical documentation
        critical_patterns = [
            'dependencies', 'requirements', 'installation', 'prerequisites',
            'version', 'release-notes', 'compatibility', 'support-matrix',
            'configuration', 'deployment', 'upgrade'
        ]
        
        for pattern in critical_patterns:
            if pattern in url_lower:
                priority += 10.0
            if pattern in title_lower:
                priority += 5.0
        
        # Product-specific boosts
        product_patterns = {
            'iap': 8.0, 'automation-platform': 8.0,
            'iag': 7.0, 'automation-gateway': 7.0,
            'platform-6': 6.0, 'platform-7': 6.0
        }
        
        for pattern, boost in product_patterns.items():
            if pattern in url_lower:
                priority += boost
        
        # Dependency-specific boosts
        for category, keywords in self.technical_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    priority += 3.0
                    break
        
        # Version number boosts
        version_indicators = ['2023.1', '2023.2', '2022.1', '2024.1', 'v6', 'v7', 'v8']
        for indicator in version_indicators:
            if indicator in url_lower or indicator in content_lower:
                priority += 4.0
        
        return priority

    async def rate_limited_get(self, url: str) -> aiohttp.ClientResponse:
        """Rate-limited HTTP request."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
        
        if self.session is None:
            raise RuntimeError("Session not initialized")
        return await self.session.get(url)

    def extract_comprehensive_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract all tables with enhanced structure preservation."""
        tables: List[Dict[str, Any]] = []
        
        for i, table in enumerate(soup.find_all('table')):
            try:
                # Multiple parsing strategies
                table_data = self._parse_table_multiple_ways(table, i)
                
                # Analyze table content for technical relevance
                table_data['technical_relevance'] = self._analyze_table_technical_content(table_data)
                
                # Extract version information from table
                table_data['extracted_versions'] = self._extract_versions_from_table(table_data)
                
                tables.append(table_data)
                
                # Replace table with enriched placeholder
                placeholder = self._create_enriched_table_placeholder(table_data, i)
                table.replace_with(placeholder)
                
            except Exception as e:
                print(f"Error processing table {i}: {e}")
                # Fallback handling
                table_text = self.clean_text(table.get_text())
                tables.append({
                    'id': f'table_{i}',
                    'text': table_text,
                    'technical_relevance': 'low',
                    'extracted_versions': []
                })
                table.replace_with(f"[TABLE_{i}] {table_text[:200]}")
        
        return tables

    def _parse_table_multiple_ways(self, table: any, table_id: int) -> Dict[str, Any]:
        """Parse table using multiple strategies for robustness."""
        table_data = {
            'id': f'table_{table_id}',
            'headers': [],
            'rows': [],
            'markdown': '',
            'csv': '',
            'raw_html': str(table),
            'context_before': '',
            'context_after': ''
        }
        
        try:
            # Strategy 1: pandas read_html (most robust)
            df = pd.read_html(str(table))[0]
            table_data['headers'] = list(df.columns)
            table_data['rows'] = df.values.tolist()
            table_data['markdown'] = df.to_markdown(index=False)
            table_data['csv'] = df.to_csv(index=False)
            table_data['success_method'] = 'pandas'
            
        except Exception:
            try:
                # Strategy 2: Manual parsing
                headers = []
                rows = []
                
                # Extract headers
                header_row = table.find('tr')
                if header_row:
                    for th in header_row.find_all(['th', 'td']):
                        headers.append(self.clean_text(th.get_text()))
                
                # Extract data rows
                for tr in table.find_all('tr')[1:]:  # Skip header row
                    row = []
                    for td in tr.find_all(['td', 'th']):
                        row.append(self.clean_text(td.get_text()))
                    if row:
                        rows.append(row)
                
                table_data['headers'] = headers
                table_data['rows'] = rows
                table_data['success_method'] = 'manual'
                
                # Create markdown manually
                if headers and rows:
                    markdown_lines = ['| ' + ' | '.join(headers) + ' |']
                    markdown_lines.append('|' + '---|' * len(headers))
                    for row in rows:
                        padded_row = row + [''] * (len(headers) - len(row))  # Pad short rows
                        markdown_lines.append('| ' + ' | '.join(padded_row[:len(headers)]) + ' |')
                    table_data['markdown'] = '\n'.join(markdown_lines)
                
            except Exception as e:
                # Strategy 3: Fallback to plain text
                table_data['text'] = self.clean_text(table.get_text())
                table_data['success_method'] = 'fallback'
        
        # Get context around table
        self._get_table_context(table, table_data)
        
        return table_data

    def _analyze_table_technical_content(self, table_data: Dict[str, Any]) -> str:
        """Analyze technical relevance of table content."""
        content = str(table_data).lower()
        
        # Critical technical indicators
        critical_indicators = [
            'version', 'dependency', 'requirement', 'prerequisite',
            'python', 'node', 'mongodb', 'redis', 'rabbitmq',
            'iap', 'iag', 'platform', 'compatibility'
        ]
        
        critical_count = sum(1 for indicator in critical_indicators if indicator in content)
        
        if critical_count >= 5:
            return 'critical'
        elif critical_count >= 3:
            return 'high'
        elif critical_count >= 1:
            return 'medium'
        else:
            return 'low'

    def _extract_versions_from_table(self, table_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract version information from table content."""
        versions = []
        content = str(table_data)
        
        for pattern in self.version_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                product = match.group(0).split()[0]  # Get product name
                version = match.group(1)
                versions.append({
                    'product': product,
                    'version': version,
                    'pattern': pattern,
                    'context': content[max(0, match.start()-50):match.end()+50]
                })
        
        return versions

    def _create_enriched_table_placeholder(self, table_data: Dict[str, Any], table_id: int) -> str:
        """Create an enriched placeholder that preserves searchable content."""
        content_parts = [f"[TABLE_{table_id}]"]
        
        # Add technical relevance
        relevance = table_data.get('technical_relevance', 'unknown')
        content_parts.append(f"Technical relevance: {relevance}")
        
        # Add extracted versions
        versions = table_data.get('extracted_versions', [])
        if versions:
            version_info = ', '.join([f"{v['product']} {v['version']}" for v in versions])
            content_parts.append(f"Versions: {version_info}")
        
        # Add table content
        markdown = table_data.get('markdown', '')
        if markdown:
            content_parts.append(markdown[:500])
        else:
            text = table_data.get('text', '')
            content_parts.append(text[:300])
        
        return ' | '.join(content_parts)

    def _get_table_context(self, table: any, table_data: Dict[str, Any]) -> None:
        """Extract comprehensive context around table."""
        try:
            # Get preceding context (headers, paragraphs)
            prev_elements = []
            current = table.previous_sibling
            
            while current and len(prev_elements) < 5:
                if hasattr(current, 'get_text'):
                    text = current.get_text().strip()
                    if text and len(text) > 10:
                        # Prioritize headers
                        if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            prev_elements.insert(0, f"HEADER: {text}")
                        elif len(text) < 200:
                            prev_elements.insert(0, text)
                current = current.previous_sibling
            
            table_data['context_before'] = ' | '.join(prev_elements)
            
            # Get following context
            next_elements = []
            current = table.next_sibling
            
            while current and len(next_elements) < 3:
                if hasattr(current, 'get_text'):
                    text = current.get_text().strip()
                    if text and len(text) > 10 and len(text) < 200:
                        next_elements.append(text)
                current = current.next_sibling
            
            table_data['context_after'] = ' | '.join(next_elements)
            
        except Exception as e:
            print(f"Error getting table context: {e}")

    def analyze_technical_content(self, url: str, title: str, content: str) -> Dict[str, Any]:
        """Comprehensive analysis of technical content."""
        analysis = {
            'priority_score': 1.0,
            'content_type': 'general',
            'technical_indicators': set(),
            'extracted_versions': {},
            'dependency_info': {},
            'is_critical': False
        }
        
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Determine content type based on comprehensive analysis
        if any(term in url_lower for term in ['dependencies', 'requirement']):
            analysis['content_type'] = 'dependencies'
            analysis['is_critical'] = True
            analysis['priority_score'] += 15.0
            
        elif any(term in url_lower for term in ['version', 'lifecycle', 'release']):
            analysis['content_type'] = 'version_lifecycle'
            analysis['is_critical'] = True
            analysis['priority_score'] += 12.0
            
        elif any(term in url_lower for term in ['installation', 'setup', 'configuration']):
            analysis['content_type'] = 'installation_config'
            analysis['priority_score'] += 8.0
            
        elif any(term in url_lower for term in ['migration', 'upgrade']):
            analysis['content_type'] = 'migration_upgrade'
            analysis['priority_score'] += 7.0
        
        # Extract version information
        for pattern in self.version_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                product = match.group(0).split()[0].lower()
                version = match.group(1)
                
                if product not in analysis['extracted_versions']:
                    analysis['extracted_versions'][product] = set()
                analysis['extracted_versions'][product].add(version)
                
                # Track globally
                self.version_registry[product].add(version)
        
        # Analyze dependencies
        dependency_keywords = {
            'python': ['python', 'pip', 'virtualenv', 'conda'],
            'nodejs': ['node.js', 'nodejs', 'npm', 'yarn'],
            'mongodb': ['mongodb', 'mongo', 'mongoose'],
            'redis': ['redis', 'redis-server'],
            'rabbitmq': ['rabbitmq', 'rabbit-mq', 'amqp'],
            'docker': ['docker', 'dockerfile', 'container'],
            'kubernetes': ['kubernetes', 'k8s', 'kubectl', 'helm']
        }
        
        for dep_type, keywords in dependency_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                analysis['dependency_info'][dep_type] = True
                analysis['technical_indicators'].add(dep_type)
                analysis['priority_score'] += 2.0
        
        # Convert sets to lists for JSON serialization
        analysis['technical_indicators'] = list(analysis['technical_indicators'])
        for product in analysis['extracted_versions']:
            analysis['extracted_versions'][product] = list(analysis['extracted_versions'][product])
        
        return analysis

    async def extract_comprehensive_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract comprehensive technical content."""
        try:
            async with self.semaphore:
                async with await self.rate_limited_get(url) as response:
                    if response.status != 200:
                        print(f"HTTP {response.status} for {url}")
                        return None

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Remove unwanted elements
                    for tag in soup(["script", "style", "nav", "footer", 
                                    "header", "iframe", "noscript", "form"]):
                        tag.decompose()

                    # Extract metadata
                    title = soup.title.get_text(strip=True) if soup.title else ""
                    description_tag = soup.find('meta', attrs={'name': 'description'})
                    description = description_tag['content'].strip() if description_tag else ""

                    # Extract heading hierarchy
                    headings: List[Dict[str, str]] = []
                    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        heading_text = self.clean_text(h.get_text())
                        if heading_text:
                            headings.append({
                                'level': h.name,
                                'text': heading_text,
                                'id': h.get('id', ''),
                                'technical_relevance': self._assess_heading_relevance(heading_text)
                            })

                    # Extract comprehensive tables (most important for dependencies)
                    tables = self.extract_comprehensive_tables(soup)

                    # Extract code blocks and configuration examples
                    code_blocks = self._extract_comprehensive_code_blocks(soup)

                    # Get main content
                    content_div = self._find_main_content(soup)
                    if content_div:
                        raw_text = self.clean_text(content_div.get_text(separator='\n'))
                        
                        # Create enhanced searchable text
                        searchable_text = self._create_enhanced_searchable_text(
                            raw_text, tables, code_blocks, headings
                        )
                    else:
                        raw_text = searchable_text = ""

                    # Extract links
                    links: List[Tuple[str, float, str]] = []
                    for a in soup.find_all('a', href=True):
                        full_url = urljoin(url, a['href'])
                        if self.is_technical_url(full_url):
                            link_text = a.get_text().strip()
                            link_priority = self.calculate_technical_priority(full_url, link_text)
                            links.append((full_url, link_priority, link_text))

                    # Comprehensive technical analysis
                    technical_analysis = self.analyze_technical_content(url, title, searchable_text)
                    
                    # Track technical findings
                    if technical_analysis['is_critical']:
                        self.found_technical_pages.add(technical_analysis['content_type'])
                        print(f"Found critical page: {technical_analysis['content_type']} - {title}")

                    # Create comprehensive document
                    return {
                        'url': url,
                        'title': title,
                        'description': description,
                        'headings': headings,
                        'tables': tables,
                        'code_blocks': code_blocks,
                        'raw_text': raw_text,
                        'searchable_text': searchable_text,
                        'links': links,
                        'timestamp': time.time(),
                        'content_type': technical_analysis['content_type'],
                        'priority_score': technical_analysis['priority_score'],
                        'is_critical': technical_analysis['is_critical'],
                        'technical_indicators': technical_analysis['technical_indicators'],
                        'extracted_versions': technical_analysis['extracted_versions'],
                        'dependency_info': technical_analysis['dependency_info'],
                        'table_count': len(tables),
                        'code_block_count': len(code_blocks),
                        'technical_relevance_score': self._calculate_technical_score(technical_analysis, tables, code_blocks)
                    }

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def _assess_heading_relevance(self, heading_text: str) -> str:
        """Assess technical relevance of headings."""
        heading_lower = heading_text.lower()
        
        critical_terms = ['dependencies', 'requirements', 'installation', 'configuration', 'version']
        high_terms = ['setup', 'deployment', 'upgrade', 'migration', 'compatibility']
        
        if any(term in heading_lower for term in critical_terms):
            return 'critical'
        elif any(term in heading_lower for term in high_terms):
            return 'high'
        else:
            return 'medium'

    def _extract_comprehensive_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code blocks with technical analysis."""
        code_blocks: List[Dict[str, Any]] = []
        
        for i, pre in enumerate(soup.find_all('pre')):
            code_text = pre.get_text()
            if code_text.strip():
                code_block = {
                    'id': f'codeblock_{i}',
                    'language': ' '.join(pre.get('class', [])),
                    'content': code_text,
                    'context': self._get_element_context(pre),
                    'code_type': self._detect_code_type_comprehensive(code_text),
                    'technical_relevance': self._assess_code_relevance(code_text),
                    'extracted_versions': self._extract_versions_from_text(code_text)
                }
                
                code_blocks.append(code_block)
                
                # Create searchable replacement
                searchable_text = self._create_searchable_code_text_comprehensive(code_block)
                enhanced_element = soup.new_tag('div')
                enhanced_element['class'] = 'enhanced-code-block'
                enhanced_element.string = searchable_text[:600]
                pre.replace_with(enhanced_element)
        
        return code_blocks

    def _detect_code_type_comprehensive(self, code_text: str) -> str:
        """Comprehensive code type detection."""
        code_lower = code_text.lower()
        
        # Configuration files
        if 'properties.json' in code_lower or ('{' in code_text and 'ldap' in code_lower):
            return 'LDAP Configuration'
        elif 'mongdb' in code_lower or 'mongodb' in code_lower:
            return 'MongoDB Configuration'
        elif 'redis' in code_lower:
            return 'Redis Configuration'
        elif 'rabbitmq' in code_lower or 'amqp' in code_lower:
            return 'RabbitMQ Configuration'
        
        # Installation commands
        elif any(cmd in code_lower for cmd in ['npm install', 'pip install', 'apt-get', 'yum install']):
            return 'Installation Commands'
        elif any(cmd in code_lower for cmd in ['docker run', 'docker build', 'kubectl']):
            return 'Deployment Commands'
        
        # Language detection
        elif any(py in code_lower for py in ['import ', 'def ', 'class ', 'python']):
            return 'Python Code'
        elif any(js in code_lower for js in ['require(', 'const ', 'let ', 'function']):
            return 'JavaScript Code'
        elif code_text.startswith('#!/bin/bash') or 'bash' in code_lower:
            return 'Shell Script'
        
        return 'Code Block'

    def _assess_code_relevance(self, code_text: str) -> str:
        """Assess technical relevance of code blocks."""
        code_lower = code_text.lower()
        
        critical_indicators = ['config', 'dependency', 'requirement', 'version', 'install']
        high_indicators = ['setup', 'deployment', 'docker', 'kubernetes']
        
        critical_count = sum(1 for indicator in critical_indicators if indicator in code_lower)
        high_count = sum(1 for indicator in high_indicators if indicator in code_lower)
        
        if critical_count >= 2:
            return 'critical'
        elif critical_count >= 1 or high_count >= 2:
            return 'high'
        else:
            return 'medium'

    def _extract_versions_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract version information from any text."""
        versions = []
        
        for pattern in self.version_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                product = match.group(0).split()[0]
                version = match.group(1)
                versions.append({
                    'product': product,
                    'version': version,
                    'context': text[max(0, match.start()-30):match.end()+30]
                })
        
        return versions

    def _create_searchable_code_text_comprehensive(self, code_block: Dict[str, Any]) -> str:
        """Create comprehensive searchable text for code blocks."""
        parts = []
        
        # Add type and relevance
        code_type = code_block.get('code_type', 'Code')
        relevance = code_block.get('technical_relevance', 'medium')
        parts.append(f"{code_type} ({relevance} relevance)")
        
        # Add context
        context = code_block.get('context', '')
        if context:
            parts.append(f"Context: {context}")
        
        # Add version information
        versions = code_block.get('extracted_versions', [])
        if versions:
            version_info = ', '.join([f"{v['product']} {v['version']}" for v in versions])
            parts.append(f"Versions: {version_info}")
        
        # Add code content
        content = code_block.get('content', '')
        parts.append(f"Code: {content[:200]}")
        
        return ' | '.join(parts)

    def _create_enhanced_searchable_text(self, raw_text: str, tables: List, code_blocks: List, headings: List) -> str:
        """Create enhanced searchable text that includes all technical content."""
        searchable_parts = [raw_text]
        
        # Add table content with context
        for table in tables:
            relevance = table.get('technical_relevance', 'low')
            if relevance in ['critical', 'high']:
                table_content = table.get('markdown', '') or table.get('text', '')
                context_before = table.get('context_before', '')
                
                searchable_parts.append(f"TABLE ({relevance}): {context_before} | {table_content[:500]}")
        
        # Add code block content
        for code_block in code_blocks:
            relevance = code_block.get('technical_relevance', 'medium')
            if relevance in ['critical', 'high']:
                code_type = code_block.get('code_type', 'Code')
                content = code_block.get('content', '')
                searchable_parts.append(f"CODE ({code_type}): {content[:300]}")
        
        # Add critical headings
        for heading in headings:
            if heading.get('technical_relevance') == 'critical':
                searchable_parts.append(f"HEADING: {heading['text']}")
        
        return '\n\n'.join(searchable_parts)

    def _calculate_technical_score(self, analysis: Dict, tables: List, code_blocks: List) -> float:
        """Calculate overall technical relevance score."""
        score = 0.0
        
        # Base content type score
        content_type_scores = {
            'dependencies': 10.0,
            'version_lifecycle': 9.0,
            'installation_config': 7.0,
            'migration_upgrade': 6.0
        }
        score += content_type_scores.get(analysis['content_type'], 1.0)
        
        # Table scoring
        for table in tables:
            relevance = table.get('technical_relevance', 'low')
            if relevance == 'critical':
                score += 5.0
            elif relevance == 'high':
                score += 3.0
        
        # Code block scoring
        for code_block in code_blocks:
            relevance = code_block.get('technical_relevance', 'medium')
            if relevance == 'critical':
                score += 3.0
            elif relevance == 'high':
                score += 2.0
        
        # Version extraction bonus
        versions = analysis.get('extracted_versions', {})
        score += len(versions) * 2.0
        
        return min(score, 50.0)  # Cap at 50

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    @staticmethod
    def _find_main_content(soup: BeautifulSoup) -> Optional[any]:
        """Find main content with improved heuristics."""
        content_selectors = [
            'main', '.content', '.main-content', '.document',
            '.docs-content', '.page-content', '.content-area',
            '#content', '#main', 'article', '[role="main"]',
            '.rst-content'  # Common in technical docs
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                return content
        
        # Fallback strategy
        body = soup.find('body')
        if body:
            # Remove common non-content elements
            for element in body.find_all(['nav', 'aside', 'menu', 'footer', 'header']):
                element.decompose()
        return body

    def _get_element_context(self, element: any) -> str:
        """Get context around an element."""
        context_parts = []
        
        # Get preceding heading
        prev_heading = element.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if prev_heading:
            context_parts.append(f"Section: {prev_heading.get_text().strip()}")
        
        # Get preceding paragraph
        prev_p = element.find_previous('p')
        if prev_p:
            text = prev_p.get_text().strip()
            if len(text) < 200:
                context_parts.append(f"Context: {text}")
        
        return ' | '.join(context_parts)

    async def process_technical_page(self, url: str, priority: float, depth: int) -> None:
        """Process a single technical URL."""
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        page_data = await self.extract_comprehensive_content(url)

        if page_data:
            # Save results
            async with self.output_lock:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(page_data, ensure_ascii=False, indent=None) + '\n')
                self.scraped_count += 1
                
                # Enhanced logging with technical metrics
                tables_count = len(page_data.get('tables', []))
                code_count = len(page_data.get('code_blocks', []))
                content_type = page_data.get('content_type', 'general')
                is_critical = page_data.get('is_critical', False)
                tech_score = page_data.get('technical_relevance_score', 0)
                extracted_versions = page_data.get('extracted_versions', {})
                
                status = "CRITICAL" if is_critical else "PAGE"
                version_info = f"Versions: {len(extracted_versions)}" if extracted_versions else "No versions"
                
                print(f"{status} {self.scraped_count}: {page_data['title'][:40]}...")
                print(f"   Type: {content_type} | Tables: {tables_count} | Code: {code_count}")
                print(f"   Tech Score: {tech_score:.1f} | {version_info} | Priority: {priority:.1f}")
                
                # Log version findings
                if extracted_versions:
                    for product, versions in extracted_versions.items():
                        print(f"   Found {product}: {versions}")

            # Add new links with technical priority filtering
            if depth < 6:  # Deeper crawling for technical docs
                technical_links = []
                for link_url, link_priority, link_text in page_data.get('links', []):
                    if link_url not in self.visited_urls and link_priority > 2.0:  # Only high-priority technical links
                        technical_links.append((link_url, link_priority, depth + 1))
                
                # Sort by priority and add to queue
                technical_links.sort(key=lambda x: x[1], reverse=True)
                for link_url, link_priority, new_depth in technical_links[:20]:  # Limit to top 20 per page
                    self.queue.append((link_url, link_priority, new_depth))

    def sort_queue_by_technical_priority(self) -> None:
        """Sort queue by technical priority."""
        queue_list = list(self.queue)
        queue_list.sort(key=lambda x: x[1], reverse=True)
        self.queue = deque(queue_list)

    async def worker(self) -> None:
        """Worker process to handle URLs from the queue."""
        worker_id = id(asyncio.current_task())
        print(f"Worker {worker_id} started")
        
        consecutive_failures = 0
        max_failures = 5
        
        while consecutive_failures < max_failures:
            try:
                url, priority, depth = self.queue.popleft()
                consecutive_failures = 0  # Reset on successful dequeue
            except IndexError:
                # Queue is empty, wait a bit
                await asyncio.sleep(2)
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"Worker {worker_id} stopping - no work available")
                    break
                continue

            try:
                await self.process_technical_page(url, priority, depth)
            except Exception as e:
                print(f"Worker {worker_id} error processing {url}: {e}")
                consecutive_failures += 1
        
        print(f"Worker {worker_id} finished")

    async def scrape_technical_documentation(self) -> None:
        """Main scraping controller optimized for technical documentation."""
        print("Initializing scraper session...")
        await self._init_session()
        
        # Start with high-priority technical URLs
        initial_urls = [
            (self.base_url, 5.0, 0),
            (f"{self.base_url}installation/", 15.0, 0),
            (f"{self.base_url}dependencies/", 20.0, 0),
            (f"{self.base_url}requirements/", 18.0, 0),
            (f"{self.base_url}release-notes/", 12.0, 0),
            (f"{self.base_url}configuration/", 10.0, 0),
            (f"{self.base_url}deployment/", 8.0, 0)
        ]
        
        print(f"Adding {len(initial_urls)} initial URLs to queue...")
        for url, priority, depth in initial_urls:
            self.queue.append((url, priority, depth))
        
        print(f"Queue initialized with {len(self.queue)} URLs")
        print("Starting worker tasks...")

        # Start worker tasks
        workers = [asyncio.create_task(self.worker()) 
                  for _ in range(self.max_concurrent)]
        
        print(f"Started {len(workers)} worker tasks")

        # Monitor progress with technical focus
        last_count = 0
        stall_counter = 0
        iteration = 0
        
        # Technical targets to find
        critical_technical_pages = {
            'dependencies', 'version_lifecycle', 'installation_config'
        }
        
        max_pages = 500  # Reduced for faster completion
        min_technical_pages = 50  # Reduced minimum
        
        print(f"Target: Find comprehensive technical documentation")
        print(f"Looking for: {critical_technical_pages}")
        print(f"Max pages: {max_pages}, Min technical: {min_technical_pages}")
        
        try:
            while self.scraped_count < max_pages and iteration < 240:  # 1 hour max (15s * 240)
                await asyncio.sleep(15)
                iteration += 1
                
                # Sort queue periodically for technical priority
                if len(self.queue) > 50:
                    self.sort_queue_by_technical_priority()
                
                # Progress reporting
                technical_found = len(self.found_technical_pages)
                version_products = len(self.version_registry)
                
                print(f"Progress: {self.scraped_count} pages | {technical_found} technical types | {version_products} products with versions")
                print(f"Found: {self.found_technical_pages}")
                print(f"Queue size: {len(self.queue)} | Workers active: {sum(1 for w in workers if not w.done())}")
                
                if len(self.version_registry) > 0:
                    print(f"Version registry: {dict(list(self.version_registry.items())[:3])}...")
                
                # Check for comprehensive coverage
                if (technical_found >= 2 and 
                    version_products >= 3 and 
                    self.scraped_count >= min_technical_pages):
                    print("Comprehensive technical coverage achieved!")
                    break
                
                # Stall detection
                if self.scraped_count == last_count:
                    stall_counter += 1
                    if stall_counter > 8:  # 2 minutes
                        print("Scraping stalled, checking status...")
                        
                        # Check if workers are still alive
                        active_workers = [w for w in workers if not w.done()]
                        print(f"Active workers: {len(active_workers)}")
                        
                        if not active_workers or len(self.queue) == 0:
                            print("No active workers or empty queue, stopping...")
                            break
                        
                        if technical_found >= 1 and self.scraped_count >= 25:
                            print("Found some technical content, stopping...")
                            break
                        
                        stall_counter = 0  # Reset counter
                else:
                    stall_counter = 0
                    last_count = self.scraped_count
                
                # Force break if we have reasonable content
                if self.scraped_count >= 100 and technical_found >= 1:
                    print("Reasonable content acquired, stopping...")
                    break
        
        except Exception as e:
            print(f"Error during scraping: {e}")
        
        finally:
            # Cleanup
            print("Cleaning up workers...")
            for task in workers:
                if not task.done():
                    task.cancel()
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*workers, return_exceptions=True), 
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                print("Worker cleanup timed out")
            except Exception as e:
                print(f"Worker cleanup error: {e}")
                
            if self.session:
                print("Closing session...")
                await self.session.close()

        # Final technical summary
        print(f"\nTechnical Documentation Scraping Complete!")
        print("=" * 60)
        print(f"Total pages scraped: {self.scraped_count}")
        print(f"Technical page types found: {self.found_technical_pages}")
        print(f"Products with versions: {len(self.version_registry)}")
        print(f"Output saved to: {self.output_file}")
        
        # Show version registry summary
        if self.version_registry:
            print(f"\nVersion Registry Summary:")
            for product, versions in self.version_registry.items():
                versions_list = sorted(list(versions))
                print(f"  {product}: {versions_list}")
        
        if len(self.found_technical_pages) >= 1 and len(self.version_registry) >= 1:
            print("SUCCESS: Technical documentation captured!")
        else:
            print("WARNING: Limited technical content found")

def main() -> None:
    """Main function to run comprehensive technical scraper."""
    scraper = TechnicalDocumentationScraper(
        base_url="https://docs.itential.com/",
        output_file="complete_technical_docs.jsonl",
        max_concurrent=3,  # Conservative for stability
        rate_limit=0.8     # Respectful rate limiting
    )

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(scraper.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Clear existing output
    if os.path.exists(scraper.output_file):
        os.remove(scraper.output_file)
        print(f"Cleared existing output file: {scraper.output_file}")

    print(f"Starting Comprehensive Technical Documentation Scraping")
    print(f"Target: docs.itential.com with focus on:")
    print(f"   • IAP, IAG, Platform versions and dependencies")
    print(f"   • MongoDB, Redis, RabbitMQ, BullMQ versions")
    print(f"   • Python, Node.js, and other runtime requirements")
    print(f"   • Installation, configuration, and deployment guides")
    print(f"Output will be saved to: {scraper.output_file}")
    print("=" * 60)
    
    # Run the scraper
    try:
        asyncio.run(scraper.scrape_technical_documentation())
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"\nScraping failed with error: {e}")

if __name__ == "__main__":
    main()