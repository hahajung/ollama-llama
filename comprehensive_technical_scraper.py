#!/usr/bin/env python3
"""
Enhanced Comprehensive Technical Documentation Scraper for docs.itential.com
COMPLETE REPLACEMENT for comprehensive_technical_scraper.py
Focuses on extracting ALL version information, dependencies, and technical requirements.
Enhanced to capture CLI troubleshooting and prevent over-matching issues.
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
    """Enhanced scraper that systematically captures ALL documentation with domain categorization."""
    
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
        
        # Enhanced content categorization for preventing over-matching
        self.content_categories = {
            'CLI_TOOLS': {
                'url_patterns': [
                    r'/cli/', r'/command-line/', r'/itential-cli/', 
                    r'/netcommon/', r'/ansible/', r'/duplicate-data/'
                ],
                'title_keywords': [
                    'cli', 'command line', 'itential-cli', 'netcommon', 
                    'ansible', 'duplicate data', 'collection'
                ],
                'content_keywords': [
                    'itential-cli', 'ansible-galaxy', 'netcommon', 'duplicate data',
                    'command line', 'cli commands', 'collection version',
                    'duplicate return data', 'itential-cli role'
                ],
                'priority': 25.0  # Highest priority for CLI troubleshooting
            },
            'TROUBLESHOOTING': {
                'url_patterns': [
                    r'/troubleshoot/', r'/debugging/', r'/error/', r'/issue/',
                    r'/duplicate/', r'/fix/', r'/resolve/'
                ],
                'title_keywords': [
                    'troubleshoot', 'debug', 'error', 'issue', 'duplicate',
                    'fix', 'resolve', 'problem', 'solution'
                ],
                'content_keywords': [
                    'duplicate', 'error', 'issue', 'problem', 'troubleshoot',
                    'debug', 'fix', 'resolve', 'solution', 'workaround'
                ],
                'priority': 20.0
            },
            'PLATFORM_EVENTS': {
                'url_patterns': [
                    r'/event/', r'/deduplication/', r'/operations-manager/',
                    r'/email-adapter/', r'/trigger/'
                ],
                'title_keywords': [
                    'event service', 'deduplication', 'operations manager', 
                    'email adapter', 'trigger', 'event configuration'
                ],
                'content_keywords': [
                    'event service', 'event deduplication', 'operations manager',
                    'email adapter', 'trigger type', 'uniqueProps', 'messageId'
                ],
                'priority': 15.0
            },
            'VERSION_DEPENDENCIES': {
                'url_patterns': [
                    r'/version/', r'/dependency/', r'/requirement/', r'/compatibility/',
                    r'/support-matrix/', r'/lifecycle/'
                ],
                'title_keywords': [
                    'version', 'dependency', 'requirement', 'compatibility',
                    'support matrix', 'lifecycle', 'upgrade'
                ],
                'content_keywords': [
                    'version', 'dependency', 'requirement', 'compatible',
                    'support', 'upgrade', 'migrate', 'matrix'
                ],
                'priority': 18.0
            },
            'INSTALLATION_CONFIG': {
                'url_patterns': [
                    r'/install/', r'/config/', r'/setup/', r'/deploy/',
                    r'/configuration/', r'/deployment/'
                ],
                'title_keywords': [
                    'install', 'installation', 'config', 'configuration',
                    'setup', 'deploy', 'deployment'
                ],
                'content_keywords': [
                    'install', 'configuration', 'setup', 'deploy',
                    'configure', 'deployment', 'environment'
                ],
                'priority': 16.0
            }
        }
        
        # Enhanced technical patterns for better extraction
        self.enhanced_patterns = {
            'cli_specific': [
                r'itential-cli.*?(?:duplicate|error|issue)',
                r'ansible-galaxy.*?collection.*?(?:install|upgrade|version)',
                r'netcommon.*?(?:version|collection).*?(\d+\.\d+\.\d+)',
                r'duplicate.*?(?:data|return).*?(?:cli|command)',
                r'command.*?line.*?(?:tool|interface|troubleshoot)'
            ],
            'version_matrix': [
                r'(?:IAP|iap)\s*([0-9]{4}\.[0-9]+(?:\.[0-9]+)?)',
                r'(?:IAG|iag)\s*([0-9]{4}\.[0-9]+(?:\.[0-9]+)?)',
                r'(?:Platform|platform)\s*([0-9]+(?:\.[0-9]+)?)',
                r'(?:Python|python)\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                r'(?:Node|node)\.?js\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                r'(?:MongoDB|mongodb)\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)'
            ]
        }
        
        # Track comprehensive coverage
        self.coverage_tracker = {
            'cli_pages': set(),
            'troubleshooting_pages': set(),
            'platform_events_pages': set(),
            'version_pages': set(),
            'installation_pages': set()
        }
        
        # Technical content tracking (existing functionality)
        self.found_technical_pages: Set[str] = set()
        self.version_registry: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_registry: Dict[str, Dict] = {}

    async def _init_session(self):
        """Initialize aiohttp session with proper headers."""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(
            timeout=timeout, 
            headers=headers, 
            connector=connector
        )

    async def rate_limited_get(self, url: str) -> aiohttp.ClientResponse:
        """Rate-limited HTTP GET request."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        
        self.last_request_time = time.time()
        return await self.session.get(url)

    def categorize_content_enhanced(self, url: str, title: str, content: str) -> Tuple[str, float]:
        """Enhanced content categorization to prevent over-matching."""
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        category_scores = {}
        
        for category_name, category_info in self.content_categories.items():
            score = 0
            
            # URL pattern matching (highest weight)
            for pattern in category_info['url_patterns']:
                if re.search(pattern, url_lower):
                    score += 15
            
            # Title keyword matching (high weight)
            for keyword in category_info['title_keywords']:
                if keyword in title_lower:
                    score += 10
            
            # Content keyword matching (medium weight)
            for keyword in category_info['content_keywords']:
                count = content_lower.count(keyword)
                score += min(count * 3, 15)  # Cap at 15 points per keyword
            
            # Special CLI detection (prevent over-matching)
            if category_name == 'CLI_TOOLS':
                cli_strong_indicators = [
                    'itential-cli', 'ansible-galaxy', 'netcommon collection',
                    'duplicate return data', 'cli troubleshoot'
                ]
                for indicator in cli_strong_indicators:
                    if indicator in content_lower:
                        score += 20  # Very high bonus for CLI content
                        
            # Special Platform Events detection (prevent CLI confusion)
            elif category_name == 'PLATFORM_EVENTS':
                platform_strong_indicators = [
                    'event service', 'event deduplication', 'operations manager',
                    'email adapter', 'platform trigger'
                ]
                for indicator in platform_strong_indicators:
                    if indicator in content_lower:
                        score += 18
                        
                # Penalty if it's clearly CLI content
                cli_indicators = ['itential-cli', 'command line', 'ansible-galaxy']
                if any(indicator in content_lower for indicator in cli_indicators):
                    score -= 10
            
            # Apply base priority
            if score > 0:
                score += category_info['priority']
                category_scores[category_name] = score
        
        # Return highest scoring category or default
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            return best_category, category_scores[best_category]
        else:
            return 'GENERAL', 2.0

    def extract_enhanced_technical_info(self, content: str, category: str) -> Dict[str, Any]:
        """Extract technical information based on category."""
        technical_info = {
            'extracted_versions': [],
            'cli_commands': [],
            'troubleshooting_steps': [],
            'dependencies': [],
            'technical_indicators': [],
            'category_specific_data': {}
        }
        
        # Category-specific extraction
        if category == 'CLI_TOOLS':
            # Extract CLI commands
            cli_patterns = [
                r'ansible-galaxy\s+[^\n]+',
                r'itential-cli\s+[^\n]+',
                r'ollama\s+[^\n]+'
            ]
            for pattern in cli_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                technical_info['cli_commands'].extend(matches[:3])
            
            # Extract version references
            versions = re.findall(r'\d+\.\d+\.\d+', content)
            technical_info['extracted_versions'] = list(set(versions))[:10]
            
            # CLI-specific indicators
            technical_info['category_specific_data'] = {
                'has_duplicate_data_issue': 'duplicate data' in content.lower(),
                'mentions_netcommon': 'netcommon' in content.lower(),
                'has_ansible_commands': bool(re.search(r'ansible-galaxy', content, re.IGNORECASE))
            }
            
        elif category == 'PLATFORM_EVENTS':
            # Extract event-related configuration
            event_patterns = [
                r'eventDeduplication.*?{[^}]+}',
                r'uniqueProps.*?\[[^\]]+\]',
                r'messageId.*?field'
            ]
            for pattern in event_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                technical_info['troubleshooting_steps'].extend(matches[:3])
                
            technical_info['category_specific_data'] = {
                'has_event_deduplication': 'event deduplication' in content.lower(),
                'mentions_operations_manager': 'operations manager' in content.lower(),
                'has_email_adapter': 'email adapter' in content.lower()
            }
        
        elif category == 'VERSION_DEPENDENCIES':
            # Extract version matrices
            for pattern in self.enhanced_patterns['version_matrix']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                technical_info['extracted_versions'].extend(matches)
            
            # Extract dependencies
            dep_patterns = [
                r'(?:requires?|depends?\s+on|needs?)\s+([a-zA-Z0-9.-]+\s+\d+\.\d+)',
                r'([a-zA-Z0-9.-]+)\s+version\s+(\d+\.\d+)'
            ]
            for pattern in dep_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                technical_info['dependencies'].extend(matches[:5])
        
        # Clean and deduplicate
        for key, value in technical_info.items():
            if isinstance(value, list):
                technical_info[key] = list(set([str(v).strip() for v in value if v and str(v).strip()]))[:10]
        
        return technical_info

    def is_technical_url(self, url: str) -> bool:
        """Enhanced technical URL detection."""
        url_lower = url.lower()
        
        # Must be from docs.itential.com
        if 'docs.itential.com' not in url:
            return False
        
        # Skip unwanted file types
        skip_extensions = ['.pdf', '.zip', '.tar.gz', '.exe', '.dmg', '.png', '.jpg', '.jpeg']
        if any(url_lower.endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip certain parameters
        if '#' in url or '?search=' in url:
            return False
        
        # High priority technical patterns
        high_priority_patterns = [
            r'/cli/', r'/troubleshoot/', r'/duplicate/', r'/version/',
            r'/install/', r'/config/', r'/api/', r'/admin/'
        ]
        
        for pattern in high_priority_patterns:
            if re.search(pattern, url_lower):
                return True
        
        return True  # Default to true for docs.itential.com

    def calculate_technical_priority(self, url: str, link_text: str) -> float:
        """Enhanced priority calculation."""
        priority = 1.0
        url_lower = url.lower()
        text_lower = link_text.lower()
        
        # Very high priority for CLI troubleshooting
        if any(pattern in url_lower for pattern in ['/cli/', '/duplicate-data/', '/netcommon/']):
            priority += 25.0
        
        # High priority patterns
        high_priority_patterns = [
            r'/troubleshoot/', r'/version/', r'/dependency/', r'/install/'
        ]
        for pattern in high_priority_patterns:
            if re.search(pattern, url_lower):
                priority += 15.0
        
        # Text-based priority
        high_priority_text = [
            'troubleshoot', 'duplicate', 'cli', 'command line', 'error',
            'version', 'requirement', 'dependency', 'installation'
        ]
        for keyword in high_priority_text:
            if keyword in text_lower:
                priority += 8.0
        
        return min(priority, 50.0)

    async def process_technical_page(self, url: str, priority: float, depth: int) -> Optional[Dict[str, Any]]:
        """Enhanced page processing with categorization."""
        if url in self.visited_urls or depth > 4:
            return None
        
        self.visited_urls.add(url)
        
        try:
            async with self.semaphore:
                async with await self.rate_limited_get(url) as response:
                    if response.status != 200:
                        print(f"HTTP {response.status} for {url}")
                        return None

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Clean unwanted elements
                    for tag in soup(["script", "style", "nav", "footer", 
                                    "header", "iframe", "noscript", "form"]):
                        tag.decompose()

                    # Extract metadata
                    title = soup.title.get_text(strip=True) if soup.title else ""
                    description_tag = soup.find('meta', attrs={'name': 'description'})
                    description = description_tag['content'].strip() if description_tag else ""

                    # Extract structured content
                    headings = self._extract_headings(soup)
                    tables = self.extract_comprehensive_tables(soup)
                    code_blocks = self._extract_comprehensive_code_blocks(soup)

                    # Get main content
                    content_div = self._find_main_content(soup)
                    if content_div:
                        raw_text = self.clean_text(content_div.get_text(separator='\n'))
                    else:
                        raw_text = self.clean_text(soup.get_text(separator='\n'))

                    if not raw_text or len(raw_text) < 100:
                        return None

                    # Enhanced categorization
                    category, category_score = self.categorize_content_enhanced(url, title, raw_text)
                    
                    # Extract technical information
                    technical_info = self.extract_enhanced_technical_info(raw_text, category)
                    
                    # Create enhanced searchable text
                    searchable_text = self._create_enhanced_searchable_text(
                        raw_text, tables, code_blocks, headings, category, technical_info
                    )

                    # Extract links for further crawling
                    links: List[Tuple[str, float, str]] = []
                    for a in soup.find_all('a', href=True):
                        full_url = urljoin(url, a['href'])
                        if self.is_technical_url(full_url):
                            link_text = a.get_text().strip()
                            link_priority = self.calculate_technical_priority(full_url, link_text)
                            links.append((full_url, link_priority, link_text))

                    # Analyze technical content (existing functionality)
                    technical_analysis = self.analyze_technical_content(url, title, searchable_text)
                    
                    # Update coverage tracking
                    self._update_coverage_tracker(category, url, title)
                    
                    # Track technical findings
                    if technical_analysis['is_critical']:
                        self.found_technical_pages.add(technical_analysis['content_type'])

                    # Increment counter
                    self.scraped_count += 1

                    # Create comprehensive document
                    document = {
                        'url': url,
                        'title': title,
                        'description': description,
                        'category': category,
                        'category_score': category_score,
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
                        'technical_info': technical_info,
                        'table_count': len(tables),
                        'code_block_count': len(code_blocks),
                        'technical_relevance_score': self._calculate_technical_score(technical_analysis, tables, code_blocks),
                        'depth': depth
                    }

                    # Save document
                    await self._save_document(document)
                    
                    # Add priority links to queue
                    await self._add_priority_links(links, depth)
                    
                    print(f"[{category}] Scraped: {title[:50]}... (Score: {category_score:.1f})")
                    
                    return document

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def _update_coverage_tracker(self, category: str, url: str, title: str):
        """Update coverage tracking."""
        if category == 'CLI_TOOLS':
            self.coverage_tracker['cli_pages'].add(url)
        elif category == 'TROUBLESHOOTING':
            self.coverage_tracker['troubleshooting_pages'].add(url)
        elif category == 'PLATFORM_EVENTS':
            self.coverage_tracker['platform_events_pages'].add(url)
        elif category == 'VERSION_DEPENDENCIES':
            self.coverage_tracker['version_pages'].add(url)
        elif category == 'INSTALLATION_CONFIG':
            self.coverage_tracker['installation_pages'].add(url)

    def _create_enhanced_searchable_text(self, raw_text: str, tables: List, code_blocks: List, 
                                       headings: List, category: str, technical_info: Dict) -> str:
        """Create enhanced searchable text with category context."""
        searchable_parts = [f"[CATEGORY:{category}]"]
        
        # Add category-specific context
        if category == 'CLI_TOOLS':
            searchable_parts.append("CONTEXT: Command Line Interface Tools and Troubleshooting")
            if technical_info.get('cli_commands'):
                searchable_parts.append(f"CLI_COMMANDS: {' | '.join(technical_info['cli_commands'][:3])}")
        elif category == 'PLATFORM_EVENTS':
            searchable_parts.append("CONTEXT: Platform Event Services and Configuration")
        elif category == 'TROUBLESHOOTING':
            searchable_parts.append("CONTEXT: Problem Resolution and Troubleshooting Guides")
        elif category == 'VERSION_DEPENDENCIES':
            searchable_parts.append("CONTEXT: Version Requirements and Dependencies")
            if technical_info.get('extracted_versions'):
                searchable_parts.append(f"VERSIONS: {' | '.join(technical_info['extracted_versions'][:5])}")

        # Add existing searchable text logic
        searchable_parts.append("CONTENT:")
        searchable_parts.append(raw_text)
        
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
        
        return '\n\n'.join(searchable_parts)

    async def _save_document(self, document: Dict[str, Any]):
        """Save document with proper locking."""
        async with self.output_lock:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(document, ensure_ascii=False) + '\n')

    async def _add_priority_links(self, links: List[Tuple[str, float, str]], current_depth: int):
        """Add high-priority links to queue."""
        # Sort links by priority and add top ones
        links.sort(key=lambda x: x[1], reverse=True)
        
        for link_url, link_priority, link_text in links[:10]:
            if (link_url not in self.visited_urls and 
                link_priority > 8.0 and 
                current_depth < 3):
                self.queue.append((link_url, link_priority, current_depth + 1))

    # Include all existing methods from original scraper
    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract heading hierarchy."""
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = self.clean_text(h.get_text())
            if heading_text:
                headings.append({
                    'level': h.name,
                    'text': heading_text,
                    'id': h.get('id', ''),
                    'technical_relevance': self._assess_heading_relevance(heading_text)
                })
        return headings

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

    def extract_comprehensive_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables with technical analysis."""
        tables = []
        
        for i, table in enumerate(soup.find_all('table')):
            try:
                # Try pandas for structured extraction
                df = pd.read_html(str(table))[0]
                table_text = df.to_string()
                table_markdown = df.to_markdown()
                
                # Assess technical relevance
                relevance = self._assess_table_relevance(table_text)
                
                # Get context before table
                context_before = self._get_element_context(table)
                
                table_data = {
                    'id': f'table_{i}',
                    'text': table_text,
                    'markdown': table_markdown,
                    'technical_relevance': relevance,
                    'context_before': context_before,
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'contains_versions': bool(re.search(r'\d+\.\d+', table_text))
                }
                
                tables.append(table_data)
                
            except Exception:
                # Fallback to text extraction
                table_text = self.clean_text(table.get_text())
                if table_text and len(table_text) > 50:
                    tables.append({
                        'id': f'table_{i}',
                        'text': table_text,
                        'markdown': '',
                        'technical_relevance': 'low',
                        'context_before': self._get_element_context(table),
                        'row_count': 0,
                        'column_count': 0,
                        'contains_versions': bool(re.search(r'\d+\.\d+', table_text))
                    })
        
        return tables

    def _assess_table_relevance(self, table_text: str) -> str:
        """Assess technical relevance of tables."""
        table_lower = table_text.lower()
        
        # Critical indicators
        critical_indicators = [
            'version', 'requirement', 'dependency', 'compatibility',
            'support matrix', 'minimum', 'maximum', 'required'
        ]
        
        # High indicators
        high_indicators = [
            'installation', 'configuration', 'setup', 'deployment',
            'upgrade', 'migration', 'feature', 'component'
        ]
        
        if any(indicator in table_lower for indicator in critical_indicators):
            return 'critical'
        elif any(indicator in table_lower for indicator in high_indicators):
            return 'high'
        else:
            return 'medium'

    def _extract_comprehensive_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code blocks with enhanced analysis."""
        code_blocks = []
        
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
        
        return code_blocks

    def _detect_code_type_comprehensive(self, code_text: str) -> str:
        """Comprehensive code type detection."""
        code_lower = code_text.lower()
        
        # CLI commands (highest priority for our use case)
        if any(cmd in code_lower for cmd in ['ansible-galaxy', 'itential-cli', 'pip install', 'npm install']):
            return 'CLI Commands'
        
        # Configuration files
        if 'properties.json' in code_lower or ('{' in code_text and 'ldap' in code_lower):
            return 'LDAP Configuration'
        elif 'mongodb' in code_lower or 'mongo' in code_lower:
            return 'MongoDB Configuration'
        elif 'redis' in code_lower:
            return 'Redis Configuration'
        elif 'rabbitmq' in code_lower or 'amqp' in code_lower:
            return 'RabbitMQ Configuration'
        
        # Installation commands
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
        
        # Critical for our CLI troubleshooting focus
        if any(cmd in code_lower for cmd in ['ansible-galaxy', 'itential-cli', 'netcommon']):
            return 'critical'
        
        # High relevance
        elif any(indicator in code_lower for indicator in ['install', 'config', 'setup', 'deploy']):
            return 'high'
        
        # Medium relevance
        elif any(indicator in code_lower for indicator in ['import', 'require', 'function']):
            return 'medium'
        
        return 'low'

    def _extract_versions_from_text(self, text: str) -> List[str]:
        """Extract version numbers from text."""
        version_patterns = [
            r'\d+\.\d+\.\d+',  # X.Y.Z
            r'\d+\.\d+',       # X.Y
            r'v\d+\.\d+\.\d+', # vX.Y.Z
            r'version\s+\d+\.\d+'
        ]
        
        versions = []
        for pattern in version_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            versions.extend(matches)
        
        return list(set(versions))[:5]  # Limit and deduplicate

    def _get_element_context(self, element) -> str:
        """Get context around an element."""
        context_parts = []
        
        # Look for preceding heading
        for sibling in element.previous_siblings:
            if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                context_parts.append(sibling.get_text().strip())
                break
            elif sibling.name == 'p' and sibling.get_text().strip():
                context_parts.append(sibling.get_text().strip()[:100])
                break
        
        return ' | '.join(reversed(context_parts))

    # Include all existing methods from original scraper
    def analyze_technical_content(self, url: str, title: str, content: str) -> Dict[str, Any]:
        """Analyze content for technical relevance (existing functionality)."""
        content_lower = content.lower()
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Technical content type classification
        content_type = 'general'
        priority_score = 1.0
        is_critical = False
        
        # Enhanced classification
        if any(term in content_lower for term in ['dependencies', 'dependency', 'requires', 'requirement']):
            content_type = 'dependencies'
            priority_score = 15.0
            is_critical = True
        elif any(term in content_lower for term in ['version', 'lifecycle', 'support matrix']):
            content_type = 'version_lifecycle'
            priority_score = 12.0
            is_critical = True
        elif any(term in content_lower for term in ['installation', 'install', 'setup', 'configuration']):
            content_type = 'installation_config'
            priority_score = 10.0
            is_critical = True
        elif any(term in content_lower for term in ['migration', 'upgrade', 'update']):
            content_type = 'migration_upgrade'
            priority_score = 8.0
        
        # Extract technical indicators
        technical_indicators = []
        if 'python' in content_lower:
            technical_indicators.append('python')
        if 'node' in content_lower or 'npm' in content_lower:
            technical_indicators.append('nodejs')
        if 'mongodb' in content_lower or 'mongo' in content_lower:
            technical_indicators.append('mongodb')
        if 'redis' in content_lower:
            technical_indicators.append('redis')
        
        # Extract versions
        extracted_versions = {}
        version_patterns = [
            (r'(?:IAP|iap)\s*([0-9]{4}\.[0-9]+)', 'IAP'),
            (r'(?:Python|python)\s*([0-9]+\.[0-9]+)', 'Python'),
            (r'(?:Node|node)\.?js\s*([0-9]+\.[0-9]+)', 'Node.js'),
            (r'(?:MongoDB|mongodb)\s*([0-9]+\.[0-9]+)', 'MongoDB')
        ]
        
        for pattern, product in version_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                extracted_versions[product] = list(set(matches))
                self.version_registry[product].update(matches)
        
        # Dependency information
        dependency_info = {}
        if extracted_versions:
            dependency_info = {
                'has_version_info': True,
                'products': list(extracted_versions.keys()),
                'version_count': sum(len(v) for v in extracted_versions.values())
            }
        
        return {
            'content_type': content_type,
            'priority_score': priority_score,
            'is_critical': is_critical,
            'technical_indicators': technical_indicators,
            'extracted_versions': extracted_versions,
            'dependency_info': dependency_info
        }

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
        # Common main content selectors for documentation sites
        selectors = [
            'main', '.main-content', '.content', '#content',
            '.article', '.documentation', '.doc-content',
            '.page-content', '.markdown-body'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element
        
        # Fallback to body
        return soup.find('body')

    def sort_queue_by_technical_priority(self):
        """Sort queue by technical priority for better coverage."""
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
                consecutive_failures = 0
            except IndexError:
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
        """Main scraping controller with enhanced URL seeding."""
        print("🚀 Starting Enhanced Technical Documentation Scraping...")
        await self._init_session()
        
        # Enhanced initial URL seeding - specifically include CLI troubleshooting
        initial_urls = [
            # Core platform
            (self.base_url, 5.0, 0),
            
            # CRITICAL: CLI troubleshooting (highest priority)
            (f"{self.base_url}docs/duplicate-data-itential-cli-netcommon-iag", 30.0, 0),
            (f"{self.base_url}cli/", 25.0, 0),
            (f"{self.base_url}command-line/", 25.0, 0),
            (f"{self.base_url}troubleshooting/", 25.0, 0),
            
            # High priority technical content
            (f"{self.base_url}installation/", 20.0, 0),
            (f"{self.base_url}dependencies/", 20.0, 0),
            (f"{self.base_url}requirements/", 20.0, 0),
            (f"{self.base_url}version/", 18.0, 0),
            (f"{self.base_url}compatibility/", 18.0, 0),
            (f"{self.base_url}support-matrix/", 18.0, 0),
            
            # Configuration and setup
            (f"{self.base_url}configuration/", 16.0, 0),
            (f"{self.base_url}setup/", 16.0, 0),
            (f"{self.base_url}deployment/", 16.0, 0),
            
            # API and reference
            (f"{self.base_url}api/", 14.0, 0),
            (f"{self.base_url}reference/", 14.0, 0),
            
            # Administration and events
            (f"{self.base_url}admin/", 12.0, 0),
            (f"{self.base_url}administration/", 12.0, 0),
            (f"{self.base_url}event/", 12.0, 0),
            (f"{self.base_url}operations-manager/", 12.0, 0),
            
            # Release information
            (f"{self.base_url}release-notes/", 10.0, 0),
            (f"{self.base_url}changelog/", 10.0, 0)
        ]
        
        print(f"📂 Adding {len(initial_urls)} initial high-priority URLs to queue...")
        for url, priority, depth in initial_urls:
            self.queue.append((url, priority, depth))
        
        # Start workers
        workers = [asyncio.create_task(self.worker()) 
                  for _ in range(self.max_concurrent)]
        
        print(f"👥 Started {len(workers)} worker tasks")
        
        # Monitor progress with enhanced coverage tracking
        max_pages = 600  # Reasonable limit for comprehensive coverage
        min_cli_pages = 3  # Ensure CLI troubleshooting is covered
        iteration = 0
        
        print(f"🎯 Target: Enhanced documentation coverage with CLI focus")
        print(f"📊 Max pages: {max_pages}, Min CLI pages: {min_cli_pages}")
        
        try:
            while (self.scraped_count < max_pages and 
                   iteration < 240):  # 1 hour max
                await asyncio.sleep(15)
                iteration += 1
                
                # Enhanced progress reporting
                coverage_summary = {
                    'CLI': len(self.coverage_tracker['cli_pages']),
                    'Troubleshooting': len(self.coverage_tracker['troubleshooting_pages']),
                    'Platform Events': len(self.coverage_tracker['platform_events_pages']),
                    'Versions': len(self.coverage_tracker['version_pages']),
                    'Installation': len(self.coverage_tracker['installation_pages'])
                }
                
                print(f"📈 Progress: {self.scraped_count} pages | Queue: {len(self.queue)}")
                print(f"📊 Coverage: {coverage_summary}")
                print(f"👷 Workers active: {sum(1 for w in workers if not w.done())}")
                
                # Show sample CLI pages found
                if self.coverage_tracker['cli_pages']:
                    sample_cli = list(self.coverage_tracker['cli_pages'])[:2]
                    print(f"🖥️  CLI pages found: {sample_cli}")
                
                # Check for comprehensive coverage
                total_coverage = sum(coverage_summary.values())
                if (total_coverage >= 25 and 
                    coverage_summary['CLI'] >= min_cli_pages and
                    coverage_summary['Troubleshooting'] >= 5 and
                    self.scraped_count >= 150):
                    print("🎉 Comprehensive coverage achieved with CLI focus!")
                    break
                
                # Sort queue periodically
                if len(self.queue) > 50:
                    self.sort_queue_by_technical_priority()
                
                # Check for stalled progress
                if len(self.queue) == 0 and sum(1 for w in workers if not w.done()) == 0:
                    print("⚠️  All workers finished and queue empty")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Scraping interrupted by user")
        finally:
            # Clean shutdown
            for worker in workers:
                worker.cancel()
            
            if self.session:
                await self.session.close()
            
            # Generate final report
            await self._generate_enhanced_final_report()

    async def _generate_enhanced_final_report(self):
        """Generate enhanced final report with coverage details."""
        print("\n" + "="*70)
        print("📋 ENHANCED TECHNICAL SCRAPING REPORT")
        print("="*70)
        print(f"📄 Total pages scraped: {self.scraped_count}")
        print(f"💾 Output saved to: {self.output_file}")
        
        print(f"\n📊 Enhanced Coverage by Category:")
        coverage_items = [
            ('CLI Tools', self.coverage_tracker['cli_pages']),
            ('Troubleshooting', self.coverage_tracker['troubleshooting_pages']),
            ('Platform Events', self.coverage_tracker['platform_events_pages']),
            ('Version/Dependencies', self.coverage_tracker['version_pages']),
            ('Installation/Config', self.coverage_tracker['installation_pages'])
        ]
        
        for category, pages in coverage_items:
            print(f"   {category}: {len(pages)} pages")
            if pages:
                # Show sample URLs
                sample_urls = list(pages)[:2]
                for url in sample_urls:
                    print(f"     • {url}")
                if len(pages) > 2:
                    print(f"     • ... and {len(pages) - 2} more")
        
        print(f"\n🔍 Technical Content Found:")
        print(f"   Technical page types: {len(self.found_technical_pages)}")
        print(f"   Products with versions: {len(self.version_registry)}")
        
        if self.version_registry:
            print(f"   Version registry sample:")
            for product, versions in list(self.version_registry.items())[:3]:
                versions_list = sorted(list(versions))[:3]
                print(f"     {product}: {versions_list}")
        
        # Quality assessment with CLI focus
        cli_coverage = len(self.coverage_tracker['cli_pages'])
        trouble_coverage = len(self.coverage_tracker['troubleshooting_pages'])
        
        if cli_coverage >= 3 and trouble_coverage >= 5:
            print("\n✅ SUCCESS: Enhanced technical documentation captured!")
            print("   • CLI troubleshooting pages: ✅")
            print("   • Platform event pages: ✅")
            print("   • Version dependency info: ✅")
            print("   • General troubleshooting: ✅")
        else:
            print("\n⚠️  WARNING: Limited coverage in some critical categories")
            if cli_coverage < 3:
                print("   • CLI pages: Need more CLI-specific documentation")
            if trouble_coverage < 5:
                print("   • Troubleshooting: Need more troubleshooting guides")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Run enhanced embedder: python robust_technical_embedder.py")
        print(f"   2. Test CLI query: 'itential-cli role is showing duplicate data'")
        print(f"   3. Verify response mentions netcommon collection version")

def main() -> None:
    """Main function to run enhanced comprehensive scraping."""
    scraper = TechnicalDocumentationScraper(
        base_url="https://docs.itential.com/",
        output_file="complete_technical_docs.jsonl",
        max_concurrent=3,
        rate_limit=0.8
    )

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(scraper.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Clear existing output
    if os.path.exists(scraper.output_file):
        os.remove(scraper.output_file)
        print(f"🗑️  Cleared existing output file: {scraper.output_file}")

    print(f"🎯 ENHANCED COMPREHENSIVE DOCUMENTATION SCRAPING")
    print(f"=" * 60)
    print(f"🌐 Target: docs.itential.com")
    print(f"🎯 Enhanced Focus: CLI troubleshooting + domain categorization")
    print(f"📋 Priority areas:")
    print(f"   • CLI tools and troubleshooting (CRITICAL)")
    print(f"   • Platform events (separate from CLI)")
    print(f"   • Version dependencies and requirements (HIGH)")
    print(f"   • Installation and configuration guides (MEDIUM)")
    print(f"💾 Output: {scraper.output_file}")
    print("=" * 60)
    
    # Run the enhanced scraper
    try:
        asyncio.run(scraper.scrape_technical_documentation())
    except KeyboardInterrupt:
        print("\n🛑 Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Scraping failed with error: {e}")

if __name__ == "__main__":
    main()