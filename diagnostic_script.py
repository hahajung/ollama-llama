#!/usr/bin/env python3
"""
Quick diagnostic script to test what's accessible on docs.itential.com
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def run_diagnostics():
    """Run diagnostics to see what's accessible."""
    
    print("🔍 ITENTIAL DOCS DIAGNOSTICS")
    print("=" * 60)
    
    # Test URLs
    test_urls = [
        "https://docs.itential.com/",
        "https://docs.itential.com/docs",
        "https://docs.itential.com/docs/",
        "https://docs.itential.com/sitemap.xml",
        "https://docs.itential.com/docs/getting-started",
        "https://docs.itential.com/docs/platform-6",
        "https://docs.itential.com/docs/installation"
    ]
    
    print("\n1️⃣ Testing URL accessibility:")
    accessible_urls = []
    
    for url in test_urls:
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            status = response.status_code
            if status == 200:
                print(f"  ✅ {url} - Status: {status}")
                accessible_urls.append(url)
            else:
                print(f"  ❌ {url} - Status: {status}")
        except Exception as e:
            print(f"  ❌ {url} - Error: {e}")
    
    if not accessible_urls:
        print("\n❌ No URLs are accessible. Check your internet connection.")
        return
    
    # Test HTML structure
    print("\n2️⃣ Testing HTML structure of main docs page:")
    
    # Try the first accessible URL
    main_url = accessible_urls[0]
    print(f"  Using: {main_url}")
    
    try:
        response = requests.get(main_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check title
        title = soup.find('title')
        if title:
            print(f"  📄 Page title: {title.get_text().strip()}")
        
        # Count different types of elements
        print(f"  🔗 Total links: {len(soup.find_all('a'))}")
        print(f"  📝 Total headings: {len(soup.find_all(['h1', 'h2', 'h3', 'h4']))}")
        print(f"  📦 Total divs: {len(soup.find_all('div'))}")
        
        # Check for common selectors
        selectors = ['main', '.main-content', '.content', 'article', '#content', '.docs-content']
        print("\n  🎯 Checking for content selectors:")
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                print(f"    ✅ Found: {selector}")
            else:
                print(f"    ❌ Not found: {selector}")
        
        # Find docs links
        print("\n3️⃣ Looking for /docs/ links:")
        docs_links = []
        for link in soup.find_all('a', href=True)[:100]:  # Check first 100 links
            href = link['href']
            full_url = urljoin(main_url, href)
            
            if '/docs/' in full_url and 'itential.com' in full_url:
                docs_links.append(full_url)
                if len(docs_links) <= 10:
                    print(f"  ✅ {full_url}")
        
        print(f"\n  📊 Found {len(docs_links)} documentation links")
        
        # Test if JavaScript is required
        print("\n4️⃣ Checking if site requires JavaScript:")
        if 'noscript' in str(soup):
            print("  ⚠️ Site may require JavaScript for full functionality")
        
        if 'react' in str(soup).lower() or 'vue' in str(soup).lower() or 'angular' in str(soup).lower():
            print("  ⚠️ Site appears to use JavaScript framework")
        
        # Check for API endpoints
        print("\n5️⃣ Looking for API endpoints in page:")
        page_content = str(soup)
        if '/api/' in page_content:
            print("  ✅ Found API references")
        if 'graphql' in page_content.lower():
            print("  ✅ Found GraphQL references")
        if 'rest' in page_content.lower():
            print("  ✅ Found REST references")
        
    except Exception as e:
        print(f"  ❌ Error analyzing page: {e}")
    
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTICS COMPLETE")
    
    if docs_links:
        print("\n✅ Site is accessible and contains documentation links")
        print("💡 Recommendation: Use the fixed scraper (fixed_ultimate_scraper.py)")
    else:
        print("\n⚠️ Site is accessible but no /docs/ links found")
        print("💡 The site structure may have changed or requires JavaScript")
        print("💡 You may need to use Selenium for JavaScript-rendered content")

if __name__ == "__main__":
    run_diagnostics()