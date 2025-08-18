# run_enhanced_pipeline_clean.py - No emojis version

import subprocess
import sys
import time
from pathlib import Path
import os

class EnhancedPipelineRunner:
    """Complete pipeline runner for the enhanced RAG system."""
    
    def __init__(self):
        self.steps = [
            ("Enhanced Scraper", "comprehensive_technical_scraper.py"),
            ("Enhanced Embedder", "technical_embedder.py"),
            ("Test Enhanced Chatbot", "app/core/improved_chatbot.py")
        ]
        
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        print("Checking dependencies...")
        
        # Map package names to their import names
        packages_to_check = [
            ('pandas', 'pandas'),
            ('beautifulsoup4', 'bs4'), 
            ('aiohttp', 'aiohttp'),
            ('langchain', 'langchain'),
            ('chromadb', 'chromadb'),
            ('streamlit', 'streamlit'),
            ('numpy', 'numpy')
        ]
        
        missing = []
        for package_name, import_name in packages_to_check:
            try:
                __import__(import_name)
                print(f"   [OK] {package_name}")
            except ImportError:
                missing.append(package_name)
                print(f"   [MISSING] {package_name}")
        
        if missing:
            print(f"\nWARNING: Missing packages: {missing}")
            print("Install with: pip install " + " ".join(missing))
            return False
        
        print("All dependencies satisfied!")
        return True
    
    def check_ollama(self):
        """Check if Ollama is running."""
        print("\nChecking Ollama...")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("[OK] Ollama is running")
                return True
            else:
                print("[ERROR] Ollama responded with error")
                return False
        except Exception as e:
            print(f"[ERROR] Ollama not accessible: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False
    
    def setup_directories(self):
        """Create necessary directories."""
        print("\nSetting up directories...")
        
        directories = [
            "enhanced_pipeline",
            "enhanced_pipeline/data",
            "enhanced_pipeline/scrapers",
            "enhanced_pipeline/embedders", 
            "enhanced_pipeline/chatbots",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   [OK] {directory}/")
    
    def run_step(self, step_name: str, script_path: str) -> bool:
        """Run a pipeline step."""
        print(f"\n{step_name}")
        print("=" * 50)
        
        if not Path(script_path).exists():
            print(f"[ERROR] Script not found: {script_path}")
            return False
        
        try:
            start_time = time.time()
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=3600)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"[SUCCESS] {step_name} completed successfully!")
                print(f"Duration: {duration:.1f} seconds")
                
                # Show last few lines of output
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    print("Output:")
                    for line in lines[-5:]:  # Last 5 lines
                        print(f"   {line}")
                
                return True
            else:
                print(f"[ERROR] {step_name} failed!")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {step_name} timed out after 1 hour")
            return False
        except Exception as e:
            print(f"[ERROR] Error running {step_name}: {e}")
            return False
    
    def verify_outputs(self):
        """Verify that pipeline outputs were created."""
        print("\nVerifying pipeline outputs...")
        
        expected_files = [
            ("Scraped data", "comprehensive_itential_docs.jsonl"),
            ("Vector database", "technical_optimized_chroma_db"),
        ]
        
        all_good = True
        for description, path in expected_files:
            if Path(path).exists():
                if path.endswith('.jsonl'):
                    # Check file size
                    size = Path(path).stat().st_size
                    print(f"   [OK] {description}: {size:,} bytes")
                else:
                    # Check directory exists
                    print(f"   [OK] {description}: directory exists")
            else:
                print(f"   [MISSING] {description}: not found at {path}")
                all_good = False
        
        return all_good
    
    def run_complete_pipeline(self):
        """Run the complete enhanced pipeline."""
        print("Enhanced RAG Pipeline Runner")
        print("=" * 50)
        
        # Pre-flight checks
        if not self.check_dependencies():
            return False
        
        if not self.check_ollama():
            return False
        
        self.setup_directories()
        
        # Run pipeline steps
        print(f"\nRunning {len(self.steps)} pipeline steps...")
        
        for i, (step_name, script_path) in enumerate(self.steps, 1):
            print(f"\nStep {i}/{len(self.steps)}: {step_name}")
            
            if not self.run_step(step_name, script_path):
                print(f"\n[FAILED] Pipeline failed at step {i}")
                return False
            
            # Brief pause between steps
            if i < len(self.steps):
                time.sleep(2)
        
        # Verify outputs
        if not self.verify_outputs():
            print("\n[WARNING] Pipeline completed but some outputs missing")
            return False
        
        print("\n[SUCCESS] Enhanced RAG Pipeline Completed Successfully!")
        print("\nNext Steps:")
        print("   1. Your vector database is at: technical_optimized_chroma_db")
        print("   2. Test with queries like:")
        print("      - 'Tell me all the MongoDB properties for properties.json'")
        print("      - 'What versions of IAP are there?'")
        print("      - 'Python version for IAP 2023.1?'")
        print("   3. Run: streamlit run app/enhanced_ui.py")
        
        return True

def main():
    """Main function to run the enhanced pipeline."""
    runner = EnhancedPipelineRunner()
    
    try:
        success = runner.run_complete_pipeline()
        if success:
            print("\n[SUCCESS] Pipeline completed successfully!")
        else:
            print("\n[FAILED] Pipeline failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRASHED] Pipeline crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()