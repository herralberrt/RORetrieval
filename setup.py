"""
Setup script for RORetrieval project.

Downloads necessary models and datasets.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import ensure_dir


def setup_directories():
    """Create necessary directories."""
    print(" Creating directory structure...")
    
    dirs = [
        'data/corpus',
        'data/queries',
        'data/negatives',
        'models',
        'results',
        'notebooks'
    ]
    
    for d in dirs:
        ensure_dir(d)
        print(f"  ✓ {d}/")


def download_embeddings_model():
    """Download embeddings model."""
    print("\n Downloading embeddings model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # multilingual-e5-large is published by intfloat, not sentence-transformers.
        model_name = os.getenv(
            'EMBEDDINGS_MODEL', 'intfloat/multilingual-e5-large'
        )
        print(f"  Downloading {model_name}...")
        model = SentenceTransformer(model_name)
        print(f"  ✓ Model ready: {model_name}")
        
    except ImportError:
        print("    sentence-transformers not installed")
        print("     Run: pip install -r requirements.txt")


def setup_env_file():
    """Create .env file if not exists."""
    print("\n  Setting up configuration...")
    
    if not os.path.exists('.env'):
        print("  Creating .env from .env.example...")
        os.system('cp .env.example .env')
        print("  ✓ .env created (update with your API keys)")
    else:
        print("  ✓ .env exists")


def main():
    """Run all setup steps."""
    print("\n" + "="*60)
    print("  RORetrieval - Setup")
    print("="*60)
    
    try:
        setup_directories()
        setup_env_file()
        download_embeddings_model()
        
        print("\n" + "="*60)
        print("   Setup complete!")
        print("="*60)
        print("\n  Next steps:")
        print("  1. Update .env with your API keys")
        print("  2. Run: python src/task1_prompt.py --help")
        print("  3. See README.md for more info")
        print()
        
    except Exception as e:
        print(f"\n Setup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
