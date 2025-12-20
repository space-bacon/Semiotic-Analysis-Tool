#!/usr/bin/env python3
"""
Test script to verify the Semiotic Analysis Tool setup
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import numpy as np
        print("✅ numpy:", np.__version__)
        
        import nltk
        print("✅ nltk:", nltk.__version__)
        
        import sklearn
        print("✅ scikit-learn:", sklearn.__version__)
        
        import torch
        print("✅ torch:", torch.__version__)
        
        import transformers
        print("✅ transformers:", transformers.__version__)
        
        import pandas as pd
        print("✅ pandas:", pd.__version__)
        
        import matplotlib
        print("✅ matplotlib:", matplotlib.__version__)
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_nltk_data():
    """Test that NLTK data is available"""
    print("\nTesting NLTK data packages...")
    
    try:
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        
        # Test tokenization
        test_text = "This is a test sentence. Here is another one!"
        words = word_tokenize(test_text)
        sentences = sent_tokenize(test_text)
        
        print(f"✅ Word tokenization: {len(words)} tokens")
        print(f"✅ Sentence tokenization: {len(sentences)} sentences")
        
        # Test stopwords
        stop_words = set(stopwords.words('english'))
        print(f"✅ Stopwords loaded: {len(stop_words)} English stopwords")
        
        return True
    except Exception as e:
        print(f"❌ NLTK data error: {e}")
        return False

def test_transformers():
    """Test that transformers models can be loaded"""
    print("\nTesting transformer models...")
    
    try:
        from transformers import pipeline
        
        # This will use a cached model if available
        print("Loading sentiment analysis pipeline (this may take a moment)...")
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Test the pipeline
        result = sentiment_analyzer("This is a great tool!")[0]
        print(f"✅ Sentiment analysis: {result['label']} (score: {result['score']:.2f})")
        
        return True
    except Exception as e:
        print(f"⚠️  Transformer test: {e}")
        print("   Note: This is optional, models will download on first use")
        return True  # Don't fail on this

def test_environment_variables():
    """Check environment variable configuration"""
    print("\nChecking environment variables...")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    encryption_key = os.getenv("ENCRYPTION_KEY")
    
    if google_api_key and google_api_key != "your_google_api_key_here":
        print("✅ GOOGLE_API_KEY is configured")
    else:
        print("⚠️  GOOGLE_API_KEY not configured (optional)")
    
    if encryption_key and encryption_key != "your_encryption_key_here":
        print("✅ ENCRYPTION_KEY is configured")
    else:
        print("⚠️  ENCRYPTION_KEY not configured (optional)")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Semiotic Analysis Tool - Setup Verification")
    print("=" * 60)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("NLTK Data", test_nltk_data),
        ("Transformers", test_transformers),
        ("Environment Variables", test_environment_variables),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    if all(results):
        print("🎉 All tests passed! Setup is complete and ready to use.")
        print("\nNext steps:")
        print("1. Place input files (.txt or images) in this directory")
        print("2. Run: python semiotic_analysis_tool.py")
        print("3. Check the generated output files")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
