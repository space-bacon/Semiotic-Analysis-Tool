#!/usr/bin/env python3
"""
Fast Semiotic Analysis - Simplified version for quick analysis
Skips slow operations like LIME explanations
"""

import os
import json
import sys

# Set the file to analyze
if len(sys.argv) > 1:
    TARGET_FILE = sys.argv[1]
    # If no path prefix, assume it's in data/
    if not os.path.dirname(TARGET_FILE):
        TARGET_FILE = os.path.join('data', TARGET_FILE)
else:
    TARGET_FILE = "data/Autobiography_of_a_Yogi.txt"

# Import the main script functions
os.chdir('/Users/burtron/development/semiotics/Semiotic-Analysis-Tool')
sys.path.insert(0, '/Users/burtron/development/semiotics/Semiotic-Analysis-Tool')

# Only load the file we want
print(f"🔍 Analyzing: {TARGET_FILE}")
print("=" * 60)

if not os.path.exists(TARGET_FILE):
    print(f"❌ Error: File '{TARGET_FILE}' not found")
    print(f"   Please place your .txt files in the 'data/' directory")
    exit(1)

with open(TARGET_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"✅ Loaded {len(text)} characters")
print(f"✅ Starting quick analysis...")

# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ssl

# Download NLTK data with SSL workaround
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Basic text analysis
sentences = sent_tokenize(text)
words = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.isalnum() and w not in stop_words]

print(f"\n📊 Basic Statistics:")
print(f"   Characters: {len(text):,}")
print(f"   Sentences: {len(sentences):,}")
print(f"   Words: {len(words):,}")
print(f"   Unique words: {len(set(filtered_words)):,}")

# Word frequency
word_freq = Counter(filtered_words)
print(f"\n🔤 Top 20 Most Frequent Words:")
for word, count in word_freq.most_common(20):
    print(f"   {word:20s} : {count:5d}")

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = analyzer.polarity_scores(text[:5000])  # First 5000 chars
blob = TextBlob(text[:5000])

print(f"\n😊 Sentiment Analysis:")
print(f"   VADER Compound: {sentiment_scores['compound']:.3f}")
print(f"   VADER Positive: {sentiment_scores['pos']:.3f}")
print(f"   VADER Negative: {sentiment_scores['neg']:.3f}")
print(f"   VADER Neutral: {sentiment_scores['neu']:.3f}")
print(f"   TextBlob Polarity: {blob.sentiment.polarity:.3f}")
print(f"   TextBlob Subjectivity: {blob.sentiment.subjectivity:.3f}")

# Sample sentences analysis
print(f"\n📝 Sample Sentences (first 5):")
for i, sent in enumerate(sentences[:5], 1):
    print(f"   {i}. {sent[:100]}...")

# Named entities using simple approach
print(f"\n👤 Analyzing named entities with transformers...")
from transformers import pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Analyze first 10 sentences for NER
sample_text = " ".join(sentences[:10])
entities = ner_pipeline(sample_text)

print(f"   Found {len(entities)} named entities in first 10 sentences:")
entity_counts = Counter([e['entity_group'] for e in entities])
for entity_type, count in entity_counts.most_common():
    print(f"   {entity_type:15s} : {count}")

print(f"\n   Sample entities:")
for entity in entities[:10]:
    print(f"   {entity['word']:30s} ({entity['entity_group']})")

# Save results
output = {
    "file": TARGET_FILE,
    "statistics": {
        "characters": len(text),
        "sentences": len(sentences),
        "words": len(words),
        "unique_words": len(set(filtered_words))
    },
    "top_words": dict(word_freq.most_common(50)),
    "sentiment": {
        "vader": sentiment_scores,
        "textblob": {
            "polarity": float(blob.sentiment.polarity),
            "subjectivity": float(blob.sentiment.subjectivity)
        }
    },
    "named_entities": [
        {"text": e['word'], "type": e['entity_group'], "score": float(e['score'])}
        for e in entities
    ]
}

output_file = f"output/quick_analysis_{TARGET_FILE.replace('.txt', '')}.json"
os.makedirs('output', exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Analysis complete!")
print(f"📁 Results saved to: {output_file}")
print("=" * 60)
