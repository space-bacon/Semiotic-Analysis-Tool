# Autobiography of a Yogi - Analysis In Progress

## Current Status

**Analysis Started:** December 18, 2025 at ~12:14 PM
**File Size:** 960KB (~980,000 characters)
**Current Status:** ✅ RUNNING

## Process Details

- **PID:** 38970
- **CPU Usage:** 100% (actively processing)
- **Runtime:** 22+ minutes
- **Status:** In computationally intensive phase

## Progress

### Completed
- ✅ Phase 1: Text Processing & Extraction
  - Created: `Autobiography_of_a_Yogi.txt_discourse_tree.html` (6.2KB)
  - Extracted text and initial discourse structure

### In Progress
- 🔄 Phases 2-14: Advanced analysis phases

## Why It Takes Time

This is an exceptionally large file, and several phases are computationally intensive:

1. **Phase 3: Word2Vec Training**
   - Training neural network model on ~8,000+ sentences
   - Building semantic vector representations
   - Most CPU-intensive phase for large texts

2. **Phases 4-7: Matrix Calculations**
   - Computing syntagmatic relationships (co-occurrence)
   - Computing paradigmatic relationships (semantic similarity)
   - Matrix operations on thousands of signs

3. **Phase 8: Sentiment Analysis**
   - Analyzing sentiment for each sentence
   - Running VADER and TextBlob on entire corpus
   - Transformer-based sentiment analysis

4. **Phase 9: Topic Modeling**
   - LDA (Latent Dirichlet Allocation) on large corpus
   - Identifying themes and topics

## Expected Timeline

- **Small files (test_sample.txt):** 30-45 seconds
- **Medium files:** 2-5 minutes  
- **Large files (Autobiography):** 15-30 minutes ⏱️

## What Will Be Generated

Once complete, you'll see these files in `output/`:

1. `semiotic_analysis_output.json` - Complete analysis data with **OVERVIEW section**
2. `output_analysis.json` - Alternative JSON export
3. `output_analysis.csv` - Tabular data export
4. `output_analysis.ipynb` - Jupyter notebook
5. `syntagmatic_matrix.png` - Co-occurrence visualization
6. `paradigmatic_matrix.png` - Similarity visualization
7. `sign_network.png` - Network graph
8. `*.html` - Discourse trees (already created)
9. PDF report - Comprehensive analysis report

## New Feature: Analysis Overview

The output will now include a comprehensive overview section with:

- **Analysis Metadata**
  - Timestamp
  - Files analyzed
  - Total characters processed

- **Corpus Statistics**
  - Total signs extracted
  - Unique signs
  - Sign diversity ratio
  - System significance score

- **Sentiment Overview**
  - Average polarity
  - VADER compound scores
  - Sentiment range

- **Top Signs**
  - 10 most significant signs with scores

- **Key Insights**
  - Automatically generated insights
  - Corpus size summary
  - Sentiment characterization
  - Significance highlights

## Monitoring

The process is healthy and progressing normally. The 100% CPU usage indicates active computation (expected behavior for this workload).

## Next Steps

1. **Wait for completion** - The script will finish all 14 phases
2. **Check output/** folder for generated files
3. **View the overview** in the JSON output and terminal display
4. **Explore visualizations** - PNG files and HTML interactive graphs

---

**Status:** 🟢 ACTIVE - Analysis in progress, no errors
**Last Updated:** December 18, 2025 12:40 PM
