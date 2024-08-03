# Semiotic Analysis Tool

## Overview

The Semiotic Analysis Tool is a comprehensive and sophisticated Python-based application designed to analyze various sign systems within textual and visual data. This tool integrates multiple advanced NLP techniques, machine learning models, and external knowledge sources to provide an in-depth analysis of the meaning and context of the input data.

## Features

1. **Data Input and Preprocessing**:
   - Supports text and image inputs.
   - Advanced text preprocessing including language detection, translation, coreference resolution, and anonymization.
   - Utilizes OCR (Optical Character Recognition) for extracting text from images.

2. **Sign Extraction and Context Analysis**:
   - Extracts significant signs and contexts from the text.
   - Leverages named entity recognition (NER) using transformer models.

3. **Discourse Analysis**:
   - Analyzes the discourse structure of the text using Rhetorical Structure Theory (RST).
   - Visualizes discourse trees using pyvis for interactive exploration.

4. **Sentiment and Emotion Analysis**:
   - Performs sentiment analysis using VADER, TextBlob, and transformer-based models.
   - Provides detailed sentiment scores and explanations using LIME.

5. **Syntagmatic and Paradigmatic Relationships**:
   - Calculates term frequencies, prominence, and syntagmatic relationships.
   - Computes paradigmatic relationships using cosine similarity.

6. **Contextual Influence and Topic Modeling**:
   - Uses Word2Vec for computing contextual influence.
   - Performs topic modeling with Latent Dirichlet Allocation (LDA) and explains topics using SHAP.

7. **Data Enrichment**:
   - Enriches data with external knowledge from Wikipedia and Google Knowledge Graph.

8. **Visualization**:
   - Visualizes syntagmatic and paradigmatic matrices.
   - Generates interactive network graphs for signs.
   - Creates detailed PDF reports and exports results to CSV, JSON, and Jupyter Notebooks.

9. **Error Handling and Robustness**:
   - Comprehensive error handling and logging.
   - Secure API communication with encryption for sensitive data.

## Installation

### Prerequisites

- Python 3.7 or higher
- Necessary libraries and packages as specified in `requirements.txt`

### Install Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Additional Setup

1. **Download NLTK Data**:

Ensure NLTK data packages are downloaded:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

2. **Set Up Google Knowledge Graph API**:

If using Google Knowledge Graph, set up your Google Cloud credentials and ensure the `ENCRYPTION_KEY` environment variable is set for secure API communication.

## Usage

### Running the Script

1. Place your input files (text, images) in the root directory or specify the path.
2. Run the script:

```bash
python semiotic_analysis_tool.py
```

### Output

The script generates several outputs:
- JSON file with detailed analysis results.
- PDF report summarizing key findings.
- CSV file with analysis data.
- Jupyter Notebook with structured analysis results.
- Interactive HTML files for discourse tree visualizations.

### Streamlit Dashboard

To explore the results interactively, use the Streamlit dashboard:

```bash
streamlit run semiotic_analysis_tool.py
```

## Configuration

### Environment Variables

- `ENCRYPTION_KEY`: Encryption key for secure API communication.

### Parameters

You can customize various parameters in the script, such as:
- Number of topics for topic modeling.
- Paths for input and output directories.
- Language settings for translation.

## Documentation

### Functions

#### `load_input_data(input_folder='.'):`

Loads input data from the specified directory, supporting text and image files.

#### `detect_and_translate(text, target_language='en'):`

Detects the language of the input text and translates it to the specified target language using Google Translate API.

#### `advanced_text_preprocessing(text):`

Performs advanced text preprocessing, including coreference resolution and anonymization.

#### `extract_discourse_structure(text):`

Analyzes the discourse structure of the text and returns an RST tree.

#### `visualize_discourse_tree_pyvis(tree, filename):`

Visualizes the discourse tree using pyvis and saves it as an HTML file.

#### `get_named_entities_advanced(text):`

Extracts named entities from the text using transformer models.

#### `enrich_data_with_external_knowledge(named_entities):`

Enriches named entities with additional context from Wikipedia and Google Knowledge Graph.

#### `extract_signs_and_contexts(text):`

Extracts significant signs and contexts from the text using TF-IDF and NER.

#### `lime_explain_sentiment(text, transformer_sentiment_analyzer):`

Generates explanations for sentiment analysis results using LIME.

#### `shap_explain_topics(texts, num_topics=5):`

Generates explanations for topic modeling using SHAP.

#### `compute_term_frequencies(signs, texts):`

Calculates the frequency of each sign in the provided texts.

#### `compute_prominence(signs, texts):`

Calculates the prominence of each sign based on its positional relevance.

#### `compute_syntagmatic_matrix(signs, sentences):`

Computes the syntagmatic relationship matrix for the signs.

#### `compute_similarity(signs):`

Computes the paradigmatic relationship matrix using cosine similarity.

#### `compute_contextual_influence(signs, contexts, model):`

Calculates the contextual influence of signs using Word2Vec.

#### `perform_topic_modeling(texts, num_topics=5):`

Performs topic modeling on the input texts using LDA.

#### `analyze_sentiment(text):`

Analyzes sentiment and emotion of the text using VADER, TextBlob, and transformer models.

#### `visualize_matrices(sequence_matrix, paradigmatic_matrix, signs):`

Visualizes syntagmatic and paradigmatic matrices using heatmaps.

#### `visualize_sign_network(matrix, signs):`

Generates a network graph for the signs.

#### `generate_pdf_report(output_data, filename="report.pdf"):`

Generates a detailed PDF report summarizing the analysis results.

#### `export_to_csv(data, filename="analysis_results.csv"):`

Exports the analysis results to a CSV file.

#### `export_to_json(data, filename="analysis_results.json"):`

Exports the analysis results to a JSON file.

#### `create_jupyter_notebook(data, filename="analysis_results.ipynb"):`

Creates a Jupyter Notebook with the analysis results.

## Error Handling

The script includes comprehensive error handling to ensure robustness. Detailed error messages are logged, and the script continues to execute other parts of the analysis even if one component fails.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License.

