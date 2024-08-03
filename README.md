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

### Description of Packages Used in the Script

1. **os**: Provides a way of using operating system-dependent functionality like reading or writing to the file system, accessing environment variables, and more.

2. **json**: Used to parse JSON formatted data and convert it to Python objects, and vice versa.

3. **numpy**: A fundamental package for numerical computing in Python, providing support for arrays, matrices, and mathematical functions.

4. **nltk**: The Natural Language Toolkit, a library used for working with human language data, including tasks like tokenization, stemming, and tagging.

5. **requests**: A simple HTTP library for Python, used to make requests to web services.

6. **sklearn.metrics.pairwise**: Part of scikit-learn, this module includes tools for pairwise metrics and computation, such as calculating cosine similarity.

7. **sklearn.feature_extraction.text**: Provides methods for extracting features from text, such as the TfidfVectorizer used for converting text data into numerical features.

8. **sklearn.preprocessing**: Includes utility functions and transformer classes to convert raw feature vectors into a more suitable form for downstream estimators.

9. **nltk.tokenize**: Provides functions for tokenizing text into words or sentences.

10. **nltk.corpus**: A module for accessing a variety of linguistic resources, such as corpora and lexical resources.

11. **collections.Counter**: A container in Python's collections module for counting hashable objects, useful for counting word frequencies.

12. **gensim.models.Word2Vec**: A model for generating word embeddings, a method for computing vector representations of words.

13. **scipy.spatial.distance.cosine**: Computes the cosine distance between two vectors, often used to measure similarity.

14. **pytesseract**: A Python wrapper for Google's Tesseract-OCR Engine, used to extract text from images.

15. **PIL (Pillow)**: A fork of the Python Imaging Library (PIL) that adds image processing capabilities.

16. **textblob**: A library for processing textual data, providing a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, and more.

17. **vaderSentiment.vaderSentiment**: A library that implements the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool.

18. **networkx**: A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

19. **matplotlib.pyplot**: A state-based interface to matplotlib, a comprehensive library for creating static, animated, and interactive visualizations in Python.

20. **seaborn**: A Python visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

21. **transformers**: A library by Hugging Face that provides general-purpose architectures for natural language understanding and generation with pretrained models.

22. **torch**: The core library for PyTorch, a deep learning framework that provides tensors and dynamic neural networks in Python with strong GPU acceleration.

23. **sklearn.decomposition.LatentDirichletAllocation**: A module for performing Latent Dirichlet Allocation (LDA), a common method for topic modeling.

24. **fpdf**: A library for generating PDF documents in Python, offering an API for creating, formatting, and outputting PDFs.

25. **logging**: A built-in module for tracking events that happen when software runs, useful for debugging and understanding application behavior.

26. **discopy.parsers.SimpleRSTParser**: A parser from the discopy library for handling Rhetorical Structure Theory (RST) parsing, which involves analyzing the discourse structure of texts.

27. **discopy.core.Document**: Represents a document for discourse parsing and analysis.

28. **discopy.representations.build_tree**: A function for building and visualizing discourse trees.

29. **pyvis.network**: A library for visualizing networks using the Vis.js library. It allows for interactive network visualizations in Python.

30. **multiprocessing**: A package that supports spawning processes using an API similar to the threading module, useful for parallel processing.

31. **pandas**: A powerful data analysis and manipulation library for Python, providing data structures like DataFrame.

32. **nbformat**: A library for reading and writing Jupyter Notebook files.

33. **wikipediaapi**: A Python wrapper for the Wikipedia API, used to fetch data from Wikipedia.

34. **shap**: A library for interpreting predictions made by machine learning models, providing explanations for individual predictions using Shapley values.

35. **lime**: A library for explaining the predictions of any machine learning classifier, especially useful for explaining the predictions of black-box models.

36. **lime.lime_text**: A module within LIME specifically for explaining predictions of text classifiers.

37. **cryptography.fernet**: A symmetric encryption method included in the cryptography library, used to encrypt and decrypt data securely.

38. **googletrans**: A Python wrapper for Google Translate, used for translating text between languages.

These packages together support the various functionalities of the script, including text analysis, sentiment detection, data visualization, natural language processing, and the integration of external knowledge sources.

