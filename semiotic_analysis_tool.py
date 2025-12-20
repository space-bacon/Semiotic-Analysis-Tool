import os
import json
import numpy as np
import nltk
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import pytesseract
from PIL import Image
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from sklearn.decomposition import LatentDirichletAllocation as LDA
from fpdf import FPDF
import logging
# Note: discopy v1.2.1 is a category theory library and doesn't have discourse parsing
# Discourse analysis features are disabled for compatibility
from pyvis.network import Network
import multiprocessing as mp
import pandas as pd
import nbformat as nbf
import wikipediaapi
import shap
import lime
import lime.lime_text
from cryptography.fernet import Fernet
# googletrans has compatibility issues with Python 3.13 (missing cgi module)
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logging.warning(f"Translation disabled: {e}")
    Translator = None
    TRANSLATION_AVAILABLE = False

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure nltk packages are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize sentiment analysis tools (lazy-loaded to avoid startup delay)
analyzer = None
transformer_sentiment_analyzer = None
tokenizer = None
ner_model = None

def get_sentiment_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()
    return analyzer

def get_transformer_sentiment():
    global transformer_sentiment_analyzer
    if transformer_sentiment_analyzer is None:
        transformer_sentiment_analyzer = pipeline("sentiment-analysis")
    return transformer_sentiment_analyzer

def get_ner_tools():
    global tokenizer, ner_model
    if tokenizer is None or ner_model is None:
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    return tokenizer, ner_model

# Note: Discourse parser disabled - discopy v1.2.1 API changed

# Setup encryption for secure API calls
encryption_key = os.getenv("ENCRYPTION_KEY")
if encryption_key:
    cipher_suite = Fernet(encryption_key)
else:
    cipher_suite = None

# Initialize Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Load input data from the data directory
def load_input_data(input_folder='data'):
    input_data = {}
    if not os.path.exists(input_folder):
        logging.error(f"Input folder '{input_folder}' does not exist.")
        return input_data
    
    for filename in os.listdir(input_folder):
        # Skip hidden files and directories
        if filename.startswith('.') or os.path.isdir(os.path.join(input_folder, filename)):
            continue
            
        filepath = os.path.join(input_folder, filename)
        try:
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as file:
                    input_data[filename] = file.read()
            elif filename.endswith(('.png', '.jpg', '.jpeg')):
                input_data[filename] = pytesseract.image_to_string(Image.open(filepath))
            else:
                logging.warning(f"Unsupported file type: {filename}")
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
    return input_data

# Detect and translate text to English
def detect_and_translate(text, target_language='en'):
    """
    Language detection and translation - disabled due to Python 3.13 compatibility
    """
    if not TRANSLATION_AVAILABLE:
        logging.warning("Translation unavailable - googletrans not compatible with Python 3.13")
        return text
    
    try:
        translator = Translator()
        detected_lang = translator.detect(text).lang
        if detected_lang != target_language:
            text = translator.translate(text, dest=target_language).text
        return text
    except Exception as e:
        logging.error(f"Error in language detection/translation: {e}")
        return text

# Advanced text preprocessing
def advanced_text_preprocessing(text):
    try:
        # Add any additional preprocessing steps here
        return text
    except Exception as e:
        logging.error(f"Error in advanced text preprocessing: {e}")
        return text

# Extract discourse structure from text
def extract_discourse_structure(text):
    """
    Discourse structure analysis - currently disabled due to discopy API changes
    Returns a simple sentence-based structure instead
    """
    try:
        sentences = sent_tokenize(text)
        # Return a simple structure representation
        return {
            'type': 'simple_discourse',
            'sentence_count': len(sentences),
            'sentences': sentences[:10],  # First 10 sentences
            'note': 'Full RST discourse parsing disabled - discopy v1.2.1 API incompatible'
        }
    except Exception as e:
        logging.error(f"Error during discourse analysis: {e}")
        return None

# Visualize and save discourse tree using pyvis
def visualize_discourse_tree_pyvis(tree, filename):
    """
    Discourse tree visualization - simplified for compatibility
    """
    try:
        if tree and isinstance(tree, dict) and tree.get('type') == 'simple_discourse':
            # Create a simple visualization of sentences
            net = Network(height='750px', width='100%', notebook=True, directed=True)
            net.add_node(0, label="Document", title="Root document node")
            
            for i, sentence in enumerate(tree.get('sentences', [])[:10], 1):
                net.add_node(i, label=f"S{i}", title=sentence[:100])
                net.add_edge(0, i)
            
            output_path = f'output/{filename}_discourse_tree.html'
            os.makedirs('output', exist_ok=True)
            net.show(output_path)
            logging.info(f"Discourse tree visualization saved to {output_path}")
        else:
            logging.warning("Discourse tree visualization skipped - no valid tree structure")
    except Exception as e:
        logging.error(f"Error visualizing discourse tree: {e}")
    except Exception as e:
        logging.error(f"Error visualizing discourse tree: {e}")

# Extract named entities using a transformer model
def get_named_entities_advanced(text):
    try:
        tokenizer, ner_model = get_ner_tools()
        tokens = tokenizer(text, return_tensors="pt")
        outputs = ner_model(**tokens).logits
        predictions = torch.argmax(outputs, dim=2)
        predicted_token_classes = [ner_model.config.id2label[p.item()] for p in predictions[0]]
        named_entities = []
        current_entity = []
        for token, label in zip(tokens.tokens(), predicted_token_classes):
            if label.startswith("B-") or label.startswith("I-"):
                current_entity.append(token)
            elif current_entity:
                named_entities.append(" ".join(current_entity).replace("##", ""))
                current_entity = []
        return named_entities
    except Exception as e:
        logging.error(f"Error extracting named entities: {e}")
        return []

# Enrich data with external knowledge sources
def enrich_data_with_external_knowledge(named_entities):
    """
    Enrich data with Wikipedia - limited to first 5 entities for performance
    """
    enriched_knowledge = {}
    
    # Limit to first 5 entities to avoid long waits
    entities_to_check = list(set(named_entities))[:5]
    
    if not entities_to_check:
        logging.info("No entities to enrich")
        return enriched_knowledge
    
    try:
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='SemioticAnalysisTool/1.0 (Educational Research)',
            timeout=5  # 5 second timeout per request
        )
    except Exception as e:
        logging.error(f"Error initializing Wikipedia API: {e}")
        return enriched_knowledge
    
    logging.info(f"Enriching {len(entities_to_check)} entities with Wikipedia data...")
    
    try:
        for entity in entities_to_check:
            try:
                # Use Wikipedia API for enrichment
                page = wiki_wiki.page(entity)
                if page.exists():
                    # Get only first 200 chars of summary for performance
                    summary = page.summary[:200] + "..." if len(page.summary) > 200 else page.summary
                    enriched_knowledge[entity] = {
                        "summary": summary,
                        "url": page.fullurl
                    }
                    logging.info(f"  ✓ Enriched: {entity}")
            except Exception as e:
                logging.warning(f"  ✗ Could not enrich {entity}: {str(e)[:50]}")
                continue

        return enriched_knowledge
    except Exception as e:
        logging.error(f"Error enriching data with external knowledge: {e}")
        return enriched_knowledge

# Extract signs and contexts from text
def extract_signs_and_contexts(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    signs = [word for word in words if word.isalpha() and word not in stop_words]
    named_entities = get_named_entities_advanced(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    terms = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray().flatten()
    top_indices = np.argsort(tfidf_scores)[::-1][:5]
    important_terms = [terms[index] for index in top_indices]
    contexts = list(set(named_entities + important_terms))
    return list(set(signs)), contexts

# LIME for explaining sentiment analysis results
def lime_explain_sentiment(text, transformer_sentiment_analyzer):
    """
    LIME explanation for sentiment analysis - simplified for compatibility
    """
    try:
        sentiment_pipeline = get_transformer_sentiment()
        
        # Create a wrapper function that returns probability arrays
        def predict_proba(texts):
            import numpy as np
            results = []
            for t in texts:
                # Get prediction from transformer
                pred = sentiment_pipeline(t[:512])[0]  # Limit to 512 chars for speed
                
                # Convert to probability array [negative_prob, positive_prob]
                if pred['label'] == 'POSITIVE':
                    pos_prob = pred['score']
                    neg_prob = 1 - pred['score']
                else:  # NEGATIVE
                    neg_prob = pred['score']
                    pos_prob = 1 - pred['score']
                
                results.append([neg_prob, pos_prob])
            
            return np.array(results)
        
        # Use LIME with the wrapper function
        explainer = lime.lime_text.LimeTextExplainer(class_names=["Negative", "Positive"])
        # Limit text length for performance
        exp = explainer.explain_instance(text[:1000], predict_proba, num_features=10, num_samples=100)
        return exp.as_list()
    except Exception as e:
        logging.error(f"Error generating LIME explanations for sentiment analysis: {e}")
        return []

# SHAP for explaining topic modeling results
def shap_explain_topics(texts, num_topics=5):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        lda = LDA(n_components=num_topics, random_state=0)
        lda.fit(X)

        explainer = shap.KernelExplainer(lda.transform, X)
        shap_values = explainer.shap_values(X)
        return shap_values
    except Exception as e:
        logging.error(f"Error generating SHAP explanations for topic modeling: {e}")
        return []

# Compute term frequencies
def compute_term_frequencies(signs, texts):
    all_text = ' '.join(texts).lower()
    words = word_tokenize(all_text)
    word_counts = Counter([word for word in words if word.isalpha()])
    return [word_counts[sign] for sign in signs]

# Calculate prominence based on positional relevance
def compute_prominence(signs, texts):
    prominence_scores = []
    for sign in signs:
        prominence = 0
        for text in texts:
            words = word_tokenize(text.lower())
            if sign in words[:100]: prominence += 1
            if sign in words[:100]: prominence += 2
        prominence_scores.append(prominence)
    return prominence_scores

# Compute syntagmatic relationships
def compute_syntagmatic_matrix(signs, sentences):
    sign_index = {sign: idx for idx, sign in enumerate(signs)}
    syntagmatic_matrix = np.zeros((len(signs), len(signs)))
    for sentence in sentences:
        tokens = [token for token in word_tokenize(sentence.lower()) if token in sign_index]
        for i in range(len(tokens) - 1):
            syntagmatic_matrix[sign_index[tokens[i]]][sign_index[tokens[i + 1]]] += 1
    return normalize(syntagmatic_matrix, norm='l1')

# Compute similarity matrix for paradigmatic relationships
def compute_similarity(signs):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(signs)
    return cosine_similarity(tfidf_matrix)

# Compute contextual influence using word embeddings
def compute_contextual_influence(signs, contexts, model):
    context_weights = np.ones(len(contexts))  # Initialize with equal weights
    context_relevance = np.zeros((len(signs), len(contexts)))
    for i, sign in enumerate(signs):
        for j, context in enumerate(contexts):
            if sign in model.wv and context in model.wv:
                similarity = 1 - cosine(model.wv[sign], model.wv[context])
                context_relevance[i][j] = similarity
    return np.dot(context_relevance, context_weights)

# Perform topic modeling on the context data
def perform_topic_modeling(texts, num_topics=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    lda = LDA(n_components=num_topics, random_state=0)
    lda.fit(X)
    topics = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic in topics:
        top_keywords = [feature_names[i] for i in topic.argsort()[-10:]]
        topic_keywords.append(top_keywords)
    return topic_keywords

# Analyze sentiment and emotion of a text
def analyze_sentiment(text):
    try:
        vader_analyzer = get_sentiment_analyzer()
        sentiment_pipeline = get_transformer_sentiment()
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        vader_scores = vader_analyzer.polarity_scores(text)
        transformer_sentiments = sentiment_pipeline(text)
        lime_explanation = lime_explain_sentiment(text, sentiment_pipeline)
        return {
            "textblob_polarity": polarity,
            "textblob_subjectivity": subjectivity,
            "vader_scores": vader_scores,
            "transformer_sentiments": transformer_sentiments,
            "lime_explanation": lime_explanation
        }
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return {}

# Visualize syntagmatic and paradigmatic matrices
def visualize_matrices(sequence_matrix, paradigmatic_matrix, signs):
    """
    Visualize and save matrices without showing them interactively
    """
    try:
        # Use non-interactive backend
        plt.switch_backend('Agg')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(sequence_matrix, xticklabels=signs, yticklabels=signs, cmap="Blues", cbar_kws={'label': 'Frequency'})
        plt.title('Syntagmatic Matrix')
        plt.xlabel('Signs')
        plt.ylabel('Signs')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('output/syntagmatic_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 10))
        sns.heatmap(paradigmatic_matrix, xticklabels=signs, yticklabels=signs, cmap="Greens", cbar_kws={'label': 'Similarity'})
        plt.title('Paradigmatic Matrix')
        plt.xlabel('Signs')
        plt.ylabel('Signs')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('output/paradigmatic_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"Error visualizing matrices: {e}")

# Visualize a network graph of signs
def visualize_sign_network(matrix, signs):
    """
    Visualize and save sign network without showing it interactively
    """
    try:
        plt.switch_backend('Agg')
        
        G = nx.Graph()
        for i, sign in enumerate(signs):
            for j, connection_strength in enumerate(matrix[i]):
                if i != j and connection_strength > 0.1:
                    G.add_edge(sign, signs[j], weight=connection_strength)
        
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G)
        weights = list(nx.get_edge_attributes(G, 'weight').values())
        nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', 
                font_size=10, edge_color=weights, edge_cmap=plt.cm.Blues, width=2)
        plt.title('Sign Network Graph')
        plt.tight_layout()
        plt.savefig('output/sign_network.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"Error visualizing sign network: {e}")

# Generate a detailed PDF report
def generate_pdf_report(output_data, filename="report.pdf"):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Semiotic Analysis Report", ln=True, align="C")

        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt=f"System Significance: {output_data['system_significance']}\n")

        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt="Sentiment Analysis:\n")
        for doc, sentiment in output_data["sentiment_analysis"].items():
            pdf.multi_cell(0, 10, txt=f"{doc}: {sentiment}\n")

        pdf.ln(10)
        pdf.multi_cell(0, 10, txt="Topic Modeling:\n")
        for i, topic in enumerate(output_data["topics"]):
            pdf.multi_cell(0, 10, txt=f"Topic {i + 1}: {', '.join(topic)}\n")

        pdf.ln(10)
        pdf.multi_cell(0, 10, txt="Discourse Structures:\n")
        for doc, structure in output_data["discourse_structures"].items():
            pdf.multi_cell(0, 10, txt=f"{doc}: {structure}\n")

        pdf.output(filename)
        logging.info(f"Report generated: {filename}")
    except Exception as e:
        logging.error(f"Error generating PDF report: {e}")

# Export results to CSV
def export_to_csv(data, filename="analysis_results.csv"):
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"Data exported to {filename}")
    except Exception as e:
        logging.error(f"Error exporting data to CSV: {e}")

# Export results to JSON
def export_to_json(data, filename="analysis_results.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Data exported to {filename}")
    except Exception as e:
        logging.error(f"Error exporting data to JSON: {e}")

# Create a Jupyter notebook with analysis results
def create_jupyter_notebook(data, filename="analysis_results.ipynb"):
    try:
        nb = nbf.v4.new_notebook()
        cells = []
        
        # Title cell
        cells.append(nbf.v4.new_markdown_cell(
            "# Semiotic Analysis Results\n\n"
            f"**Analysis Date:** {data.get('overview', {}).get('analysis_metadata', {}).get('timestamp', 'N/A')}\n\n"
            "This notebook contains the complete analysis results with interactive code cells."
        ))
        
        # Import cell
        cells.append(nbf.v4.new_code_cell(
            "# Import required libraries\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from IPython.display import display, HTML\n\n"
            "# Set display options\n"
            "pd.set_option('display.max_rows', 100)\n"
            "sns.set_style('whitegrid')"
        ))
        
        # Overview section
        if 'overview' in data:
            ov = data['overview']
            cells.append(nbf.v4.new_markdown_cell("## 📊 Analysis Overview"))
            
            overview_code = "# Display overview\n"
            overview_code += f"overview = {json.dumps(ov, indent=2)}\n"
            overview_code += "print('Analysis Metadata:')\n"
            overview_code += "for key, value in overview.get('analysis_metadata', {}).items():\n"
            overview_code += "    print(f'  {key}: {value}')\n"
            cells.append(nbf.v4.new_code_cell(overview_code))
        
        # Corpus Statistics
        cells.append(nbf.v4.new_markdown_cell("## 📈 Corpus Statistics"))
        stats_code = "# Corpus statistics\n"
        if 'overview' in data and 'corpus_statistics' in data['overview']:
            stats = data['overview']['corpus_statistics']
            stats_code += f"stats = {json.dumps(stats, indent=2)}\n"
            stats_code += "df_stats = pd.DataFrame([stats]).T\n"
            stats_code += "df_stats.columns = ['Value']\n"
            stats_code += "display(df_stats)\n"
        cells.append(nbf.v4.new_code_cell(stats_code))
        
        # Top Signs
        cells.append(nbf.v4.new_markdown_cell("## 🏆 Top Significant Signs"))
        if 'overview' in data and 'top_signs' in data['overview']:
            top_signs = data['overview']['top_signs']
            signs_code = "# Top signs visualization\n"
            signs_code += f"top_signs = {json.dumps(top_signs, indent=2)}\n"
            signs_code += "df_signs = pd.DataFrame(top_signs)\n"
            signs_code += "display(df_signs)\n\n"
            signs_code += "# Plot top signs\n"
            signs_code += "plt.figure(figsize=(10, 6))\n"
            signs_code += "plt.barh(df_signs['sign'], df_signs['significance_score'])\n"
            signs_code += "plt.xlabel('Significance Score')\n"
            signs_code += "plt.title('Top 10 Most Significant Signs')\n"
            signs_code += "plt.gca().invert_yaxis()\n"
            signs_code += "plt.tight_layout()\n"
            signs_code += "plt.show()\n"
            cells.append(nbf.v4.new_code_cell(signs_code))
        
        # Sentiment Analysis
        if 'sentiment_analysis' in data:
            cells.append(nbf.v4.new_markdown_cell("## 💭 Sentiment Analysis"))
            sent_code = "# Sentiment analysis results\n"
            sent_code += f"sentiment = {json.dumps(data['sentiment_analysis'], indent=2)}\n"
            sent_code += "print(json.dumps(sentiment, indent=2))\n"
            cells.append(nbf.v4.new_code_cell(sent_code))
        
        # Topics
        if 'topics' in data:
            cells.append(nbf.v4.new_markdown_cell("## 📚 Topic Modeling"))
            topics_code = "# Topics identified\n"
            topics_code += f"topics = {json.dumps(data['topics'], indent=2)}\n"
            topics_code += "for i, topic in enumerate(topics, 1):\n"
            topics_code += "    print(f'Topic {i}: {topic}')\n"
            cells.append(nbf.v4.new_code_cell(topics_code))
        
        # Load matrices section
        cells.append(nbf.v4.new_markdown_cell(
            "## 🔢 Matrices\n\n"
            "The full matrices are stored in `matrices.npz` to keep this notebook manageable.\n"
            "Run the cell below to load them."
        ))
        
        matrix_code = "# Load matrices from compressed file\n"
        matrix_code += "matrices = np.load('matrices.npz')\n"
        matrix_code += "syntagmatic_matrix = matrices['syntagmatic_matrix']\n"
        matrix_code += "paradigmatic_matrix = matrices['paradigmatic_matrix']\n"
        matrix_code += "contextual_influence = matrices['contextual_influence']\n\n"
        matrix_code += "print(f'Syntagmatic matrix shape: {syntagmatic_matrix.shape}')\n"
        matrix_code += "print(f'Paradigmatic matrix shape: {paradigmatic_matrix.shape}')\n"
        matrix_code += "print(f'Contextual influence shape: {contextual_influence.shape}')\n"
        cells.append(nbf.v4.new_code_cell(matrix_code))
        
        # Matrix summaries
        if 'syntagmatic_matrix_summary' in data:
            cells.append(nbf.v4.new_markdown_cell("### Matrix Statistics"))
            summary_code = "# Matrix summaries\n"
            summary_code += f"syn_summary = {json.dumps(data['syntagmatic_matrix_summary'], indent=2)}\n"
            summary_code += f"para_summary = {json.dumps(data['paradigmatic_matrix_summary'], indent=2)}\n"
            summary_code += "print('Syntagmatic Matrix:')\n"
            summary_code += "for k, v in syn_summary.items():\n"
            summary_code += "    print(f'  {k}: {v}')\n"
            summary_code += "print('\\nParadigmatic Matrix:')\n"
            summary_code += "for k, v in para_summary.items():\n"
            summary_code += "    print(f'  {k}: {v}')\n"
            cells.append(nbf.v4.new_code_cell(summary_code))
        
        # Instructions for further analysis
        cells.append(nbf.v4.new_markdown_cell(
            "## 🔬 Further Analysis\n\n"
            "You can now perform additional analysis on the loaded data:\n\n"
            "```python\n"
            "# Example: Visualize a subset of the syntagmatic matrix\n"
            "import seaborn as sns\n"
            "plt.figure(figsize=(12, 10))\n"
            "sns.heatmap(syntagmatic_matrix[:50, :50], cmap='coolwarm')\n"
            "plt.title('Syntagmatic Matrix (first 50x50 signs)')\n"
            "plt.show()\n"
            "```"
        ))
        
        nb.cells = cells
        with open(filename, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        logging.info(f"Jupyter notebook created: {filename}")
    except Exception as e:
        logging.error(f"Error creating Jupyter notebook: {e}")

# Process sentiment analysis in parallel
def parallel_sentiment_analysis(text):
    return analyze_sentiment(text)

def process_sentiments_parallel(texts):
    try:
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(parallel_sentiment_analysis, texts)
        return results
    except Exception as e:
        logging.error(f"Error in parallel sentiment analysis: {e}")
        return []

# Validate signs are unique and not empty
def validate_signs(signs):
    if not signs:
        raise ValueError("No signs extracted from the text.")
    return list(set(signs))

# Ensure the output directory exists
def ensure_output_directory(directory='output'):
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating output directory: {e}")

# Generate comprehensive analysis overview
def generate_overview(output, input_files):
    """Generate a comprehensive overview of the analysis results"""
    try:
        overview = {
            "analysis_metadata": {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "files_analyzed": list(input_files.keys()),
                "total_files": len(input_files),
                "total_characters": sum(len(text) for text in input_files.values())
            },
            "corpus_statistics": {
                "total_signs": len(output["signs"]),
                "total_contexts": len(output["contexts"]),
                "unique_signs": len(set(output["signs"])),
                "system_significance": float(output["system_significance"])
            },
            "sentiment_overview": {},
            "topic_overview": {
                "num_topics": len(output.get("topics", [])),
                "topics": output.get("topics", [])
            },
            "top_signs": [],
            "key_insights": []
        }
        
        # Calculate sentiment statistics
        if output.get("sentiment_analysis"):
            sentiments = output["sentiment_analysis"]
            if isinstance(sentiments, dict):
                sentiments = [sentiments]
            
            polarities = [s.get("textblob_polarity", 0) for s in sentiments]
            vader_compound = [s.get("vader_scores", {}).get("compound", 0) for s in sentiments]
            
            overview["sentiment_overview"] = {
                "average_polarity": float(np.mean(polarities)) if polarities else 0,
                "average_vader_compound": float(np.mean(vader_compound)) if vader_compound else 0,
                "sentiment_range": {
                    "min_polarity": float(min(polarities)) if polarities else 0,
                    "max_polarity": float(max(polarities)) if polarities else 0
                }
            }
        
        # Get top signs by significance
        if len(output["total_significance_scores"]) > 0:
            top_indices = np.argsort(output["total_significance_scores"])[-10:][::-1]
            overview["top_signs"] = [
                {
                    "sign": output["signs"][i] if i < len(output["signs"]) else "unknown",
                    "significance_score": float(output["total_significance_scores"][i])
                }
                for i in top_indices if i < len(output["signs"])
            ]
        
        # Generate key insights
        insights = []
        
        # Insight 1: Corpus size
        total_chars = overview["analysis_metadata"]["total_characters"]
        insights.append(f"Analyzed {total_chars:,} characters across {len(input_files)} file(s)")
        
        # Insight 2: Sign diversity
        unique_ratio = overview["corpus_statistics"]["unique_signs"] / max(overview["corpus_statistics"]["total_signs"], 1)
        insights.append(f"Sign diversity: {unique_ratio:.2%} ({overview['corpus_statistics']['unique_signs']} unique signs from {overview['corpus_statistics']['total_signs']} total)")
        
        # Insight 3: Sentiment
        if overview["sentiment_overview"]:
            avg_pol = overview["sentiment_overview"]["average_polarity"]
            sentiment_label = "positive" if avg_pol > 0.1 else "negative" if avg_pol < -0.1 else "neutral"
            insights.append(f"Overall sentiment: {sentiment_label} (polarity: {avg_pol:.3f})")
        
        # Insight 4: Top sign
        if overview["top_signs"]:
            top_sign = overview["top_signs"][0]
            insights.append(f"Most significant sign: '{top_sign['sign']}' (score: {top_sign['significance_score']:.3f})")
        
        # Insight 5: System significance
        insights.append(f"System significance score: {overview['corpus_statistics']['system_significance']:.2f}")
        
        overview["key_insights"] = insights
        
        return overview
        
    except Exception as e:
        logging.error(f"Error generating overview: {e}")
        return {
            "error": str(e),
            "analysis_metadata": {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "partial_failure"
            }
        }

# Main script logic
input_data = load_input_data('data')
if not input_data:
    logging.error("No input data found. Please add .txt or image files to the 'data/' directory.")
    exit(1)

print(f"\n{'='*60}")
print(f"🔍 SEMIOTIC ANALYSIS TOOL - Starting Analysis")
print(f"{'='*60}")
print(f"📁 Found {len(input_data)} file(s) to analyze")
for filename in input_data.keys():
    print(f"   • {filename}")
print(f"{'='*60}\n")

text_list = list(input_data.values())

# Ensure the output directory exists
ensure_output_directory('output')

# Initialize data structures
all_signs = []
all_contexts = set()
discourse_structures = {}

# Process each input file and handle possible errors
print("📝 Phase 1: Text Processing & Extraction")
print("-" * 60)
for i, (filename, content) in enumerate(input_data.items(), 1):
    print(f"[{i}/{len(input_data)}] Processing: {filename}")
    try:
        # Translate text to English if needed
        print(f"   → Detecting language & translating...")
        translated_content = detect_and_translate(content)
        
        # Advanced text preprocessing
        print(f"   → Preprocessing text...")
        preprocessed_content = advanced_text_preprocessing(translated_content)
        
        print(f"   → Extracting signs and contexts...")
        signs, contexts = extract_signs_and_contexts(preprocessed_content)
        all_signs.extend(signs)
        all_contexts.update(contexts)
        print(f"   ✓ Extracted {len(signs)} signs, {len(contexts)} contexts")

        print(f"   → Analyzing discourse structure...")
        discourse_structure = extract_discourse_structure(preprocessed_content)
        if discourse_structure:
            discourse_structures[filename] = discourse_structure
            visualize_discourse_tree_pyvis(discourse_structure, filename)
            print(f"   ✓ Discourse visualization saved")
    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")
        print(f"   ✗ Error: {e}")

print(f"\n{'='*60}")
print(f"✓ Phase 1 Complete: Extracted {len(all_signs)} total signs")
print(f"{'='*60}\n")

# Ensure unique signs and contexts
print("🔧 Phase 2: Validating & Preparing Data")
print("-" * 60)
print("   → Validating signs...")
all_signs = validate_signs(all_signs)
all_contexts = list(all_contexts)
print(f"   ✓ Validated {len(all_signs)} unique signs")
print(f"   ✓ Prepared {len(all_contexts)} contexts\n")

# Train or load Word2Vec model safely
print("🧠 Phase 3: Training Word2Vec Model")
print("-" * 60)
try:
    print("   → Tokenizing sentences...")
    sentences = [word_tokenize(text.lower()) for text in text_list]
    print(f"   → Training Word2Vec on {len(sentences)} sentences...")
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    print(f"   ✓ Word2Vec model trained successfully\n")
except Exception as e:
    logging.error(f"Error training Word2Vec model: {e}")
    print(f"   ✗ Error training model: {e}\n")

# Calculate Term Frequencies and Prominence
print("📊 Phase 4: Calculating Significance Metrics")
print("-" * 60)
print("   → Computing term frequencies...")
frequency = compute_term_frequencies(all_signs, text_list)
print("   → Computing prominence scores...")
prominence = compute_prominence(all_signs, text_list)
print("   ✓ Metrics calculated\n")

# Calculate Importance (I)
print("📈 Phase 5: Computing Importance Scores")
print("-" * 60)
importance = []
alpha, beta, gamma = 0.5, 0.3, 0.2
print(f"   → Calculating importance for {len(all_signs)} signs...")
for i in range(len(all_signs)):
    relevance = frequency[i] * prominence[i]
    importance.append(alpha * frequency[i] + beta * prominence[i] + gamma * relevance)
print(f"   ✓ Importance scores computed\n")

# Calculate Syntagmatic Relationships
print("🔗 Phase 6: Computing Relationships")
print("-" * 60)
print("   → Computing syntagmatic relationships...")
sentences_flat = [sent_tokenize(text.lower()) for text in text_list]
flat_sentences = [sentence for sublist in sentences_flat for sentence in sublist]
sequence_matrix = compute_syntagmatic_matrix(all_signs, flat_sentences)
print(f"   ✓ Syntagmatic matrix: {sequence_matrix.shape}")

# Calculate Paradigmatic Relationships
print("   → Computing paradigmatic relationships...")
paradigmatic_matrix = compute_similarity(all_signs)
print(f"   ✓ Paradigmatic matrix: {paradigmatic_matrix.shape}")

# Calculate Contextual Influence (C)
print("   → Computing contextual influence...")
contextual_influence = compute_contextual_influence(all_signs, all_contexts, model)
print(f"   ✓ Contextual influence computed\n")

# Calculate Total Significance Score (TS)
print("🎯 Phase 7: Computing Total Significance")
print("-" * 60)
print("   → Calculating total significance scores...")
total_significance_scores = importance + contextual_influence

# Aggregate System Significance (TS)
system_significance = np.sum(total_significance_scores)
print(f"   ✓ System significance: {system_significance:.2f}\n")

# Analyze sentiment and emotion for each document
print("😊 Phase 8: Sentiment Analysis")
print("-" * 60)
print(f"   → Analyzing sentiment for {len(input_data)} document(s)...")
sentiment_analysis = {filename: analyze_sentiment(content) for filename, content in input_data.items()}
print(f"   ✓ Sentiment analysis complete\n")

# Perform topic modeling on the contexts
print("📚 Phase 9: Topic Modeling")
print("-" * 60)
print("   → Extracting topics from text...")
topics = perform_topic_modeling(text_list)
print(f"   ✓ Topics identified\n")

# Enrich with external knowledge
print("🌐 Phase 10: External Knowledge Enrichment")
print("-" * 60)
external_knowledge = enrich_data_with_external_knowledge(all_contexts)
print(f"   ✓ Enriched {len(external_knowledge)} entities\n")

# Output the results
print("💾 Phase 11: Generating Outputs")
print("-" * 60)
print("   → Preparing output data...")

# Save large matrices separately in compressed format
print("   → Saving matrices separately (compressed)...")
np.savez_compressed('output/matrices.npz',
                    syntagmatic_matrix=sequence_matrix,
                    paradigmatic_matrix=paradigmatic_matrix,
                    contextual_influence=contextual_influence)
print(f"   ✓ Matrices saved to matrices.npz")

# Create matrix summaries for JSON (instead of full matrices)
def get_matrix_summary(matrix, name):
    """Create a summary of a large matrix instead of storing all values"""
    return {
        "shape": matrix.shape,
        "dtype": str(matrix.dtype),
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
        "mean": float(np.mean(matrix)),
        "std": float(np.std(matrix)),
        "stored_in": "output/matrices.npz",
        "note": f"Full {name} matrix saved separately to save space"
    }

output = {
    "signs": all_signs,
    "contexts": all_contexts,
    "importance": importance.tolist() if hasattr(importance, 'tolist') else importance,
    "syntagmatic_matrix_summary": get_matrix_summary(sequence_matrix, "syntagmatic"),
    "paradigmatic_matrix_summary": get_matrix_summary(paradigmatic_matrix, "paradigmatic"),
    "contextual_influence": contextual_influence.tolist(),
    "total_significance_scores": total_significance_scores.tolist(),
    "system_significance": system_significance,
    "sentiment_analysis": sentiment_analysis,
    "topics": topics,
    "discourse_structures": {doc: str(struct) for doc, struct in discourse_structures.items()},
    "external_knowledge": external_knowledge
}

# Generate comprehensive overview
print("   → Generating analysis overview...")
overview = generate_overview(output, input_data)
output["overview"] = overview
print(f"   ✓ Overview generated\n")

# Save the output to a JSON file
output_file = 'output/semiotic_analysis_output.json'
print(f"   → Saving JSON output to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
print(f"   ✓ JSON saved\n")

logging.info(f"Analysis complete. Results saved to {output_file}.")

# Generate a detailed PDF report
print("📄 Phase 12: Generating PDF Report")
print("-" * 60)
print("   → Creating PDF report...")
generate_pdf_report(output)
print("   ✓ PDF report generated\n")

# Visualize matrices
print("📊 Phase 13: Creating Visualizations")
print("-" * 60)
print("   → Visualizing matrices...")
visualize_matrices(sequence_matrix, paradigmatic_matrix, all_signs)
print("   ✓ Matrix visualizations saved")

# Visualize sign network
print("   → Creating sign network graph...")
visualize_sign_network(paradigmatic_matrix, all_signs)
print("   ✓ Network graph saved\n")

# Export results to CSV, JSON, and Jupyter Notebook
print("💾 Phase 14: Exporting Additional Formats")
print("-" * 60)
print("   → Exporting to CSV...")
export_to_csv(output, filename="output/output_analysis.csv")
print("   ✓ CSV exported")

print("   → Exporting to JSON...")
export_to_json(output, filename="output/output_analysis.json")
print("   ✓ JSON exported")

print("   → Creating Jupyter Notebook...")
create_jupyter_notebook(output, filename="output/output_analysis.ipynb")
print("   ✓ Jupyter Notebook created\n")

print(f"{'='*60}")
print(f"🎉 ANALYSIS COMPLETE!")
print(f"{'='*60}")

# Display overview
if "overview" in output:
    ov = output["overview"]
    print(f"\n📊 ANALYSIS OVERVIEW")
    print(f"{'='*60}")
    
    # Metadata
    if "analysis_metadata" in ov:
        meta = ov["analysis_metadata"]
        print(f"📅 Timestamp: {meta.get('timestamp', 'N/A')}")
        print(f"📁 Files Analyzed: {meta.get('total_files', 0)}")
        for fname in meta.get('files_analyzed', []):
            print(f"   • {fname}")
        print(f"📝 Total Characters: {meta.get('total_characters', 0):,}")
    
    # Corpus Statistics
    if "corpus_statistics" in ov:
        stats = ov["corpus_statistics"]
        print(f"\n📈 Corpus Statistics:")
        print(f"   • Total Signs: {stats.get('total_signs', 0):,}")
        print(f"   • Unique Signs: {stats.get('unique_signs', 0):,}")
        print(f"   • Contexts: {stats.get('total_contexts', 0):,}")
        print(f"   • System Significance: {stats.get('system_significance', 0):.2f}")
    
    # Sentiment Overview
    if "sentiment_overview" in ov and ov["sentiment_overview"]:
        sent = ov["sentiment_overview"]
        print(f"\n💭 Sentiment Analysis:")
        print(f"   • Average Polarity: {sent.get('average_polarity', 0):.3f}")
        print(f"   • VADER Compound: {sent.get('average_vader_compound', 0):.3f}")
        if "sentiment_range" in sent:
            sr = sent["sentiment_range"]
            print(f"   • Range: [{sr.get('min_polarity', 0):.3f}, {sr.get('max_polarity', 0):.3f}]")
    
    # Top Signs
    if "top_signs" in ov and ov["top_signs"]:
        print(f"\n🏆 Top 10 Most Significant Signs:")
        for i, sign_info in enumerate(ov["top_signs"][:10], 1):
            print(f"   {i:2d}. '{sign_info['sign']}' (score: {sign_info['significance_score']:.3f})")
    
    # Key Insights
    if "key_insights" in ov and ov["key_insights"]:
        print(f"\n💡 Key Insights:")
        for insight in ov["key_insights"]:
            print(f"   • {insight}")
    
    print(f"\n{'='*60}")

print(f"\n📁 Output files saved in: ./output/")
print(f"   • semiotic_analysis_output.json (compact - matrices separate)")
print(f"   • matrices.npz (compressed matrices)")
print(f"   • output_analysis.json")
print(f"   • output_analysis.csv")
print(f"   • output_analysis.ipynb (interactive notebook)")
print(f"   • PDF report")
print(f"   • Visualizations (PNG/HTML)")
print(f"{'='*60}\n")
