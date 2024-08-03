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
from discopy.parsers import SimpleRSTParser
from discopy.core import Document as DiscourseDocument
from discopy.representations import build_tree
from pyvis.network import Network
import multiprocessing as mp
import pandas as pd
import nbformat as nbf
import wikipediaapi
import shap
import lime
import lime.lime_text
from cryptography.fernet import Fernet
from googletrans import Translator

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure nltk packages are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize sentiment analysis tools
analyzer = SentimentIntensityAnalyzer()
transformer_sentiment_analyzer = pipeline("sentiment-analysis")

# Initialize NER tools
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize discourse parser
discourse_parser = SimpleRSTParser()

# Setup encryption for secure API calls
encryption_key = os.getenv("ENCRYPTION_KEY")
cipher_suite = Fernet(encryption_key)

# Initialize Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Load input data from the root directory
def load_input_data(input_folder='.'):
    input_data = {}
    for filename in os.listdir(input_folder):
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
    try:
        discourse_doc = DiscourseDocument(text)
        rst_tree = discourse_parser.parse(discourse_doc)
        return build_tree(rst_tree)
    except Exception as e:
        logging.error(f"Error during discourse parsing: {e}")
        return None

# Visualize and save discourse tree using pyvis
def visualize_discourse_tree_pyvis(tree, filename):
    try:
        net = Network(height='750px', width='100%', notebook=True, directed=True)
        for node in tree.nodes:
            label = f"{node.label}\n{node.data.get('text', '')}"
            title = f"Node: {node.label}\nText: {node.data.get('text', '')}"
            net.add_node(node.identifier, label=label, title=title)
        for edge in tree.edges:
            net.add_edge(edge.source, edge.target)
        output_path = f'output/{filename}_discourse_tree.html'
        net.show(output_path)
        logging.info(f"Discourse tree visualization saved to {output_path}")
    except Exception as e:
        logging.error(f"Error visualizing discourse tree: {e}")

# Extract named entities using a transformer model
def get_named_entities_advanced(text):
    try:
        tokens = tokenizer(text, return_tensors="pt")
        outputs = model(**tokens).logits
        predictions = torch.argmax(outputs, dim=2)
        predicted_token_classes = [model.config.id2label[p.item()] for p in predictions[0]]
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
    enriched_knowledge = {}
    wiki_wiki = wikipediaapi.Wikipedia('en')
    try:
        for entity in named_entities:
            try:
                # Use Wikipedia API for enrichment
                page = wiki_wiki.page(entity)
                if page.exists():
                    enriched_knowledge[entity] = {
                        "summary": page.summary,
                        "url": page.fullurl
                    }
            except Exception as e:
                logging.error(f"Error fetching Wikipedia data for {entity}: {e}")

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
    try:
        explainer = lime.lime_text.LimeTextExplainer(class_names=["Negative", "Positive"])
        exp = explainer.explain_instance(text, lambda x: transformer_sentiment_analyzer(x), num_features=10)
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
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        vader_scores = analyzer.polarity_scores(text)
        transformer_sentiments = transformer_sentiment_analyzer(text)
        lime_explanation = lime_explain_sentiment(text, transformer_sentiment_analyzer)
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
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(sequence_matrix, xticklabels=signs, yticklabels=signs, cmap="Blues", cbar_kws={'label': 'Frequency'})
        plt.title('Syntagmatic Matrix')
        plt.xlabel('Signs')
        plt.ylabel('Signs')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()

        plt.figure(figsize=(12, 10))
        sns.heatmap(paradigmatic_matrix, xticklabels=signs, yticklabels=signs, cmap="Greens", cbar_kws={'label': 'Similarity'})
        plt.title('Paradigmatic Matrix')
        plt.xlabel('Signs')
        plt.ylabel('Signs')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing matrices: {e}")

# Visualize a network graph of signs
def visualize_sign_network(matrix, signs):
    try:
        G = nx.Graph()
        for i, sign in enumerate(signs):
            for j, connection_strength in enumerate(matrix[i]):
                if i != j and connection_strength > 0.1:
                    G.add_edge(sign, signs[j], weight=connection_strength)
        pos = nx.spring_layout(G)
        weights = nx.get_edge_attributes(G, 'weight').values()
        nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', font_size=10, edge_color=weights, edge_cmap=plt.cm.Blues, width=2)
        plt.title('Sign Network Graph')
        plt.show()
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
        text_cells = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                content = json.dumps(value, indent=4)
            else:
                content = str(value)
            text_cells.append(nbf.v4.new_markdown_cell(f"## {key}\n```json\n{content}\n```"))
        nb.cells = text_cells
        with open(filename, 'w') as f:
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

# Main script logic
input_data = load_input_data()
text_list = list(input_data.values())

# Ensure the output directory exists
ensure_output_directory('output')

# Initialize data structures
all_signs = []
all_contexts = set()
discourse_structures = {}

# Process each input file and handle possible errors
for filename, content in input_data.items():
    try:
        # Translate text to English if needed
        translated_content = detect_and_translate(content)
        
        # Advanced text preprocessing
        preprocessed_content = advanced_text_preprocessing(translated_content)
        
        signs, contexts = extract_signs_and_contexts(preprocessed_content)
        all_signs.extend(signs)
        all_contexts.update(contexts)

        discourse_structure = extract_discourse_structure(preprocessed_content)
        if discourse_structure:
            discourse_structures[filename] = discourse_structure
            visualize_discourse_tree_pyvis(discourse_structure, filename)
    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")

# Ensure unique signs and contexts
all_signs = validate_signs(all_signs)
all_contexts = list(all_contexts)

# Train or load Word2Vec model safely
try:
    sentences = [word_tokenize(text.lower()) for text in text_list]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
except Exception as e:
    logging.error(f"Error training Word2Vec model: {e}")

# Calculate Term Frequencies and Prominence
frequency = compute_term_frequencies(all_signs, text_list)
prominence = compute_prominence(all_signs, text_list)

# Calculate Importance (I)
importance = []
alpha, beta, gamma = 0.5, 0.3, 0.2
for i in range(len(all_signs)):
    relevance = frequency[i] * prominence[i]
    importance.append(alpha * frequency[i] + beta * prominence[i] + gamma * relevance)

# Calculate Syntagmatic Relationships
sentences_flat = [sent_tokenize(text.lower()) for text in text_list]
flat_sentences = [sentence for sublist in sentences_flat for sentence in sublist]
sequence_matrix = compute_syntagmatic_matrix(all_signs, flat_sentences)

# Calculate Paradigmatic Relationships
paradigmatic_matrix = compute_similarity(all_signs)

# Calculate Contextual Influence (C)
contextual_influence = compute_contextual_influence(all_signs, all_contexts, model)

# Calculate Total Significance Score (TS)
total_significance_scores = importance + contextual_influence

# Aggregate System Significance (TS)
system_significance = np.sum(total_significance_scores)

# Analyze sentiment and emotion for each document
sentiment_analysis = {filename: analyze_sentiment(content) for filename, content in input_data.items()}

# Perform topic modeling on the contexts
topics = perform_topic_modeling(text_list)

# Output the results
output = {
    "signs": all_signs,
    "contexts": all_contexts,
    "importance": importance,
    "syntagmatic_matrix": sequence_matrix.tolist(),
    "paradigmatic_matrix": paradigmatic_matrix.tolist(),
    "contextual_influence": contextual_influence.tolist(),
    "total_significance_scores": total_significance_scores.tolist(),
    "system_significance": system_significance,
    "sentiment_analysis": sentiment_analysis,
    "topics": topics,
    "discourse_structures": {doc: str(struct) for doc, struct in discourse_structures.items()},
    "external_knowledge": enrich_data_with_external_knowledge(all_contexts)
}

# Save the output to a JSON file
output_file = 'output/semiotic_analysis_output.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

logging.info(f"Analysis complete. Results saved to {output_file}.")

# Generate a detailed PDF report
generate_pdf_report(output)

# Visualize matrices
visualize_matrices(sequence_matrix, paradigmatic_matrix, all_signs)

# Visualize sign network
visualize_sign_network(paradigmatic_matrix, all_signs)

# Export results to CSV, JSON, and Jupyter Notebook
export_to_csv(output, filename="output/output_analysis.csv")
export_to_json(output, filename="output/output_analysis.json")
create_jupyter_notebook(output, filename="output/output_analysis.ipynb")
