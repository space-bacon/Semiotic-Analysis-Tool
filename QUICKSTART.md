# Quick Start Guide

## ✅ Setup Complete!

Your Semiotic Analysis Tool is fully configured and tested.

## Running the Tool

### Basic Usage

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Place your input files in the project directory:**
   - Text files: `.txt` format
   - Images: `.png`, `.jpg`, `.jpeg` format (will use OCR)

3. **Run the analysis:**
   ```bash
   python semiotic_analysis_tool.py
   ```

### Example Test Run

I've included an `example_input.txt` file about semiotics. Try analyzing it:

```bash
source venv/bin/activate
python semiotic_analysis_tool.py
```

The tool will process all `.txt` and image files in the current directory.

## Expected Outputs

After running the analysis, you'll get:

- **JSON file**: Detailed analysis results
- **PDF report**: Summary of key findings
- **CSV file**: Tabular analysis data
- **Jupyter Notebook**: Interactive analysis results
- **HTML files**: Interactive discourse tree visualizations

## Verification Test

Run the verification script anytime to check your setup:

```bash
source venv/bin/activate
python test_setup.py
```

## Optional: Google Knowledge Graph Integration

To enable external knowledge enrichment:

1. **Generate an encryption key:**
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

2. **Get a Google API key:**
   - Visit: https://console.cloud.google.com/apis/credentials
   - Enable the Knowledge Graph API
   - Create an API key

3. **Set environment variables:**
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   export ENCRYPTION_KEY="your_encryption_key_here"
   ```

   Or create a `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   source .env
   ```

## Features Available

✅ **Text Preprocessing**
- Language detection & translation
- Tokenization & stopword removal
- Coreference resolution

✅ **Sign Extraction**
- Named Entity Recognition (NER)
- Context analysis

✅ **Discourse Analysis**
- Rhetorical Structure Theory (RST)
- Interactive discourse trees

✅ **Sentiment Analysis**
- VADER sentiment scoring
- Transformer-based analysis
- LIME explainability

✅ **Relationship Analysis**
- Syntagmatic relationships
- Paradigmatic relationships
- Term frequency & prominence

✅ **Topic Modeling**
- LDA (Latent Dirichlet Allocation)
- SHAP explanations

✅ **Visualization**
- Network graphs
- Matrix visualizations
- PDF reports

## Troubleshooting

### If you see import errors:
```bash
source venv/bin/activate  # Make sure venv is activated
```

### If you see NLTK data errors:
```bash
python test_setup.py  # This will show what's missing
```

### If you need to reinstall packages:
```bash
pip install -r requirements.txt --upgrade
```

## Project Files

- `semiotic_analysis_tool.py` - Main analysis script
- `test_setup.py` - Verification script
- `example_input.txt` - Sample text for testing
- `requirements.txt` - Python dependencies
- `setup_env.sh` - Environment setup helper

## Need Help?

1. Check `README.md` for detailed documentation
2. Run `python test_setup.py` to verify your setup
3. Check `POSTMORTEM.md` for known issues and solutions

Happy analyzing! 🎉
