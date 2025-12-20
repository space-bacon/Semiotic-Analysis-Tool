# Data Folder

This folder contains your input files for semiotic analysis.

## Supported File Types

- **Text files** (`.txt`) - Plain text documents
- **Images** (`.png`, `.jpg`, `.jpeg`) - OCR will be performed to extract text

## Usage

1. Place your files in this folder
2. Run the analysis from the project root:
   ```bash
   source venv/bin/activate
   python semiotic_analysis_tool.py
   ```

## Current Files

- `Autobiography_of_a_Yogi.txt` - Sample text for analysis
- `example_input.txt` - Example input file
- `test_sample.txt` - Short test file for quick testing

## Notes

- The script will automatically scan this folder for all compatible files
- Hidden files (starting with `.`) are ignored
- Subdirectories are not scanned (files must be directly in this folder)
- All analysis results are saved to the `output/` folder
