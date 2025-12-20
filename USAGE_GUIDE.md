# Semiotic Analysis Tool - Usage Guide

## 🎯 Quick Start

### 1. Organize Your Files

```bash
# Put files you want to analyze in the data/ folder
cp your_file.txt data/

# Large files or files you're not analyzing? Put them in archive
mv large_file.txt data/archive/
```

### 2. Run Analysis

```bash
# Activate environment
source venv/bin/activate

# Run full analysis (processes ALL .txt files in data/)
python semiotic_analysis_tool.py

# Or run quick analysis on a specific file
python quick_analysis.py your_file.txt
```

### 3. View Results

All outputs are saved in the `output/` folder:
- `semiotic_analysis_output.json` - Complete analysis data
- `*.png` - Visualization images (matrices, networks)
- `*.ipynb` - Jupyter notebook for interactive exploration
- `*.html` - Interactive discourse tree visualizations

## 📁 Folder Structure

```
Semiotic-Analysis-Tool/
├── data/                   ← INPUT: Place your files here
│   ├── README.md
│   ├── test_sample.txt     (Example - small file for testing)
│   └── archive/            (Storage for large/unused files)
│       └── Autobiography_of_a_Yogi.txt
│
├── output/                 ← OUTPUT: Analysis results saved here
│   ├── *.json             (Structured data)
│   ├── *.png              (Visualizations)
│   ├── *.ipynb            (Jupyter notebooks)
│   └── *.html             (Interactive visualizations)
│
├── semiotic_analysis_tool.py   (Main analysis script)
├── quick_analysis.py           (Fast analysis for large files)
└── venv/                       (Python virtual environment)
```

## 🔧 Key Improvements

### ✅ Organized Structure
- **Before**: Files scattered everywhere, script analyzed itself
- **After**: Clean separation - `data/` for input, `output/` for results

### ⚡ Performance Optimizations
- **Lazy Loading**: ML models only load when needed (faster startup)
- **No Self-Analysis**: Script only processes files in `data/` folder
- **Archive Support**: Keep large files separate in `data/archive/`

### 🎛️ Two Analysis Modes

1. **Full Analysis** (`semiotic_analysis_tool.py`)
   - Complete semiotic analysis with all features
   - 14 processing phases
   - Multiple visualizations
   - Best for: Small to medium files

2. **Quick Analysis** (`quick_analysis.py <filename>`)
   - Faster processing
   - Skips slow operations (LIME explanations)
   - Best for: Large files like Autobiography_of_a_Yogi.txt

## 💡 Usage Examples

### Example 1: Analyze a Small File
```bash
source venv/bin/activate
cp my_essay.txt data/
python semiotic_analysis_tool.py
ls output/  # View results
```

### Example 2: Analyze Multiple Files
```bash
# The script will process ALL .txt files in data/
cp file1.txt file2.txt file3.txt data/
python semiotic_analysis_tool.py
```

### Example 3: Quick Analysis of Large File
```bash
source venv/bin/activate
python quick_analysis.py Autobiography_of_a_Yogi.txt
# Output: output/quick_analysis_Autobiography_of_a_Yogi.json
```

### Example 4: Clean Up Between Runs
```bash
# Remove old output files
rm output/*.json output/*.png output/*.ipynb

# Run new analysis
python semiotic_analysis_tool.py
```

## 📊 Understanding Output Files

### JSON Files (`*.json`)
Complete structured data including:
- Signs and their contexts
- Sentiment analysis results
- Topic modeling
- Named entities
- Relationship matrices
- Statistical metrics

### Visualizations (`*.png`)
- `syntagmatic_matrix.png` - Word co-occurrence heatmap
- `paradigmatic_matrix.png` - Semantic similarity heatmap
- `sign_network.png` - Network graph of sign relationships

### Interactive Files
- `*.ipynb` - Open in Jupyter Notebook for interactive analysis
- `*.html` - Open in browser for interactive discourse trees

## ⚠️ Important Notes

1. **Processing All Files**: The main script processes **ALL** `.txt` files in the `data/` folder
   - To analyze only one file, move others to `data/archive/`
   - Or use `quick_analysis.py` for single-file analysis

2. **Large Files Take Time**: 
   - Files like Autobiography_of_a_Yogi.txt (960KB) can take 5-15 minutes
   - Use `quick_analysis.py` for faster results on large files
   - Progress indicators show what phase is running

3. **File Types Supported**:
   - Text files: `.txt`
   - Images: `.png`, `.jpg`, `.jpeg` (OCR will extract text)
   - Hidden files (starting with `.`) are ignored
   - Subdirectories are ignored

4. **Memory Usage**:
   - Large files require significant RAM
   - Multiple large files may cause memory issues
   - Process one large file at a time

## 🔍 Troubleshooting

### Script Freezes or Hangs
- **Cause**: Too many/large files being processed
- **Solution**: Move large files to `data/archive/`

### "No input data found" Error
- **Cause**: No `.txt` files in `data/` folder
- **Solution**: Copy your text files to `data/`

### Out of Memory
- **Cause**: File too large for available RAM
- **Solution**: Use `quick_analysis.py` instead

### Models Loading Slowly
- **Normal**: First run downloads ML models (2-3GB)
- **Subsequent runs**: Much faster with lazy loading

## 📈 Performance Tips

1. **Test with small files first** - Use `test_sample.txt` to verify setup
2. **One large file at a time** - Move others to archive
3. **Clear output between runs** - Prevents confusion
4. **Use quick_analysis for large files** - Much faster for initial exploration
5. **Archive unused files** - Keeps `data/` folder clean

## 🎓 Next Steps

- View generated visualizations in `output/`
- Open `.ipynb` file in Jupyter Notebook for interactive analysis
- Check JSON files for detailed numerical results
- Try different texts and compare results!
