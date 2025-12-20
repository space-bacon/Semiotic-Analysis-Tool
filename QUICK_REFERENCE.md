# Semiotic Analysis Tool - Quick Reference

## 📂 Where Files Go

```
data/           ← PUT YOUR .TXT FILES HERE
data/archive/   ← LARGE FILES GO HERE (won't be processed)
output/         ← RESULTS APPEAR HERE (automatic)
```

## ⚡ Quick Commands

### Analyze Files
```bash
source venv/bin/activate
python semiotic_analysis_tool.py
```
→ Processes ALL .txt files in `data/` folder

### Quick Analysis (Large Files)
```bash
python quick_analysis.py filename.txt
```
→ Faster, single file analysis

### Clean Up
```bash
rm output/*.json output/*.png
```
→ Remove old results before new run

## 🎯 Common Tasks

### Task: Analyze one small file
```bash
cp my_file.txt data/
python semiotic_analysis_tool.py
```

### Task: Analyze large file (fast)
```bash
python quick_analysis.py Autobiography_of_a_Yogi.txt
```

### Task: Analyze only one file (when multiple exist)
```bash
# Move others to archive temporarily
mv data/*.txt data/archive/
mv data/archive/my_file.txt data/
python semiotic_analysis_tool.py
```

### Task: Analyze Autobiography book (full analysis)
```bash
mv data/archive/Autobiography_of_a_Yogi.txt data/
python semiotic_analysis_tool.py
# Takes 5-15 minutes - be patient!
```

## 📊 Output Files Explained

| File | Description |
|------|-------------|
| `semiotic_analysis_output.json` | Complete analysis data (JSON format) |
| `output_analysis.json` | Alternative JSON export |
| `output_analysis.ipynb` | Jupyter notebook (interactive) |
| `syntagmatic_matrix.png` | Word co-occurrence visualization |
| `paradigmatic_matrix.png` | Semantic similarity visualization |
| `sign_network.png` | Network graph of relationships |
| `*_discourse_tree.html` | Interactive discourse tree (browser) |

## ⚠️ Important Rules

1. **Only .txt files in `data/` are processed**
   - Move large files to `data/archive/` when not using them
   
2. **Script processes ALL files in `data/` folder**
   - Use `quick_analysis.py` for single files
   - Or move unwanted files to archive

3. **Large files take time**
   - test_sample.txt: ~30 seconds
   - Autobiography_of_a_Yogi.txt: 5-15 minutes

4. **Clear output/ between runs**
   - Prevents confusion about which results are current

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Script freezes | Move large files to `data/archive/` |
| "No input data" | Add .txt files to `data/` folder |
| Takes too long | Use `quick_analysis.py` instead |
| Ran on wrong files | Move unwanted files to `data/archive/` |

## 📖 Full Documentation

- **USAGE_GUIDE.md** - Detailed usage instructions
- **REORGANIZATION_COMPLETE.md** - What changed and why
- **data/README.md** - Data folder instructions

## 🎓 Tips

- ✓ Always test with `test_sample.txt` first
- ✓ Use `data/archive/` for files you're not currently analyzing
- ✓ Clear `output/` before important runs
- ✓ Large files? Use `quick_analysis.py` first to preview

---

**Status**: ✅ Ready to use!
