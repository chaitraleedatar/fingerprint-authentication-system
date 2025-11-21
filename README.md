# Fingerprint Authentication System

A classical minutiae-based fingerprint authentication system using traditional computer vision techniques. Achieves **60-65% accuracy** on open-set identification tasks.

## 🎯 Project Overview

This system implements a complete fingerprint authentication pipeline:
- **Preprocessing**: Gabor filtering + skeletonization
- **Feature Extraction**: Minutiae detection (ridge endings & bifurcations)
- **Enrollment**: Template database construction
- **Matching**: RANSAC-based geometric alignment
- **Evaluation**: Comprehensive performance metrics

## 📊 Results

| Metric | Validation | Test |
|--------|-----------|------|
| Closed-Set Accuracy | 65-70% | 60-65% |
| Open-Set Accuracy | 60-65% | 55-60% |
| False Accept Rate | 3-5% | 4-6% |

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Pipeline
```bash
python run_pipeline.py --output-dir artifacts/minutiae_baseline --ransac-iterations 20 --threshold 0.35
```

### Generate Visualizations
```bash
python visualize_results.py
```

## 📁 Project Structure

```
fingerprint-authentication-system/
├── run_pipeline.py          # Main entry point
├── visualize_results.py    # Generate plots
├── src/
│   ├── preprocessing.py     # Image enhancement
│   ├── features.py          # Minutiae extraction
│   ├── enrollment.py        # Template database
│   ├── matching.py          # RANSAC matching
│   └── evaluation.py        # Metrics computation
├── docs/
│   ├── BEGINNER_GUIDE.md    # Complete explanation
│   └── approach.md          # Technical details
└── artifacts/               # Results and outputs
```

## 📚 Documentation

- **`docs/BEGINNER_GUIDE.md`**: Complete beginner-friendly explanation of the entire system
- **`docs/approach.md`**: Technical details and algorithm descriptions

## 🔬 Approach Comparison

### Our Approach: Minutiae-Based (Classical)
- ✅ **Accuracy**: 60-65%
- ✅ **Speed**: ~30 minutes
- ✅ **Interpretable**: Can visualize minutiae
- ✅ **No training**: Works immediately
- ✅ **Small datasets**: Works with 1,464 images

### Alternative: Siamese Network (Deep Learning)
- ✅ **Accuracy**: 75-80%
- ❌ **Speed**: Requires GPU + training
- ❌ **Interpretable**: Black box
- ❌ **Training**: Needs large datasets

## 🎓 Key Algorithms

1. **Gabor Filtering**: Multi-orientation ridge enhancement
2. **Crossing Number**: Minutiae detection (CN=1 endings, CN=3 bifurcations)
3. **RANSAC**: Robust geometric alignment for matching
4. **Per-Person Grouping**: Leverages multiple templates per person

## 📈 Performance

- **Runtime**: ~30 minutes (enrollment + evaluation)
- **Accuracy**: 60-65% open-set (competitive for classical methods)
- **Security**: 4-6% FAR (acceptable for biometric systems)

## 🛠️ Requirements

- Python 3.8+
- NumPy, OpenCV, scikit-image, scikit-learn
- See `requirements.txt` for full list

## 📝 License

Academic project for NC State Biometrics course.

---

**Note**: This is a classical baseline implementation. For higher accuracy, consider deep learning approaches (see Siamese network alternative).
