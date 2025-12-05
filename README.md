# ThinkAHead

**Motorcycle Helmet Violation & Overloading Detection System**

A real-time computer vision system that detects motorcycle safety violations including helmet non-compliance and overloading (≥3 riders). Built for CS 176 (Computer Vision) Final Project.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-red)

---

## Table of Contents

- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Model Details](#model-details)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Structure](#project-structure)
- [Authors](#authors)

---

## Installation

### Prerequisites

- Python 3.10+

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/thinkahead.git
cd thinkahead
```

### Step 2: Create Conda Environment

```bash
conda create -n thinkahead python=3.10
conda activate thinkahead
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```



---

## Usage Guide


### Web Interface (Streamlit)

```bash
streamlit run src/app_streamlit.py
```

Then open http://localhost:8501 in your browser.

**Features:**
- Upload images or videos
- Adjust detection confidence threshold
- View real-time statistics
- Download violation logs as CSV

---

## Model Details

### Architecture

- **Base Model:** YOLOv8m (Medium)
- **Parameters:** 25.9M
- **Input Size:** 640×640
- **Framework:** Ultralytics + PyTorch

### Detection Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | `motorcycle` | Two-wheeled motor vehicle |
| 1 | `rider` | Person on motorcycle |
| 2 | `helmet` | Person wearing helmet |
| 3 | `no_helmet` | Person without helmet |
| 4 | `license_plate` | Vehicle registration plate |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 8 |
| Image Size | 640×640 |
| Optimizer | AdamW |
| Learning Rate | 0.001 → 0.00001 |
| Augmentation | Mosaic, HSV, Flip, Scale |

### Dataset

| Split | Images |
|-------|--------|
| Train | 1,798 |
| Validation | 224 |
| Test | 226 |
| **Total** | **2,248** |

**Sources:**
- Roboflow Helmet-and-Number-Plate dataset
- Roboflow Triple Riding Detection dataset
- Kaggle Helmet Detection dataset

### Post-Processing Pipeline

```
YOLO Detection → Geometric Grouping → Violation Logic → OCR (optional)
```

1. **Geometric Grouping:** Associates riders with motorcycles based on bounding box overlap and proximity
2. **Helmet Detection:** Checks if helmet bbox overlaps upper portion of rider bbox
3. **Overload Detection:** Flags motorcycles with ≥3 associated riders
4. **OCR Module:** Extracts license plate text for violations (using EasyOCR)

---

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **mAP@50** | 76.1% |
| **mAP@50-95** | 39.3% |
| **Precision** | 74.9% |
| **Recall** | 77.2% |

### Per-Class Performance (AP@50)

| Class | AP@50 | Notes |
|-------|-------|-------|
| license_plate | 99.5% | Excellent detection |
| helmet | 84.5% | Strong performance |
| motorcycle | 80.2% | Reliable detection |
| no_helmet | 59.6% | Moderate - challenging class |
| rider | 27.1% | Low - often subsumed by helmet/no_helmet |

### Confusion Matrix Analysis

- **Motorcycle:** 85% correct classification
- **Helmet:** 87% correct classification  
- **No_helmet:** 63% correct (some confusion with background)
- **Rider:** 29% correct (often classified as background)


---

## Limitations

### Detection Limitations

1. **False positives in dense traffic** - Crowded scenes may cause incorrect rider-motorcycle associations
2. **Extreme viewing angles** - Side/rear views may miss helmet detection
3. **Nighttime performance** - Depends heavily on lighting conditions
4. **Rider occlusion** - Overloading detection fails when riders occlude each other
5. **Small objects** - Distant motorcycles may not be detected reliably

### Dataset Limitations

1. **Class imbalance** - "Rider" class underrepresented relative to helmet/no_helmet
2. **Geographic bias** - Primarily South Asian traffic scenarios
3. **License plate formats** - Optimized for certain plate styles

### Technical Limitations

1. **VRAM requirements** - Minimum 4GB for inference
2. **Real-time constraints** - May require frame skipping on lower-end hardware
3. **OCR accuracy** - License plate reading affected by motion blur and resolution

---

## 🔮 Future Improvements

- [ ] Add temporal tracking (DeepSORT) for consistent ID assignment across frames
- [ ] Implement confidence calibration for violation alerts
- [ ] Add support for more license plate formats
- [ ] Nighttime enhancement using IR cameras or low-light models
- [ ] Multi-camera support for wider coverage
- [ ] Database integration for violation logging
- [ ] Expand to other violations (wrong-way driving, lane filtering)
- [ ] Edge deployment (Jetson Nano, Raspberry Pi)
- [ ] Integration with traffic management systems

---

##  References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Streamlit](https://streamlit.io/)

### Datasets

- Roboflow Universe - Helmet and Number Plate Detection
- Roboflow Universe - Triple Riding Detection
- Kaggle - Helmet Detection Dataset

---

## Authors

**Noval & Sacramento**

CS 176 - Computer Vision  
University of the Philippines Diliman  
2nd Semester, AY 2024-2025

---

## 📄 License

This project is for educational purposes as part of CS 176 coursework.

---