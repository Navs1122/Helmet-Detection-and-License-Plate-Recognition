#  Helmet Detection and License Plate Recognition

A Computer Vision project for automated traffic violation detection using **YOLOv5**, **CNN**, and **OCR**.

> **B.Tech Major Project** — Department of Computer Science and Engineering  
> Gokaraju Rangaraju Institute of Engineering and Technology (GRIET), Hyderabad  
> Academic Year: 2023–2024

---

## Abstract

This project proposes an automated solution for enhancing road safety through:
- **Helmet Detection** using YOLOv5
- **License Plate Recognition** using CNN + OCR (Tesseract)
- **Automatic Challan Generation** for traffic violators

The system detects motorbike riders, checks for helmet compliance, extracts license plate numbers, and notifies registered vehicle owners.

---

## System Architecture

```
Input Image
    │
    ▼
[Module 1] YOLOv5 → Detect Person + Motorbike
    │
    ▼ (if detected)
[Module 2] YOLOv5 → Helmet Detection
    │
    ▼ (if no helmet)
[Module 3] CNN + OCR → License Plate Recognition
    │
    ▼
[Module 4] Generate Challan → Notify Owner
```

---

## System Requirements

### Software
- Python 3.12.3
- Libraries: TensorFlow, OpenCV, NumPy, Tesseract OCR, PIL, lxml

### Hardware
- OS: Windows
- Processor: Intel i5 or above
- RAM: 4 GB or above
- Storage: 50 GB

---

## Project Structure

```
helmet-detection/
├── data/
│   ├── train/          # Training images (JPEGImages + Annotations)
│   └── test/           # Test images
├── models/
│   └── faster_RCNN_Inception_v2/
│       └── helmet_label_map.pbtxt
├── utils/
│   └── helpers.py
├── outputs/            # Detection results saved here
├── create_tf_record.py # TFRecord generation script
├── detect.py           # Inference / detection script
├── gui.py              # Graphical User Interface
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/helmet-detection.git
cd helmet-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to system PATH

### 4. Prepare the Dataset
Place training images in `data/train/JPEGImages/` and XML annotations in `data/train/Annotations/`.

### 5. Generate TF Records
```bash
python create_tf_record.py
```

### 6. Run the Application
```bash
python gui.py
```

---

## Results

| Metric | Score |
|--------|-------|
| YOLOv5 Accuracy | 95.83% |
| YOLOv5 Precision | 96.43% |
| YOLOv5 Recall | 95.45% |
| YOLOv5 F-Score | 95.77% |

---

## Test Cases Summary

| Test Case | Description | Status |
|-----------|-------------|--------|
| Rider without helmet (front view) | Sends fine message | ✅ Pass |
| Rider wearing helmet | No action taken | ✅ Pass |
| Non-rider/trespasser | No action taken | ✅ Pass |
| Rider without helmet (side view) | License plate not visible | ❌ Fail |
| Rider without helmet, invisible plate | Cannot send fine | ❌ Fail |

---

## Future Scope

- Integration with augmented reality and edge computing
- Differentiation between helmet types
- IoT integration for smart city monitoring
- Global standards compliance

---

## Publication

**"Smart Approach for Helmet and Number Plate Detection Using ML"**  
Published in ZKG International Journal  
🔗 [View Paper](https://zkginternational.com/archive/volume9/SMART-APPROACH-FOR-HELMET-AND-NUMBER-PLATE-DETECTION-USING-ML.pdf)

---

## References

1. YOLO V4-based helmet detection — Bin Yang, Jie Wang (2022)
2. Deep learning for motorcycle helmet detection — Hanhe Lin, Felix Siebert (2019)
3. Automatic License Plate Recognition based on YOLO — Rayson Laroca et al. (2019)
4. Detection of Motorcyclists without Helmet using CNN — Chalavadi Vishnu (2017)
5. Two Wheelers Traffic Violation Finder — Hari Vignesh, Arul Selvam (2023)
