# CoLeaf — Coffee Leaf Nutrient Deficiency Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nutrient-detection.streamlit.app/)

**Live Demo:** https://nutrient-detection.streamlit.app/

<a href="https://nutrient-detection.streamlit.app" target="_blank">
  <img 
    src="https://api.microlink.io/?url=https%3A%2F%2Fnutrient-detection.streamlit.app&%20screenshot=true&meta=false&embed=screenshot.url"
    alt="CoLeaf — Live App Preview"
    width="100%"
  />
</a>

A two-stage deep learning app that detects and classifies nutrient deficiencies in coffee plant leaves.

## How It Works

| Stage | Model | Role |
|---|---|---|
| 1 | YOLOv8 | Detects and localizes the deficient region on the leaf |
| 2 | MobileNet | Classifies the type of nutrient deficiency |

## Detectable Classes

| Deficiency | Label |
|---|---|
| Nitrogen | nitrogen-N |
| Phosphorus | phosphorus-P |
| Potassium | potasium-K |
| Calcium | calcium-Ca |
| Magnesium | magnesium-Mg |
| Iron | iron-Fe |
| Boron | boron-B |
| Manganese | manganese-Mn |
| Multiple Deficiencies | more-deficiencies |
| Healthy | healthy |

## Setup & Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/DETECTION-OF-COFFEE-PLANTS-NUTRIENT-DEFICIENCY-USING-LEAF-ANALYSIS.git
cd DETECTION-OF-COFFEE-PLANTS-NUTRIENT-DEFICIENCY-USING-LEAF-ANALYSIS
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add model files** (if not already present)
- `best_mobilenet_model.keras` — MobileNet classification model
- `weights/best.pt` — YOLOv8 detection model *(optional — app works without it)*
- `label_encoder.pkl` — class label encoder
- `bg.jpg` — background image

**4. Run the app**
```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → main file: `app.py`
4. Click **Deploy**

## Deploy to Hugging Face Spaces (Free)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) with SDK: **Streamlit**
2. Upload `app.py`, `requirements.txt`, `best_mobilenet_model.keras`, `label_encoder.pkl`, `bg.jpg`
3. Space auto-deploys on file upload

## Project Structure

```
├── app.py                      # Main Streamlit app (YOLO + MobileNet pipeline)
├── Augmentation.py             # Dataset augmentation script
├── best_mobilenet_model.keras  # Trained MobileNet model
├── label_encoder.pkl           # Scikit-learn LabelEncoder (10 classes)
├── bg.jpg                      # UI background image
├── weights/
│   └── best.pt                 # YOLOv8 weights (add manually)
└── requirements.txt
```
