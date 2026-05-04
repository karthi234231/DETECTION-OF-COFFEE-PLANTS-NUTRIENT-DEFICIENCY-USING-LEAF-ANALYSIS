import os
import numpy as np
import cv2
import pickle
import base64
import streamlit as st
from tensorflow.keras.models import load_model

# ── Optional YOLO ───────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = os.path.exists("weights/best.pt")
except ImportError:
    YOLO_AVAILABLE = False

# ── Deficiency data ─────────────────────────────────────────────────────────────
deficiency_info = {
    "nitrogen-N": {
        "icon": "🟡",
        "Symptoms": [
            "Older leaves turn yellow (chlorosis), starting from tips and edges.",
            "Stunted plant growth and smaller leaves."
        ],
        "Natural Remedies": [
            "Add well-rotted manure or compost to the soil.",
            "Grow nitrogen-fixing plants (e.g., clover or cowpeas).",
            "Use fish emulsion or seaweed extract as foliar sprays."
        ]
    },
    "phosphorus-P": {
        "icon": "🟣",
        "Symptoms": [
            "Dark green or purplish tint on older leaves.",
            "Slow growth and delayed flowering."
        ],
        "Natural Remedies": [
            "Apply bone meal for slow phosphorus release.",
            "Use rock phosphate for long-term replenishment.",
            "Chop and bury banana peels around the plant."
        ]
    },
    "potasium-K": {
        "icon": "🟤",
        "Symptoms": [
            "Leaf edges and tips turn brown and scorched.",
            "Weak branches and poor crop development."
        ],
        "Natural Remedies": [
            "Spread wood ash lightly around plants.",
            "Compost banana peels as they are rich in potassium.",
            "Apply poultry manure in small amounts."
        ]
    },
    "calcium-Ca": {
        "icon": "⚪",
        "Symptoms": [
            "New leaves appear deformed or distorted.",
            "Roots are weak and underdeveloped."
        ],
        "Natural Remedies": [
            "Crush and scatter eggshells around the plant for slow calcium release.",
            "Use agricultural lime to neutralize acidity and add calcium.",
            "Add gypsum if calcium is needed without altering soil pH."
        ]
    },
    "magnesium-Mg": {
        "icon": "🟢",
        "Symptoms": [
            "Yellowing between veins of older leaves, leaving green veins intact."
        ],
        "Natural Remedies": [
            "Dissolve 1 tbsp of Epsom salt in 1 gallon of water and spray on leaves.",
            "Add dolomite lime to correct magnesium and neutralize soil acidity.",
            "Use compost made from green leafy vegetables."
        ]
    },
    "iron-Fe": {
        "icon": "🔴",
        "Symptoms": [
            "Yellowing between veins of young leaves (interveinal chlorosis)."
        ],
        "Natural Remedies": [
            "Apply foliar sprays of chelated iron for quick absorption.",
            "Use spent coffee grounds to slightly acidify soil and improve iron uptake.",
            "Enrich compost with iron-rich materials like green leafy waste."
        ]
    },
    "boron-B": {
        "icon": "🟠",
        "Symptoms": [
            "Poor flowering and fruit set.",
            "Deformed or brittle leaves."
        ],
        "Natural Remedies": [
            "Add a small amount of borax to the soil (max 1 tsp per gallon).",
            "Use well-aged manure for trace boron content."
        ]
    },
    "manganese-Mn": {
        "icon": "🔵",
        "Symptoms": [
            "Interveinal chlorosis on younger leaves with brown spots."
        ],
        "Natural Remedies": [
            "Apply liquid seaweed extract for manganese and other trace elements.",
            "Use manganese sulfate as a foliar spray.",
            "Mulch with decomposed leaves to add manganese over time."
        ]
    },
    "healthy": {
        "icon": "✅",
        "Symptoms": [
            "No deficiency detected — the leaf appears healthy."
        ],
        "Natural Remedies": [
            "Maintain current fertilization and watering schedule.",
            "Continue regular soil health monitoring."
        ]
    },
    "more-deficiencies": {
        "icon": "⚠️",
        "Symptoms": [
            "Multiple nutrient deficiencies detected simultaneously.",
            "Complex overlapping symptoms across deficiency types."
        ],
        "Natural Remedies": [
            "Conduct a detailed soil test to identify all deficient nutrients.",
            "Apply a balanced multi-nutrient fertilizer.",
            "Consult an agronomist for a targeted treatment plan."
        ]
    }
}

# ── Background helper ───────────────────────────────────────────────────────────
def get_base64_of_image(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoLeaf — Coffee Nutrient Detector",
    page_icon="🌿",
    layout="wide"
)

image_base64 = get_base64_of_image("bg.jpg")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    /* Background */
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                    url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
        background-size: cover;
    }}

    /* Hide default streamlit header */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* Hero title */
    .hero {{
        text-align: center;
        padding: 2rem 1rem 1rem;
    }}
    .hero h1 {{
        font-size: 2.4rem;
        font-weight: 700;
        color: #f0e6d3;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
        margin-bottom: 0.3rem;
    }}
    .hero p {{
        font-size: 1rem;
        color: #d4c5b0;
        margin: 0;
    }}

    /* Stage badge */
    .stage-badge {{
        display: inline-block;
        background: rgba(75,56,42,0.85);
        color: #f0e6d3;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 8px;
    }}

    /* Upload card */
    .upload-card {{
        background: rgba(30,25,20,0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}

    /* Result card */
    .result-card {{
        background: rgba(20,18,15,0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.8rem;
        height: 100%;
    }}

    /* Prediction label */
    .pred-label {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #2ecc71;
        background: rgba(46,204,113,0.12);
        border: 1.5px solid rgba(46,204,113,0.4);
        border-radius: 10px;
        padding: 10px 18px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }}

    /* Confidence bar container */
    .conf-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0.6rem 0 1.2rem;
    }}
    .conf-bar-bg {{
        flex: 1;
        height: 10px;
        background: rgba(255,255,255,0.15);
        border-radius: 5px;
        overflow: hidden;
    }}
    .conf-bar-fill {{
        height: 100%;
        border-radius: 5px;
        transition: width 0.6s ease;
    }}
    .conf-text {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #f0e6d3;
        min-width: 48px;
        text-align: right;
    }}

    /* Info blocks */
    .info-block {{
        background: rgba(255,255,255,0.06);
        border-left: 3px solid #c8a96e;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin-top: 12px;
    }}
    .info-block.remedy {{
        border-left-color: #2ecc71;
    }}
    .info-block h4 {{
        color: #c8a96e;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0 0 8px;
    }}
    .info-block.remedy h4 {{
        color: #2ecc71;
    }}
    .info-block ul {{
        margin: 0;
        padding-left: 1.2rem;
        color: #d4c5b0;
        font-size: 0.88rem;
        line-height: 1.7;
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: rgba(15,12,10,0.85) !important;
        backdrop-filter: blur(10px);
    }}
    section[data-testid="stSidebar"] * {{
        color: #d4c5b0 !important;
    }}

    /* Image caption */
    .img-caption {{
        text-align: center;
        font-size: 0.78rem;
        color: #a0907e;
        margin-top: 6px;
        letter-spacing: 0.5px;
    }}

    /* Divider */
    .divider {{
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 1rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 CoLeaf")
    st.markdown("**Coffee Leaf Nutrient Deficiency Detector**")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("1. **YOLO** — locates the deficient region on the leaf")
    st.markdown("2. **MobileNet** — classifies the deficiency type")
    st.markdown("---")
    st.markdown("**Detectable Deficiencies**")
    for key, val in deficiency_info.items():
        label = key.replace("-", " ").title()
        st.markdown(f"{val['icon']} {label}")

# ── Model loaders ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_mobilenet():
    return load_model("best_mobilenet_model.keras")

@st.cache_resource(show_spinner=False)
def load_yolo():
    if not YOLO_AVAILABLE:
        return None
    return _YOLO("weights/best.pt")

@st.cache_resource()
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

# ── Processing ──────────────────────────────────────────────────────────────────
def classify(model, le, image):
    img = cv2.resize(image, (128, 128)).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)
    idx = int(np.argmax(preds))
    confidence = float(preds[0][idx]) * 100
    label = le.inverse_transform([idx])[0]
    return label, confidence

def confidence_color(conf):
    if conf >= 75:
        return "#2ecc71"
    elif conf >= 50:
        return "#f39c12"
    return "#e74c3c"

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 CoLeaf</h1>
    <p>Upload a coffee leaf image to detect and classify nutrient deficiencies</p>
</div>
""", unsafe_allow_html=True)

# ── Load models ──────────────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    mobilenet = load_mobilenet()
    yolo      = load_yolo()
    le        = load_label_encoder()

# ── Upload ───────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a coffee leaf image",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

# ── Pipeline ─────────────────────────────────────────────────────────────────────
if uploaded_file:
    file_bytes     = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    display_image  = original_image.copy()
    classify_target = original_image

    # Stage 1 — YOLO region detection
    yolo_ran = False
    if yolo is not None:
        yolo_results = yolo(original_image)
        display_image = yolo_results[0].plot()
        boxes = yolo_results[0].boxes
        if boxes is not None and len(boxes) > 0:
            best_idx = int(boxes.conf.argmax())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
            classify_target = original_image[y1:y2, x1:x2]
        yolo_ran = True

    # Stage 2 — MobileNet classification
    predicted_class, confidence = classify(mobilenet, le, classify_target)
    info = deficiency_info.get(predicted_class, {})
    color = confidence_color(confidence)
    display_label = predicted_class.replace("-", " ").title()

    # ── Layout ──────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        if yolo_ran:
            st.markdown('<span class="stage-badge">Stage 1 — YOLO Detection</span>', unsafe_allow_html=True)
            caption = "Deficient region highlighted"
        else:
            caption = "Uploaded image"
        st.image(display_image, channels="BGR", use_column_width=True)
        st.markdown(f'<p class="img-caption">{caption}</p>', unsafe_allow_html=True)

    with col2:
        st.markdown('<span class="stage-badge">Stage 2 — MobileNet Classification</span>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        icon = info.get("icon", "🔬")
        st.markdown(f'<div class="pred-label">{icon} {display_label}</div>', unsafe_allow_html=True)

        # Confidence bar
        st.markdown(f"""
        <div class="conf-row">
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence:.1f}%; background:{color};"></div>
            </div>
            <span class="conf-text" style="color:{color};">{confidence:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        if info:
            symptoms_html = "".join(f"<li>{s}</li>" for s in info["Symptoms"])
            remedies_html = "".join(f"<li>{r}</li>" for r in info["Natural Remedies"])

            st.markdown(f"""
            <div class="info-block">
                <h4>Symptoms</h4>
                <ul>{symptoms_html}</ul>
            </div>
            <div class="info-block remedy">
                <h4>Natural Remedies</h4>
                <ul>{remedies_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
