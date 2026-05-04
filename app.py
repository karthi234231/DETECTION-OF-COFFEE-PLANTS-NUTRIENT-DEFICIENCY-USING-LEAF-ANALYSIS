import os
import io
import numpy as np
import cv2
import pickle
import base64
import streamlit as st
from datetime import datetime
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

# ── Helpers ─────────────────────────────────────────────────────────────────────
def get_base64_of_image(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def confidence_color(conf):
    if conf >= 75:
        return "#2ecc71"
    elif conf >= 50:
        return "#f39c12"
    return "#e74c3c"

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CoLeaf", page_icon="🌿", layout="wide")

image_base64 = get_base64_of_image("bg.jpg")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.58), rgba(0,0,0,0.58)),
                    url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
        background-size: cover;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .hero {{ text-align: center; padding: 2rem 1rem 0.5rem; }}
    .hero h1 {{
        font-size: 2.4rem; font-weight: 700; color: #f0e6d3;
        letter-spacing: 2px; text-transform: uppercase;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6); margin-bottom: 0.3rem;
    }}
    .hero p {{ font-size: 1rem; color: #d4c5b0; margin: 0; }}
    .stage-badge {{
        display: inline-block; background: rgba(75,56,42,0.85);
        color: #f0e6d3; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 1px; text-transform: uppercase;
        padding: 4px 12px; border-radius: 20px; margin-bottom: 8px;
    }}
    .result-card {{
        background: rgba(20,18,15,0.78); backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px; padding: 1.5rem;
    }}
    .pred-label {{
        font-size: 1.4rem; font-weight: 700; color: #2ecc71;
        background: rgba(46,204,113,0.12);
        border: 1.5px solid rgba(46,204,113,0.4);
        border-radius: 10px; padding: 8px 16px;
        display: inline-block; margin-bottom: 0.4rem;
    }}
    .top3-row {{
        display: flex; align-items: center; gap: 8px; margin: 5px 0;
    }}
    .top3-label {{
        font-size: 0.8rem; color: #d4c5b0;
        min-width: 130px; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis;
    }}
    .top3-bar-bg {{
        flex: 1; height: 8px; background: rgba(255,255,255,0.12);
        border-radius: 4px; overflow: hidden;
    }}
    .top3-bar-fill {{ height: 100%; border-radius: 4px; }}
    .top3-pct {{
        font-size: 0.78rem; font-weight: 600;
        min-width: 42px; text-align: right;
    }}
    .info-block {{
        background: rgba(255,255,255,0.06);
        border-left: 3px solid #c8a96e;
        border-radius: 0 10px 10px 0;
        padding: 10px 14px; margin-top: 10px;
    }}
    .info-block.remedy {{ border-left-color: #2ecc71; }}
    .info-block h4 {{
        color: #c8a96e; font-size: 0.8rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase; margin: 0 0 6px;
    }}
    .info-block.remedy h4 {{ color: #2ecc71; }}
    .info-block ul {{
        margin: 0; padding-left: 1.1rem; color: #d4c5b0;
        font-size: 0.84rem; line-height: 1.65;
    }}
    .multi-card {{
        background: rgba(231,76,60,0.1); border: 1px solid rgba(231,76,60,0.35);
        border-radius: 10px; padding: 10px 14px; margin-top: 10px;
    }}
    .multi-card h4 {{
        color: #e74c3c; font-size: 0.8rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase; margin: 0 0 6px;
    }}
    .multi-card ul {{
        margin: 0; padding-left: 1.1rem; color: #d4c5b0;
        font-size: 0.84rem; line-height: 1.65;
    }}
    .hist-item {{
        background: rgba(255,255,255,0.06); border-radius: 8px;
        padding: 8px 10px; margin-bottom: 6px;
        font-size: 0.8rem; color: #d4c5b0;
    }}
    .img-caption {{
        text-align: center; font-size: 0.76rem;
        color: #a0907e; margin-top: 4px; letter-spacing: 0.5px;
    }}
    section[data-testid="stSidebar"] {{
        background: rgba(15,12,10,0.88) !important;
        backdrop-filter: blur(10px);
    }}
    section[data-testid="stSidebar"] * {{ color: #d4c5b0 !important; }}
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(255,255,255,0.05); border-radius: 10px; padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #d4c5b0 !important; border-radius: 8px;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(75,56,42,0.7) !important; color: #f0e6d3 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 CoLeaf")
    st.markdown("**Coffee Leaf Nutrient Deficiency Detector**")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("1. **YOLO** — locates deficient region")
    st.markdown("2. **MobileNet** — classifies deficiency type")
    yolo_status = "✅ Loaded" if YOLO_AVAILABLE else "⚠️ Not found (using full image)"
    st.markdown(f"YOLO: {yolo_status}")
    st.markdown("---")
    st.markdown("**Detectable Classes**")
    for key, val in deficiency_info.items():
        st.markdown(f"{val['icon']} {key.replace('-', ' ').title()}")
    st.markdown("---")
    st.markdown("**Recent Predictions**")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"""
            <div class="hist-item">
                {h['icon']} <b>{h['label']}</b><br>
                <span style="font-size:0.72rem;color:#888;">{h['conf']:.1f}% &nbsp;·&nbsp; {h['time']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<span style='font-size:0.8rem;color:#666;'>No predictions yet</span>", unsafe_allow_html=True)

# ── Model loaders ────────────────────────────────────────────────────────────────
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

# ── Classification ───────────────────────────────────────────────────────────────
def classify_all(model, le, image):
    img = cv2.resize(image, (128, 128)).astype("float32") / 255.0
    preds = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    classes = le.inverse_transform(list(range(len(preds))))
    ranked = sorted(zip(classes, preds), key=lambda x: x[1], reverse=True)
    return ranked  # list of (class_name, confidence_float)

def build_report(top_label, confidence, top3, info, multi_detected):
    lines = [
        "CoLeaf — Nutrient Deficiency Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 40,
        f"Primary Detection : {top_label.replace('-', ' ').title()}",
        f"Confidence        : {confidence:.1f}%",
        "",
        "Top 3 Predictions:",
    ]
    for cls, conf in top3:
        lines.append(f"  {cls:<20} {conf*100:.1f}%")
    if multi_detected:
        lines += ["", "Possible Co-occurring Deficiencies (>15%):"]
        for cls, conf in multi_detected:
            lines.append(f"  {cls:<20} {conf*100:.1f}%")
    if info:
        lines += ["", "Symptoms:"]
        for s in info.get("Symptoms", []):
            lines.append(f"  • {s}")
        lines += ["", "Natural Remedies:"]
        for r in info.get("Natural Remedies", []):
            lines.append(f"  • {r}")
    return "\n".join(lines)

# ── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 CoLeaf</h1>
    <p>Upload or capture a coffee leaf to detect and classify nutrient deficiencies</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading models..."):
    mobilenet = load_mobilenet()
    yolo      = load_yolo()
    le        = load_label_encoder()

# ── Input tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

raw_image = None

with tab1:
    uploaded = st.file_uploader("Upload a coffee leaf image", type=["jpg","png","jpeg"],
                                label_visibility="collapsed")
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        raw_image  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

with tab2:
    camera_shot = st.camera_input("Take a photo of the leaf")
    if camera_shot:
        file_bytes = np.asarray(bytearray(camera_shot.read()), dtype=np.uint8)
        raw_image  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ── Pipeline ─────────────────────────────────────────────────────────────────────
if raw_image is not None:
    display_image   = raw_image.copy()
    classify_target = raw_image
    yolo_ran        = False

    # Stage 1 — YOLO
    if yolo is not None:
        results = yolo(raw_image)
        display_image = results[0].plot()
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            best_idx = int(boxes.conf.argmax())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
            classify_target = raw_image[y1:y2, x1:x2]
        yolo_ran = True

    # Stage 2 — MobileNet (all predictions)
    ranked      = classify_all(mobilenet, le, classify_target)
    top_label   = ranked[0][0]
    confidence  = ranked[0][1] * 100
    top3        = ranked[:3]
    info        = deficiency_info.get(top_label, {})

    # Multi-deficiency: show all others > 15% when top is more-deficiencies
    multi_detected = None
    if top_label == "more-deficiencies":
        multi_detected = [(cls, conf) for cls, conf in ranked[1:]
                         if conf > 0.15 and cls not in ("more-deficiencies", "healthy")]

    # Save to history
    st.session_state.history.append({
        "label": top_label.replace("-", " ").title(),
        "conf":  confidence,
        "icon":  info.get("icon", "🔬"),
        "time":  datetime.now().strftime("%H:%M:%S")
    })

    # ── Layout ───────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        stage_txt = "Stage 1 — YOLO Detection" if yolo_ran else "Uploaded Image"
        st.markdown(f'<span class="stage-badge">{stage_txt}</span>', unsafe_allow_html=True)
        st.image(display_image, channels="BGR", use_column_width=True)
        caption = "Deficient region highlighted" if yolo_ran else "Original image"
        st.markdown(f'<p class="img-caption">{caption}</p>', unsafe_allow_html=True)

    with col2:
        st.markdown('<span class="stage-badge">Stage 2 — MobileNet Classification</span>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # Top prediction
        icon = info.get("icon", "🔬")
        display_label = top_label.replace("-", " ").title()
        st.markdown(f'<div class="pred-label">{icon} {display_label}</div>', unsafe_allow_html=True)

        # Top-3 bars
        st.markdown("<div style='margin:10px 0 14px;'>", unsafe_allow_html=True)
        for cls, conf in top3:
            color = confidence_color(conf * 100)
            bar_label = cls.replace("-", " ").title()
            st.markdown(f"""
            <div class="top3-row">
                <span class="top3-label">{bar_label}</span>
                <div class="top3-bar-bg">
                    <div class="top3-bar-fill" style="width:{conf*100:.1f}%;background:{color};"></div>
                </div>
                <span class="top3-pct" style="color:{color};">{conf*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Multi-deficiency breakdown
        if multi_detected:
            items = "".join(f"<li>{cls.replace('-',' ').title()} — {conf*100:.1f}%</li>"
                            for cls, conf in multi_detected)
            st.markdown(f"""
            <div class="multi-card">
                <h4>Possible Co-occurring Deficiencies</h4>
                <ul>{items}</ul>
            </div>
            """, unsafe_allow_html=True)

        # Symptoms & Remedies
        if info:
            syms = "".join(f"<li>{s}</li>" for s in info["Symptoms"])
            rems = "".join(f"<li>{r}</li>" for r in info["Natural Remedies"])
            st.markdown(f"""
            <div class="info-block">
                <h4>Symptoms</h4><ul>{syms}</ul>
            </div>
            <div class="info-block remedy">
                <h4>Natural Remedies</h4><ul>{rems}</ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Download report
        report_text = build_report(top_label, confidence, top3, info, multi_detected)
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"coleaf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
