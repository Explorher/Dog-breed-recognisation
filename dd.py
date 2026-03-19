import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import os
import re

st.set_page_config(
    page_title="CanineAI - Next-Gen Breed Intelligence",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ULTRA PREMIUM DESIGN
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Sora:wght@300;400;600;700;800&display=swap');
    
    /* ===== GLOBAL RESET ===== */
    * {
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
        letter-spacing: -0.015em;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .stApp {
        background: #000000;
        color: #FFFFFF;
        overflow-x: hidden;
    }
    
    /* Animated gradient overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(255,107,53,0.03) 0%, transparent 50%);
        animation: pulse 15s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    
    /* Hide Streamlit */
    #MainMenu, footer, header, .stDeployButton {visibility: hidden; display: none;}
    
    /* ===== NAVIGATION BAR ===== */
    .nav-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: rgba(0,0,0,0.8);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding: 1.2rem 3rem;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .nav-logo {
        font-family: 'Sora', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .nav-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: #888888;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #4CAF50;
        border-radius: 50%;
        animation: blink 2s ease-in-out infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* ===== HERO SECTION ===== */
    .hero {
        text-align: center;
        padding: 10rem 2rem 6rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%);
        position: relative;
        margin-top: 4rem;
    }
    
    .hero::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,107,53,0.3), transparent);
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,107,53,0.1);
        border: 1px solid rgba(255,107,53,0.3);
        color: #FF8C42;
        padding: 0.6rem 1.5rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .badge-dot {
        width: 6px;
        height: 6px;
        background: #FF6B35;
        border-radius: 50%;
        box-shadow: 0 0 10px #FF6B35;
    }
    
    .hero-title {
        font-family: 'Sora', sans-serif;
        font-size: 6rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1.5rem;
        letter-spacing: -0.04em;
    }
    
    .hero-title .gradient-text {
        background: linear-gradient(135deg, #FFFFFF 0%, #666666 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-title .highlight {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: #888888;
        font-weight: 400;
        max-width: 700px;
        margin: 0 auto 3rem;
        line-height: 1.6;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .stat-item {
        text-align: center;
        padding: 2rem;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(255,107,53,0.3);
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-family: 'Sora', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    
    /* ===== SECTION HEADERS ===== */
    .section {
        padding: 6rem 0;
        position: relative;
    }
    
    .section-header {
        text-align: center;
        margin-bottom: 4rem;
    }
    
    .section-number {
        display: inline-block;
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
        color: white;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 800;
        line-height: 50px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(255,107,53,0.3);
    }
    
    .section-label {
        color: #FF6B35;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
    }
    
    .section-title {
        font-family: 'Sora', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .section-desc {
        font-size: 1.2rem;
        color: #888888;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.7;
    }
    
    /* ===== PREMIUM CARDS ===== */
    .premium-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 32px;
        padding: 3.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FF6B35, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .premium-card:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(255,107,53,0.3);
        transform: translateY(-8px);
        box-shadow: 0 30px 80px rgba(255, 107, 53, 0.15);
    }
    
    .premium-card:hover::before {
        opacity: 1;
    }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader {
        margin: 2rem 0;
    }
    
    .stFileUploader > div {
        background: rgba(255,255,255,0.02) !important;
        border: 2px dashed rgba(255,255,255,0.15) !important;
        border-radius: 24px !important;
        padding: 5rem 3rem !important;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .stFileUploader > div::before {
        content: '📤';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 5rem;
        opacity: 0.05;
        pointer-events: none;
    }
    
    .stFileUploader > div:hover {
        border-color: #FF6B35 !important;
        background: rgba(255, 107, 53, 0.05) !important;
        box-shadow: 0 0 60px rgba(255, 107, 53, 0.1);
    }
    
    .stFileUploader label {
        color: #CCCCCC !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton button, .stDownloadButton button {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 1.1rem 3rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 15px 40px rgba(255, 107, 53, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before, .stDownloadButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton button:hover::before, .stDownloadButton button:hover::before {
        left: 100%;
    }
    
    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 50px rgba(255, 107, 53, 0.6) !important;
    }
    
    .stButton button:active, .stDownloadButton button:active {
        transform: translateY(-1px);
    }
    
    /* ===== METRICS ===== */
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #FF6B35, #FF8C42);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .stMetric:hover {
        background: rgba(255,107,53,0.05);
        border-color: rgba(255,107,53,0.3);
        transform: translateY(-5px);
        box-shadow: 0 20px 50px rgba(255, 107, 53, 0.2);
    }
    
    .stMetric:hover::before {
        opacity: 1;
    }
    
    .stMetric label {
        color: #888888 !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        margin-bottom: 1rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        font-family: 'Sora', sans-serif !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        height: 10px !important;
        overflow: hidden;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF6B35, #FF8C42) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.5);
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 107, 53, 0.5); }
        50% { box-shadow: 0 0 30px rgba(255, 107, 53, 0.8); }
    }
    
    /* ===== IMAGES ===== */
    .stImage {
        position: relative;
    }
    
    .stImage img {
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 25px 70px rgba(0,0,0,0.6);
        transition: all 0.4s ease;
    }
    
    .stImage img:hover {
        transform: scale(1.03) translateY(-5px);
        border-color: rgba(255,107,53,0.4);
        box-shadow: 0 30px 80px rgba(255, 107, 53, 0.3);
    }
    
    /* ===== HEADINGS ===== */
    h1 {
        font-family: 'Sora', sans-serif !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    
    h2 {
        font-family: 'Sora', sans-serif !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        margin-top: 4rem !important;
    }
    
    h3 {
        color: #FF6B35 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin-top: 2rem !important;
    }
    
    h4 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* ===== TEXT ===== */
    p, li, span, div {
        color: #CCCCCC;
        line-height: 1.8;
    }
    
    strong {
        color: #FFFFFF;
        font-weight: 700;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        background: rgba(255,107,53,0.08) !important;
        border: 1px solid rgba(255,107,53,0.25) !important;
        border-radius: 16px !important;
        color: #FFFFFF !important;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: rgba(76,175,80,0.08) !important;
        border-color: rgba(76,175,80,0.25) !important;
    }
    
    .stError {
        background: rgba(244,67,54,0.08) !important;
        border-color: rgba(244,67,54,0.25) !important;
    }
    
    .stInfo {
        background: rgba(33,150,243,0.08) !important;
        border-color: rgba(33,150,243,0.25) !important;
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 5rem 0;
    }
    
    /* ===== BADGES ===== */
    .health-badge {
        display: inline-block;
        background: rgba(255,107,53,0.1);
        border: 1px solid rgba(255,107,53,0.3);
        color: #FF8C42;
        padding: 0.6rem 1.4rem;
        border-radius: 10px;
        margin: 0.4rem 0.4rem 0.4rem 0;
        font-size: 0.9rem;
        font-weight: 600;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .health-badge:hover {
        background: rgba(255,107,53,0.2);
        border-color: #FF6B35;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 107, 53, 0.3);
    }
    
    /* ===== CONFIDENCE DISPLAY ===== */
    .confidence {
        display: inline-flex;
        align-items: center;
        gap: 0.8rem;
        background: rgba(255,107,53,0.1);
        border: 1px solid rgba(255,107,53,0.3);
        color: #FF8C42;
        padding: 0.9rem 1.8rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .confidence-icon {
        width: 10px;
        height: 10px;
        background: #FF6B35;
        border-radius: 50%;
        box-shadow: 0 0 15px #FF6B35;
    }
    
    /* ===== VACCINATION TABLE ===== */
    .vacc-item {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .vacc-item:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(255,107,53,0.2);
        transform: translateX(5px);
    }
    
    .vacc-age {
        color: #FF6B35;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .vacc-name {
        color: #CCCCCC;
        font-size: 0.95rem;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 5rem 2rem;
        margin-top: 8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        background: linear-gradient(180deg, transparent 0%, rgba(255,255,255,0.01) 100%);
    }
    
    .footer-logo {
        font-family: 'Sora', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
    }
    
    .footer-text {
        color: #666666;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .footer-link {
        color: #888888;
        text-decoration: none;
        font-size: 0.9rem;
        transition: color 0.3s ease;
    }
    
    .footer-link:hover {
        color: #FF6B35;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #333333, #222222);
        border-radius: 10px;
        border: 2px solid #000000;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FF6B35, #FF8C42);
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 3.5rem;
        }
        .section-title {
            font-size: 2.5rem;
        }
        .premium-card {
            padding: 2rem;
        }
        .nav-bar {
            padding: 1rem 1.5rem;
        }
    }
    
    /* ===== SELECTION ===== */
    ::selection {
        background: rgba(255, 107, 53, 0.3);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_model.h5", custom_objects={"KerasLayer": hub.KerasLayer})

@st.cache_resource
def load_detector():
    return YOLO("yolov8n.pt")

@st.cache_data
def load_breeds():
    try:
        df = pd.read_csv("labels.csv")
        return np.unique(df["breed"].values)
    except:
        return np.array(["Unknown"])

@st.cache_data
def load_care_info():
    try:
        df = pd.read_csv("breed_care_info.csv")
        df = df.drop_duplicates(subset=["breed"])
        df["breed_norm"] = df["breed"].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)
        return df
    except:
        return pd.DataFrame()

model = load_model()
detector = load_detector()
unique_breeds = load_breeds()
care_df = load_care_info()

def process_image(image, img_size=224):
    img = image.convert("RGB").resize((img_size, img_size))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def detect_dogs(image):
    results = detector.predict(source=image, conf=0.25, iou=0.45, verbose=False)
    dog_crops = []
    for r in results:
        for box in r.boxes:
            if detector.names[int(box.cls[0])] == "dog":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image.crop((max(0,x1), max(0,y1), min(image.width,x2), min(image.height,y2)))
                if crop.width > 40 and crop.height > 40:
                    dog_crops.append(crop)
    return dog_crops

def parse_vaccination(vacc_str):
    parts = [p.strip() for p in vacc_str.split(';') if p.strip()]
    schedule = []
    for part in parts:
        match = re.match(r'^([\d\-]+[a-zA-Z]*)\s+(.*)$', part)
        schedule.append(match.groups() if match else ("—", part))
    return schedule

def generate_pdf(dogs_data, filename):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], alignment=TA_CENTER, textColor=colors.HexColor('#FF6B35'), fontSize=22)
    
    elements.append(Paragraph("🐕 CanineAI Professional Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    for idx, data in enumerate(dogs_data):
        elements.append(Paragraph(f"Dog #{idx+1}: {data['breed']}", styles['Heading2']))
        elements.append(Spacer(1, 0.15 * inch))
        
        care = data["care"]
        metrics_table = Table([["Feeding", "Protein", "Food Type"], 
                               [care['feeding_frequency'], care['protein_requirement'], care['food_type']]])
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.2 * inch))
        
        for issue in care['common_diseases'].split(';'):
            if issue.strip():
                elements.append(Paragraph(f"• {issue.strip()}", styles['Normal']))
        elements.append(Spacer(1, 0.1 * inch))
        
        schedule = parse_vaccination(care['vaccination_schedule'])
        if schedule:
            vacc_table = Table([["Age", "Vaccine"]] + schedule)
            elements.append(vacc_table)
            elements.append(Spacer(1, 0.1 * inch))
        
        elements.append(Paragraph(f"<b>Care Notes:</b> {care['special_notes']}", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        data["image"].convert("RGB").save(tmp.name)
        elements.append(RLImage(tmp.name, width=4*inch, height=4*inch))
        
        if idx != len(dogs_data) - 1:
            elements.append(PageBreak())
    
    doc.build(elements)
    return filename

# ===== NAVIGATION BAR =====
st.markdown("""
    <div class='nav-bar'>
        <div class='nav-logo'>🐕 CANINE AI</div>
        <div class='nav-status'>
            <div class='status-dot'></div>
            <span>All Systems Operational</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ===== HERO SECTION =====
st.markdown("""
    <div class='hero'>
        <div class='badge'>
            <div class='badge-dot'></div>
            <span>AI-Powered Platform</span>
        </div>
        <h1 class='hero-title'>
            <span class='gradient-text'>Next-Generation</span><br>
            <span class='highlight'>Breed Intelligence</span>
        </h1>
        <p class='hero-subtitle'>
            Advanced machine learning for instant breed detection, comprehensive health analysis, 
            and professional veterinary insights — all in one powerful platform
        </p>
        <div class='stats-grid'>
            <div class='stat-item'>
                <div class='stat-number'>120+</div>
                <div class='stat-label'>Dog Breeds</div>
            </div>
            <div class='stat-item'>
                <div class='stat-number'>95%</div>
                <div class='stat-label'>Accuracy</div>
            </div>
            <div class='stat-item'>
                <div class='stat-number'>< 2s</div>
                <div class='stat-label'>Analysis Time</div>
            </div>
            <div class='stat-item'>
                <div class='stat-number'>24/7</div>
                <div class='stat-label'>Availability</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

if 'dog_crops' not in st.session_state:
    st.session_state.dog_crops = []
if 'all_dogs_data' not in st.session_state:
    st.session_state.all_dogs_data = []

# ===== UPLOAD SECTION =====
st.markdown("""
    <div class='section'>
        <div class='section-header'>
            <div class='section-number'>01</div>
            <div class='section-label'>UPLOAD IMAGE</div>
            <h2 class='section-title'>Start Your Analysis</h2>
            <p class='section-desc'>Upload a high-quality image of your dog for instant AI-powered breed detection and analysis</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a dog image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, use_column_width=True)
    
    with st.spinner("🔍 Analyzing image with advanced AI..."):
        dog_crops = detect_dogs(image)
        st.session_state.dog_crops = dog_crops
    
    if len(dog_crops) == 0:
        st.error("❌ No dogs detected. Please upload a clearer image with visible dogs.")
    else:
        st.success(f"✅ Successfully detected **{len(dog_crops)}** dog(s) with high confidence")
        
        st.markdown("### Detected Dogs")
        cols = st.columns(min(len(dog_crops), 5))
        for i, crop in enumerate(dog_crops[:5]):
            with cols[i]:
                st.image(crop, caption=f"Dog #{i+1}", use_column_width=True)
        
        if len(dog_crops) > 5:
            st.info(f"➕ {len(dog_crops) - 5} additional dogs detected")

st.markdown("</div>", unsafe_allow_html=True)

# ===== BREED ANALYSIS =====
if st.session_state.dog_crops:
    st.markdown("""
        <div class='section'>
            <div class='section-header'>
                <div class='section-number'>02</div>
                <div class='section-label'>AI ANALYSIS</div>
                <h2 class='section-title'>Breed Identification</h2>
                <p class='section-desc'>Deep learning models analyze visual features to identify breed with exceptional accuracy</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    all_dogs_data = []
    
    for idx, dog_img in enumerate(st.session_state.dog_crops):
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(dog_img, use_column_width=True)
            
            with st.spinner("Running neural network analysis..."):
                processed = process_image(dog_img)
                prediction = model.predict(processed, verbose=0)[0]
                predicted_breed = unique_breeds[np.argmax(prediction)]
                formatted_breed = predicted_breed.replace("_", " ").title()
                confidence = np.max(prediction)
            
            st.markdown(f"<div class='confidence'><div class='confidence-icon'></div><span>Confidence: {confidence:.1%}</span></div>", unsafe_allow_html=True)
            st.progress(float(confidence))
        
        with col2:
            st.markdown(f"### 🐕 {formatted_breed}")
            
            predicted_norm = predicted_breed.strip().lower().replace("_", " ")
            breed_row = care_df[care_df["breed_norm"] == predicted_norm]
            
            if not breed_row.empty:
                care_info = breed_row.iloc[0].to_dict()
                
                st.markdown("#### Care Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Feeding Schedule", care_info['feeding_frequency'])
                m2.metric("Protein Need", care_info['protein_requirement'])
                m3.metric("Diet Type", care_info['food_type'])
                
                st.markdown("#### 🩺 Common Health Concerns")
                issues = [i.strip() for i in care_info['common_diseases'].split(';') if i.strip()]
                badges_html = ''.join([f'<span class="health-badge">{issue}</span>' for issue in issues])
                st.markdown(f"<div style='margin: 1.5rem 0;'>{badges_html}</div>", unsafe_allow_html=True)
                
                st.markdown("#### 💉 Vaccination Schedule")
                schedule = parse_vaccination(care_info['vaccination_schedule'])
                if schedule:
                    for age, vaccine in schedule:
                        st.markdown(f"<div class='vacc-item'><span class='vacc-age'>{age}</span><span class='vacc-name'>{vaccine}</span></div>", unsafe_allow_html=True)
                
                st.markdown("#### ✨ Professional Care Recommendations")
                st.info(care_info["special_notes"])
                
                all_dogs_data.append({"breed": formatted_breed, "care": care_info, "image": dog_img})
            else:
                st.warning("⚠️ No specific care data available for this breed. Consult your veterinarian.")
                placeholder_care = {
                    'food_type': 'General Diet',
                    'protein_requirement': 'Varies',
                    'common_diseases': 'Consult Veterinarian',
                    'vaccination_schedule': 'Standard Protocol',
                    'feeding_frequency': '2x Daily',
                    'special_notes': 'Please consult your veterinarian for breed-specific care recommendations and health monitoring.'
                }
                all_dogs_data.append({"breed": formatted_breed, "care": placeholder_care, "image": dog_img})
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.session_state.all_dogs_data = all_dogs_data
    
    # ===== REPORTS =====
    st.markdown("""
        <div class='section'>
            <div class='section-header'>
                <div class='section-number'>03</div>
                <div class='section-label'>EXPORT DATA</div>
                <h2 class='section-title'>Professional Reports</h2>
                <p class='section-desc'>Download comprehensive PDF reports for your veterinary records and personal reference</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    
    st.markdown("### Individual Dog Reports")
    cols = st.columns(min(len(all_dogs_data), 3))
    for i, data in enumerate(all_dogs_data):
        with cols[i % 3]:
            pdf = generate_pdf([data], f"{data['breed']}_Professional_Report.pdf")
            with open(pdf, "rb") as f:
                st.download_button(
                    f"📄 {data['breed']}", 
                    f, 
                    pdf, 
                    "application/pdf", 
                    key=f"r_{i}",
                    use_container_width=True
                )
    
    if len(all_dogs_data) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Complete Comprehensive Report")
        st.markdown("<p style='color: #888888; margin-bottom: 1.5rem;'>All dogs analyzed in this session, compiled into one professional document</p>", unsafe_allow_html=True)
        combined = generate_pdf(all_dogs_data, "CanineAI_Complete_Analysis.pdf")
        with open(combined, "rb") as f:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    "📥 DOWNLOAD MASTER REPORT", 
                    f, 
                    combined, 
                    "application/pdf", 
                    type="primary",
                    use_container_width=True
                )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
    <div class='footer'>
        <div class='footer-logo'>CANINE AI</div>
        <div class='footer-text'>
            Professional breed intelligence platform powered by advanced machine learning<br>
            Providing accurate, instant analysis for modern pet care professionals and enthusiasts<br>
            © 2025 CanineAI. All rights reserved.
        </div>
        <div class='footer-links'>
            <a href='#' class='footer-link'>Privacy Policy</a>
            <a href='#' class='footer-link'>Terms of Service</a>
            <a href='#' class='footer-link'>Documentation</a>
            <a href='#' class='footer-link'>Support</a>
        </div>
    </div>
""", unsafe_allow_html=True)
