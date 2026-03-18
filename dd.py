import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    PageBreak,
    Table,
    TableStyle
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import os
import re

# ==============================
# 🎨 UI CONFIG – BREATHTAKING & SAFE
# ==============================
st.set_page_config(
    page_title="CanineAI | Breed Intelligence",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Safe, breathtaking CSS – no complex animations, just pure elegance
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }
    
    /* Main background – deep, rich gradient */
    .stApp {
        background: linear-gradient(145deg, #0c1a2b 0%, #1b3a4a 100%);
    }
    
    /* Tabs – sleek glass with vibrant active state */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        border-radius: 50px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 40px !important;
        padding: 10px 28px !important;
        color: #c0d8e8 !important;
        font-weight: 600 !important;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #2a9d8f, #1e7a6c) !important;
        color: white !important;
        box-shadow: 0 8px 20px -8px #2a9d8f;
    }
    
    /* Glass cards – premium with subtle glow */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border-radius: 32px;
        padding: 32px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 25px 50px -20px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.02) inset;
        margin: 24px 0;
        transition: border 0.2s, box-shadow 0.2s;
    }
    
    .glass-card:hover {
        border-color: #2a9d8f;
        box-shadow: 0 30px 60px -20px #2a9d8f, 0 0 0 1px #2a9d8f inset;
    }
    
    /* Metric badges – vibrant and clean */
    .metric-badge {
        background: linear-gradient(145deg, #1e3a47, #132a33);
        border-radius: 24px;
        padding: 20px 16px;
        text-align: center;
        border: 1px solid #2a9d8f;
        backdrop-filter: blur(4px);
        transition: all 0.2s;
        box-shadow: 0 8px 20px -10px black;
    }
    
    .metric-badge:hover {
        transform: translateY(-2px);
        border-color: #e76f51;
        box-shadow: 0 12px 28px -10px #e76f51;
    }
    
    .metric-title {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #a0d0e0;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    /* File uploader – elegant drop zone */
    .stFileUploader > div {
        background: rgba(255,255,255,0.02) !important;
        border: 2px dashed #2a9d8f !important;
        border-radius: 40px !important;
        padding: 50px !important;
        backdrop-filter: blur(8px);
        transition: border 0.2s, background 0.2s;
    }
    
    .stFileUploader > div:hover {
        border-color: #e76f51 !important;
        background: rgba(255,255,255,0.05) !important;
    }
    
    /* Buttons – vibrant gradient with lift */
    .stButton button, .stDownloadButton button {
        background: linear-gradient(145deg, #2a9d8f, #1f7a6c) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.7rem 2.5rem !important;
        box-shadow: 0 15px 30px -10px #1f7a6c !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
    
    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px -8px #2a9d8f !important;
    }
    
    /* Headings – with gradient */
    h1, h2, h3 {
        color: white !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(145deg, #ffffff, #b0e0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Status messages – glowing accent */
    .stAlert {
        background: rgba(42,157,143,0.15) !important;
        backdrop-filter: blur(8px);
        border-left: 6px solid #2a9d8f !important;
        color: white !important;
        border-radius: 24px !important;
        box-shadow: 0 8px 20px -8px black;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        color: #7a9aaa;
        font-size: 0.9rem;
        font-weight: 300;
    }
    
    /* Health tags – vibrant chips */
    .health-tag {
        display: inline-block;
        background: rgba(231, 111, 81, 0.15);
        border: 1px solid #e76f51;
        border-radius: 40px;
        padding: 6px 18px;
        margin: 4px 8px 4px 0;
        color: #ffc8b8;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .health-tag:hover {
        background: rgba(231, 111, 81, 0.25);
        transform: translateY(-1px);
    }
    
    /* Vaccine table – sleek and readable */
    .vaccine-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 20px -10px black;
    }
    .vaccine-table th {
        background: #1e3a47;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 12px 10px;
        text-align: left;
    }
    .vaccine-table td {
        padding: 10px;
        color: #d0e8f0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        background: #0f2630;
    }
    
    /* Special notes card – elegant */
    .special-notes {
        background: rgba(42,157,143,0.1);
        border-left: 6px solid #2a9d8f;
        border-radius: 24px;
        padding: 20px 24px;
        margin-top: 20px;
        color: #d0f0f0;
        font-style: italic;
        box-shadow: 0 8px 20px -10px black;
        transition: border 0.2s, background 0.2s;
    }
    
    .special-notes:hover {
        background: rgba(42,157,143,0.15);
        border-left-color: #e76f51;
    }
    
    /* Horizontal rule */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a9d8f, #e76f51, #2a9d8f, transparent);
        margin: 30px 0;
        opacity: 0.5;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2a9d8f, #e76f51) !important;
        border-radius: 20px !important;
    }
    
    /* Thumbnail images */
    .stImage img {
        border-radius: 24px !important;
        box-shadow: 0 15px 30px -10px black;
        transition: transform 0.2s;
    }
    
    .stImage img:hover {
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# 🛠️ LOAD MODELS & DATA (UNCHANGED)
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "dog_breed_model.h5",
        custom_objects={"KerasLayer": hub.KerasLayer}
    )

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
        df["breed_norm"] = (
            df["breed"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
        )
        return df
    except:
        return pd.DataFrame()

# Load models
model = load_model()
detector = load_detector()
unique_breeds = load_breeds()
care_df = load_care_info()

# ==============================
# 🖼️ IMAGE PROCESSING (UNCHANGED)
# ==============================
def process_image(image, img_size=224):
    img = image.convert("RGB")
    img = img.resize((img_size, img_size))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_dogs(image):
    results = detector.predict(source=image, conf=0.25, iou=0.45, verbose=False)
    dog_crops = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = detector.names[cls_id]
            if label == "dog":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)
                crop = image.crop((x1, y1, x2, y2))
                if crop.width > 40 and crop.height > 40:
                    dog_crops.append(crop)
    return dog_crops

# ==============================
# 📄 PDF GENERATION (ENHANCED)
# ==============================
def parse_vaccination(vacc_str):
    """Convert vaccination string into list of (age, vaccine) pairs."""
    parts = [p.strip() for p in vacc_str.split(';') if p.strip()]
    schedule = []
    for part in parts:
        match = re.match(r'^([\d\-]+[a-zA-Z]*)\s+(.*)$', part)
        if match:
            age, vaccine = match.groups()
        else:
            age, vaccine = "—", part
        schedule.append((age, vaccine))
    return schedule

def generate_pdf(dogs_data, filename):
    doc = SimpleDocTemplate(filename, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        alignment=TA_CENTER, textColor=colors.HexColor('#2a9d8f'), fontSize=20, spaceAfter=20
    )
    section_style = ParagraphStyle(
        'Section', parent=styles['Heading2'],
        textColor=colors.HexColor('#2a9d8f'), fontSize=16, spaceAfter=10
    )
    normal_style = styles["Normal"]
    
    elements.append(Paragraph("🐕 Dog Breed Intelligence Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    for idx, data in enumerate(dogs_data):
        breed = data["breed"]
        care_info = data["care"]
        image = data["image"]
        
        elements.append(Paragraph(f"Dog #{idx+1}: {breed}", section_style))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Metrics table
        metrics_data = [
            ["Feeding", "Protein", "Food Type"],
            [care_info['feeding_frequency'], care_info['protein_requirement'], care_info['food_type']]
        ]
        metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3b4a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0f2a35')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#b0d0e0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2a9d8f80')),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Health issues
        issues = [i.strip() for i in care_info['common_diseases'].split(';') if i.strip()]
        elements.append(Paragraph("<b>Common health issues:</b>", normal_style))
        for issue in issues:
            elements.append(Paragraph(f"• {issue}", normal_style))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Vaccination schedule
        schedule = parse_vaccination(care_info['vaccination_schedule'])
        if schedule:
            vacc_data = [["Age", "Vaccine"]] + list(schedule)
            vacc_table = Table(vacc_data, colWidths=[2*inch, 4*inch])
            vacc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3b4a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0f2a35')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#b0d0e0')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2a9d8f80')),
            ]))
            elements.append(Paragraph("<b>Vaccination schedule:</b>", normal_style))
            elements.append(vacc_table)
            elements.append(Spacer(1, 0.1 * inch))
        
        # Special notes
        elements.append(Paragraph("<b>Special notes:</b>", normal_style))
        note_data = [[care_info['special_notes']]]
        note_table = Table(note_data, colWidths=[6*inch])
        note_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1a3a45')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#d0f0f0')),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#2a9d8f')),
        ]))
        elements.append(note_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Dog image
        image_rgb = image.convert("RGB")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_rgb.save(tmp.name)
        elements.append(RLImage(tmp.name, width=4*inch, height=4*inch))
        
        if idx != len(dogs_data) - 1:
            elements.append(PageBreak())
    
    doc.build(elements)
    return filename

# ==============================
# 🚀 MAIN UI – TABS & LAYOUT (UNCHANGED)
# ==============================

st.markdown("<h1 class='main-title'>🐕 CanineAI · Breed Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #b0d0e0; font-size: 1.2rem; margin-top: -10px;'>Multi‑dog detection · Breed analysis · Health reporting</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📤 UPLOAD & DETECT", "🔬 BREED INSIGHTS", "📄 REPORTS"])

# Session state
if 'dog_crops' not in st.session_state:
    st.session_state.dog_crops = []
if 'all_dogs_data' not in st.session_state:
    st.session_state.all_dogs_data = []
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# ========== TAB 1 ==========
with tab1:
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = image
            with st.spinner("🔍 Scanning for dogs..."):
                dog_crops = detect_dogs(image)
                st.session_state.dog_crops = dog_crops
            if len(dog_crops) == 0:
                st.error("❌ No dogs detected. Please try another image.")
            else:
                st.success(f"✅ Found {len(dog_crops)} dog(s)!")
                st.markdown("##### Detected Dogs")
                thumb_cols = st.columns(min(len(dog_crops), 5))
                for i, crop in enumerate(dog_crops[:5]):
                    with thumb_cols[i]:
                        st.image(crop.resize((150,150)), caption=f"Dog #{i+1}")
        st.markdown("</div>", unsafe_allow_html=True)

# ========== TAB 2 ==========
with tab2:
    if not st.session_state.dog_crops:
        st.info("👆 Please upload an image with dogs in the 'UPLOAD & DETECT' tab first.")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🧬 Breed Analysis & Care Data")
        all_dogs_data = []
        for idx, dog_img in enumerate(st.session_state.dog_crops):
            st.markdown(f"<hr style='border-color: #2a9d8f40;'>", unsafe_allow_html=True)
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(dog_img, use_column_width=True)
                processed = process_image(dog_img)
                prediction = model.predict(processed, verbose=0)[0]
                top_idx = np.argmax(prediction)
                predicted_breed = unique_breeds[top_idx]
                formatted_breed = predicted_breed.replace("_", " ").title()
                confidence = np.max(prediction)
                st.markdown(f"**Confidence:** `{confidence:.1%}`")
                st.progress(float(confidence))
            with col_info:
                st.markdown(f"#### 🐕 {formatted_breed}")
                predicted_norm = predicted_breed.strip().lower().replace("_", " ")
                breed_row = care_df[care_df["breed_norm"] == predicted_norm]
                if not breed_row.empty:
                    care_info = breed_row.iloc[0].to_dict()
                    m1, m2, m3 = st.columns(3)
                    with m1: 
                        st.markdown(f"<div class='metric-badge'><div class='metric-title'>Feeding</div><div class='metric-value'>{care_info['feeding_frequency']}</div></div>", unsafe_allow_html=True)
                    with m2: 
                        st.markdown(f"<div class='metric-badge'><div class='metric-title'>Protein</div><div class='metric-value'>{care_info['protein_requirement']}</div></div>", unsafe_allow_html=True)
                    with m3: 
                        st.markdown(f"<div class='metric-badge'><div class='metric-title'>Food Type</div><div class='metric-value'>{care_info['food_type']}</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**🩺 Common health issues:**")
                    issues = [i.strip() for i in care_info['common_diseases'].split(';') if i.strip()]
                    tags_html = ''.join([f'<span class="health-tag">{issue}</span>' for issue in issues])
                    st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
                    
                    st.markdown("**💉 Vaccination schedule:**")
                    schedule = parse_vaccination(care_info['vaccination_schedule'])
                    if schedule:
                        table_html = '<table class="vaccine-table"><tr><th>Age</th><th>Vaccine</th></tr>'
                        for age, vaccine in schedule:
                            table_html += f'<tr><td>{age}</td><td>{vaccine}</td></tr>'
                        table_html += '</table>'
                        st.markdown(table_html, unsafe_allow_html=True)
                    
                    st.markdown("**💡 Special notes:**")
                    st.markdown(f'<div class="special-notes">{care_info["special_notes"]}</div>', unsafe_allow_html=True)
                    
                    all_dogs_data.append({"breed": formatted_breed, "care": care_info, "image": dog_img})
                else:
                    st.warning("No specific care data for this breed.")
                    placeholder_care = {
                        'food_type': 'General diet',
                        'protein_requirement': 'Varies',
                        'common_diseases': 'Consult vet',
                        'vaccination_schedule': 'Standard protocol',
                        'feeding_frequency': '2x daily',
                        'special_notes': 'No specific data'
                    }
                    all_dogs_data.append({"breed": formatted_breed, "care": placeholder_care, "image": dog_img})
        st.session_state.all_dogs_data = all_dogs_data
        st.markdown("</div>", unsafe_allow_html=True)

# ========== TAB 3 ==========
with tab3:
    if not st.session_state.all_dogs_data:
        st.info("👆 Please analyze dogs in the 'BREED INSIGHTS' tab first.")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📄 Generate Intelligence Reports")
        st.markdown("#### Individual Reports")
        cols = st.columns(len(st.session_state.all_dogs_data))
        for i, data in enumerate(st.session_state.all_dogs_data):
            with cols[i]:
                breed_name = data['breed']
                single_pdf = generate_pdf([data], f"{breed_name}_Report.pdf")
                with open(single_pdf, "rb") as f:
                    st.download_button(
                        label=f"🐕 {breed_name}",
                        data=f,
                        file_name=single_pdf,
                        mime="application/pdf",
                        key=f"report_{i}",
                        use_container_width=True
                    )
        if len(st.session_state.all_dogs_data) > 1:
            st.markdown("#### Combined Master Report")
            combined_pdf = generate_pdf(st.session_state.all_dogs_data, "Comprehensive_Canine_Report.pdf")
            with open(combined_pdf, "rb") as f:
                st.download_button(
                    label="🏆 DOWNLOAD COMPREHENSIVE REPORT",
                    data=f,
                    file_name=combined_pdf,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 CanineAI – Precision breed analysis for modern pet care</div>", unsafe_allow_html=True)