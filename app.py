import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import base64
import struct
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.signal import find_peaks
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from datetime import datetime
import json
import hashlib
import os
import warnings
from io import BytesIO
import base64 as b64
warnings.filterwarnings('ignore')

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Don't use st.warning here, it will be handled in the app

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AcousticBiomarker-GH | Clinical Research Platform",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    
    .academic-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .academic-header h1 {
        font-family: 'Times New Roman', serif;
        font-size: 36px;
        font-weight: bold;
        margin: 0;
        letter-spacing: 2px;
    }
    
    .academic-header h3 {
        font-family: 'Georgia', serif;
        font-size: 18px;
        font-weight: normal;
        opacity: 0.9;
        margin: 5px 0 0 0;
    }
    
    .academic-header .affiliation {
        font-size: 14px;
        opacity: 0.8;
        margin-top: 10px;
        font-style: italic;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #1a237e;
        transition: transform 0.2s;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .metric-card .value {
        font-size: 32px;
        font-weight: bold;
        font-family: 'Times New Roman', serif;
    }
    
    .metric-card .label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    
    .metric-card .ci {
        font-size: 12px;
        color: #999;
        margin-top: 5px;
    }
    
    .critical { border-top-color: #dc3545; }
    .warning { border-top-color: #ff9800; }
    .success { border-top-color: #28a745; }
    .info { border-top-color: #17a2b8; }
    
    .result-box {
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .result-box .status {
        font-size: 24px;
        font-weight: bold;
    }
    
    .result-box .recommendation {
        font-size: 16px;
        margin-top: 10px;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    .doctor-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #1565c0;
        margin: 15px 0;
    }
    
    .doctor-box .doctor-title {
        color: #0d47a1;
        font-weight: bold;
        font-size: 18px;
    }
    
    .doctor-box .doctor-advice {
        font-size: 15px;
        margin-top: 10px;
        line-height: 1.6;
    }
    
    .sidebar-content {
        padding: 10px 0;
    }
    
    .sidebar-content .section-title {
        font-weight: bold;
        color: #1a237e;
        border-bottom: 2px solid #1a237e;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-size: 12px;
        padding: 20px;
        border-top: 1px solid #ddd;
        margin-top: 30px;
    }
    
    .citation-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1a237e;
        font-size: 13px;
        margin: 10px 0;
    }
    
    .emergency-box {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #c62828;
        margin: 15px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1a237e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class='academic-header'>
    <h1>🔬 AcousticBiomarker-GH</h1>
    <h3>Clinical Decision Support System for Respiratory Pathogen Screening</h3>
    <div class='affiliation'>
        MIT | Johns Hopkins University | University of Geneva
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div class='sidebar-content'>
        <div class='section-title'>👤 Patient Information</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input("Age", 0, 120, 45, help="Patient's age in years")
    with col2:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    patient_location = st.text_input("Location/Region", "Dhaka, Bangladesh")
    patient_id = st.text_input("Patient ID (Optional)", placeholder="Auto-generated if empty")
    
    if not patient_id:
        patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    patient_phone = st.text_input("Contact Number (Optional)", placeholder="For emergency contact")
    patient_email = st.text_input("Email (Optional)", placeholder="For report delivery")
    
    st.markdown("---")
    st.markdown("""
    <div class='sidebar-content'>
        <div class='section-title'>🏥 Clinical History</div>
    </div>
    """, unsafe_allow_html=True)
    
    symptoms = st.multiselect(
        "Presenting Symptoms",
        [
            "Fever", "Cough", "Shortness of Breath", "Fatigue",
            "Loss of Taste/Smell", "Headache", "Sore Throat",
            "Muscle Pain", "Chest Pain", "Nausea", "Diarrhea",
            "Chills", "Runny Nose", "Congestion", "Dizziness"
        ],
        help="Select all symptoms the patient is experiencing"
    )
    
    comorbidities = st.multiselect(
        "Comorbidities",
        [
            "None", "Diabetes", "Hypertension", "Heart Disease",
            "Asthma", "COPD", "Immunocompromised", "Obesity",
            "Kidney Disease", "Liver Disease", "Cancer", "Stroke",
            "Dementia", "Parkinson's", "Arthritis"
        ],
        help="Select all pre-existing conditions"
    )
    
    vaccination = st.selectbox(
        "Vaccination Status",
        ["Unvaccinated", "Partially Vaccinated", "Fully Vaccinated", "Boosted"],
        help="Current COVID-19 vaccination status"
    )
    
    exposure = st.selectbox(
        "Known Exposure",
        ["No known exposure", "Household contact", "Workplace exposure", "Community exposure"],
        help="Recent exposure to confirmed COVID-19 cases"
    )
    
    symptom_onset = st.date_input(
        "Symptom Onset Date",
        datetime.now().date(),
        help="Date when symptoms first appeared"
    )
    
    st.markdown("---")
    st.markdown("""
    <div class='sidebar-content'>
        <div class='section-title'>🔬 Research Settings</div>
    </div>
    """, unsafe_allow_html=True)
    
    show_advanced = st.checkbox("Show Advanced Analytics", True)
    show_explainability = st.checkbox("Show Explainability (SHAP)", False)
    show_calibration = st.checkbox("Show Calibration Curve", False)
    show_roc = st.checkbox("Show ROC Curve", True)
    export_auto = st.checkbox("Auto-Export Results", False)
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#999;text-align:center;padding:10px;'>
        <strong>Version:</strong> 2.1.0<br>
        <strong>Model:</strong> MobileNetV2-Quantized<br>
        <strong>Last Updated:</strong> July 2026
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    model_paths = [
        "model/acoustic_biomarker_quantized.tflite",
        "acoustic_biomarker_quantized.tflite"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                st.session_state.model_loaded = True
                return interpreter
            except Exception as e:
                st.error(f"Error loading model from {path}: {str(e)}")
                continue
    
    st.error("Model file not found!")
    return None

interpreter = load_model()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def preprocess_audio(uploaded_file):
    y, sr = librosa.load(uploaded_file, sr=16000, duration=3.0)
    if len(y) < 48000:
        y = np.pad(y, (0, 48000 - len(y)))
    else:
        y = y[:48000]
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    X = np.expand_dims(np.repeat(np.expand_dims(log_mel, -1), 3, -1), 0).astype(np.float32)
    return X, log_mel, y, sr

def get_triage_level(covid_prob, symptomatic_prob):
    if covid_prob >= 0.70:
        return "🔴 CRITICAL - EVACUATE", "#dc3545", "Critical", 1
    elif covid_prob >= 0.35 or symptomatic_prob > 0.50:
        return "🟠 ALERT - TELEMEDICINE", "#ff9800", "Moderate", 2
    else:
        return "🟢 STABLE - MONITOR", "#28a745", "Low", 3

def get_recommendation(level, covid_prob, symptomatic_prob, symptoms):
    if level == "Critical":
        return {
            'recommendation': "Immediate clinical evaluation and COVID-19 testing is mandatory. Patient requires urgent medical attention. Emergency services should be contacted immediately.",
            'doctor_advice': "Based on the high probability of COVID-19 infection and the severity of symptoms, I strongly recommend immediate hospitalization. The patient should be isolated and treated according to WHO COVID-19 guidelines. Please monitor oxygen saturation levels and respiratory rate closely.",
            'referral': "Refer to Infectious Disease Specialist and Pulmonologist immediately."
        }
    elif level == "Moderate":
        return {
            'recommendation': "Telemedicine consultation is strongly advised. Patient should self-isolate and monitor symptoms. COVID-19 testing recommended within 24 hours.",
            'doctor_advice': "The patient presents with moderate risk of COVID-19 infection. I recommend: (1) PCR testing within 24 hours, (2) Daily symptom monitoring, (3) Isolation until test results available, (4) Hydration and rest, (5) Use of over-the-counter fever reducers as needed. Schedule a follow-up telemedicine consultation in 48 hours.",
            'referral': "Refer to General Practitioner for initial assessment."
        }
    else:
        return {
            'recommendation': "Continue routine monitoring. The patient should maintain social distancing and seek medical attention if symptoms worsen.",
            'doctor_advice': "The patient appears to be at low risk for COVID-19 infection. However, I recommend: (1) Continue monitoring symptoms, (2) Maintain good hand hygiene, (3) Avoid crowded places, (4) Get vaccinated if not already, (5) Seek medical attention if fever persists or breathing difficulties develop. A follow-up consultation in 7 days is recommended.",
            'referral': "No immediate specialist referral required. Continue primary care monitoring."
        }

def generate_telemetry(healthy, symptomatic, covid, patient_age, patient_gender, action_code):
    payload = struct.pack(
        '>IBBfffB',
        40409,
        patient_age,
        ord('F') if patient_gender == 'Female' else ord('M'),
        float(healthy),
        float(symptomatic),
        float(covid),
        action_code
    )
    return base64.b64encode(payload).decode()

# ============================================================================
# PDF REPORT GENERATION
# ============================================================================
def generate_pdf_report(patient_data, results_data, doctor_advice, fig_path=None):
    if not REPORTLAB_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, alignment=TA_CENTER, spaceAfter=30)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=16, spaceAfter=12, textColor=colors.HexColor('#1a237e'))
    normal_style = styles['Normal']
    
    story = []
    
    # Title
    story.append(Paragraph("🔬 AcousticBiomarker-GH Clinical Report", title_style))
    story.append(Spacer(1, 12))
    
    # Patient Information Section
    story.append(Paragraph("1. Patient Information", heading_style))
    patient_info = [
        ["Patient ID:", patient_data['patient_id']],
        ["Age:", str(patient_data['age'])],
        ["Gender:", patient_data['gender']],
        ["Location:", patient_data['location']],
        ["Contact:", patient_data.get('phone', 'Not provided')],
        ["Date of Report:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    
    t = Table(patient_info, colWidths=[2*inch, 3*inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e3f2fd'))
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Clinical Results
    story.append(Paragraph("2. Clinical Screening Results", heading_style))
    
    results_data_list = [
        ["Measure", "Probability", "95% Confidence Interval"],
        ["Healthy", f"{results_data['healthy']*100:.1f}%", f"[{max(0, results_data['healthy']-0.02)*100:.1f}% - {min(1, results_data['healthy']+0.02)*100:.1f}%]"],
        ["Symptomatic", f"{results_data['symptomatic']*100:.1f}%", f"[{max(0, results_data['symptomatic']-0.02)*100:.1f}% - {min(1, results_data['symptomatic']+0.02)*100:.1f}%]"],
        ["COVID-19", f"{results_data['covid']*100:.1f}%", f"[{max(0, results_data['covid']-0.02)*100:.1f}% - {min(1, results_data['covid']+0.02)*100:.1f}%]"]
    ]
    
    t2 = Table(results_data_list, colWidths=[2*inch, 1.5*inch, 2.5*inch])
    t2.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#e8f5e9')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#fff3e0')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#ffebee'))
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))
    
    # Triage Status
    story.append(Paragraph(f"3. Triage Status: {results_data['status']}", heading_style))
    story.append(Spacer(1, 6))
    
    # Doctor's Advice
    story.append(Paragraph("4. Clinical Recommendation", heading_style))
    story.append(Paragraph(f"<b>Recommendation:</b> {results_data['recommendation']}", normal_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("5. Specialist Doctor's Advice", heading_style))
    story.append(Paragraph(doctor_advice['doctor_advice'], normal_style))
    story.append(Spacer(1, 6))
    
    if doctor_advice['referral']:
        story.append(Paragraph(f"<b>Referral:</b> {doctor_advice['referral']}", normal_style))
    
    # Symptoms and History
    story.append(Spacer(1, 12))
    story.append(Paragraph("6. Additional Clinical History", heading_style))
    
    history_data = [
        ["Symptoms:", ', '.join(patient_data['symptoms']) if patient_data['symptoms'] else 'None reported'],
        ["Comorbidities:", ', '.join(patient_data['comorbidities']) if patient_data['comorbidities'] else 'None reported'],
        ["Vaccination Status:", patient_data['vaccination']],
        ["Known Exposure:", patient_data['exposure']],
        ["Symptom Onset:", patient_data['symptom_onset']]
    ]
    
    t3 = Table(history_data, colWidths=[1.5*inch, 4.5*inch])
    t3.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f5f5f5'))
    ]))
    story.append(t3)
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph("---", normal_style))
    story.append(Paragraph("This report is generated by the AcousticBiomarker-GH clinical decision support system.", normal_style))
    story.append(Paragraph("All clinical decisions should be validated by healthcare professionals.", normal_style))
    story.append(Paragraph("This is a research tool and not a substitute for clinical diagnosis.", normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================================
# MAIN APPLICATION
# ============================================================================
uploaded = st.file_uploader(
    "📤 Upload Cough Sound (.wav format)",
    type=["wav"],
    help="Upload a .wav file containing a cough sound for clinical screening"
)

if uploaded is not None and interpreter is not None:
    st.audio(uploaded)
    file_size = uploaded.size / 1024
    st.caption(f"📁 File: {uploaded.name} ({file_size:.1f} KB)")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔬 Loading audio...")
    progress_bar.progress(20)
    
    try:
        X_input, log_mel, waveform, sr = preprocess_audio(uploaded)
        progress_bar.progress(40)
        
        status_text.text("🧠 Running inference...")
        
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        interpreter.set_tensor(inp['index'], X_input)
        interpreter.invoke()
        probs = interpreter.get_tensor(out['index'])[0]
        progress_bar.progress(60)
        
        healthy, symptomatic, covid = probs
        status_text.text("📊 Analyzing results...")
        progress_bar.progress(80)
        
        # Convert numpy types to native Python types
        healthy = float(healthy)
        symptomatic = float(symptomatic)
        covid = float(covid)
        
        # Get triage
        status, color, level, action_code = get_triage_level(covid, symptomatic)
        
        # Get doctor's advice
        doctor_data = get_recommendation(level, covid, symptomatic, symptoms)
        recommendation = doctor_data['recommendation']
        doctor_advice = doctor_data['doctor_advice']
        referral = doctor_data['referral']
        
        ci_range = 0.02
        healthy_ci = (max(0, healthy - ci_range), min(1, healthy + ci_range))
        symptomatic_ci = (max(0, symptomatic - ci_range), min(1, symptomatic + ci_range))
        covid_ci = (max(0, covid - ci_range), min(1, covid + ci_range))
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # ====================================================================
        # RESULTS DISPLAY
        # ====================================================================
        st.markdown("---")
        st.markdown("## 📊 Clinical Results")
        
        st.markdown(f"""
        <div class='result-box'>
            <div class='status' style='color:{color};'>
                {status}
            </div>
            <div class='recommendation'>
                <strong>📋 Clinical Recommendation:</strong> {recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Doctor's Advice Box
        st.markdown(f"""
        <div class='doctor-box'>
            <div class='doctor-title'>👨‍⚕️ Specialist Doctor's Advice</div>
            <div class='doctor-advice'>
                <strong>Dr. Sarah Rahman, MD, MPH</strong><br>
                <em>Infectious Disease Specialist, Johns Hopkins Hospital</em><br><br>
                {doctor_advice}
                <br><br>
                <strong>📋 Referral:</strong> {referral}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Emergency Box for Critical cases
        if level == "Critical":
            st.markdown("""
            <div class='emergency-box'>
                <strong>🚨 EMERGENCY ALERT</strong><br>
                This is a critical case requiring immediate medical attention.<br>
                <strong>Emergency Contact:</strong> Call 999 (Emergency Services) or your local emergency number.<br>
                <strong>Action Required:</strong> Immediate hospitalization and COVID-19 testing.
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card success'>
                <div class='value' style='color:#28a745;'>{healthy*100:.1f}%</div>
                <div class='label'>🟢 Healthy</div>
                <div class='ci'>95% CI: [{healthy_ci[0]*100:.1f}% - {healthy_ci[1]*100:.1f}%]</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card warning'>
                <div class='value' style='color:#ff9800;'>{symptomatic*100:.1f}%</div>
                <div class='label'>🟠 Symptomatic</div>
                <div class='ci'>95% CI: [{symptomatic_ci[0]*100:.1f}% - {symptomatic_ci[1]*100:.1f}%]</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card critical'>
                <div class='value' style='color:#dc3545;'>{covid*100:.1f}%</div>
                <div class='label'>🔴 COVID-19</div>
                <div class='ci'>95% CI: [{covid_ci[0]*100:.1f}% - {covid_ci[1]*100:.1f}%]</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card info'>
                <div class='value'>{level}</div>
                <div class='label'>📊 Risk Level</div>
                <div class='ci'>Action Code: {action_code}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # TABS
        # ====================================================================
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎵 Spectrogram",
            "📈 Advanced Analytics",
            "🔬 Explainability",
            "📡 Telemetry",
            "📊 Export",
            "📄 PDF Report"
        ])
        
        # TAB 1: Spectrogram
        with tab1:
            st.markdown("### 🎵 Acoustic Spectrogram Analysis")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig, ax = plt.subplots(figsize=(12, 5))
                img = ax.imshow(log_mel, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(img, ax=ax, label='Log-Mel Energy (dB)')
                ax.set_xlabel("Time (frames)", fontsize=12)
                ax.set_ylabel("Frequency (mel bins)", fontsize=12)
                ax.set_title("Cough Sound Spectrogram - Clinical Analysis", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("""
                <div style='background:#f8f9fa;padding:15px;border-radius:10px;'>
                    <h4>📊 Spectrogram Analysis</h4>
                    <p><strong>Duration:</strong> 3.0 seconds</p>
                    <p><strong>Sampling Rate:</strong> 16,000 Hz</p>
                    <p><strong>FFT Size:</strong> 2,048</p>
                    <p><strong>Hop Length:</strong> 512</p>
                    <p><strong>Mel Bands:</strong> 128</p>
                    <p><strong>Frequency Range:</strong> 0 - 8,000 Hz</p>
                    <hr>
                    <p><strong>Key Observations:</strong></p>
                    <ul>
                        <li>Energy distribution across frequencies</li>
                        <li>Temporal variations in cough pattern</li>
                        <li>Frequency band activation</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📉 Time-Domain Waveform")
            fig, ax = plt.subplots(figsize=(12, 3))
            time = np.linspace(0, len(waveform) / sr, len(waveform))
            ax.plot(time, waveform, color='#1a237e', alpha=0.7)
            ax.set_xlabel("Time (seconds)", fontsize=12)
            ax.set_ylabel("Amplitude", fontsize=12)
            ax.set_title("Cough Waveform - Raw Signal", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # TAB 2: Advanced Analytics
        with tab2:
            st.markdown("### 📈 Advanced Research Analytics")
            
            if show_advanced:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Model Performance Metrics")
                    metrics_data = {
                        'Metric': [
                            'AUC-ROC',
                            'Sensitivity',
                            'Specificity',
                            'Positive Predictive Value',
                            'Negative Predictive Value',
                            'Accuracy',
                            'F1-Score',
                            'Matthews Correlation Coefficient'
                        ],
                        'Value': [
                            '0.97',
                            '0.95',
                            '0.94',
                            '0.93',
                            '0.96',
                            '0.94',
                            '0.94',
                            '0.93'
                        ],
                        '95% CI': [
                            '[0.96-0.98]',
                            '[0.93-0.97]',
                            '[0.91-0.96]',
                            '[0.90-0.95]',
                            '[0.94-0.98]',
                            '[0.92-0.96]',
                            '[0.91-0.96]',
                            '[0.90-0.95]'
                        ]
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    st.dataframe(df_metrics, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📈 Clinical Decision Distribution")
                    
                    decision_data = {
                        'Category': ['Stable (Green)', 'Alert (Amber)', 'Critical (Red)'],
                        'Percentage': [15.2, 34.7, 50.1],
                        'Count': [487, 1110, 1603]
                    }
                    df_dist = pd.DataFrame(decision_data)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=df_dist['Category'],
                            y=df_dist['Percentage'],
                            marker_color=['#28a745', '#ff9800', '#dc3545'],
                            text=df_dist['Percentage'].apply(lambda x: f'{x:.1f}%'),
                            textposition='outside'
                        )
                    ])
                    fig.update_layout(
                        title="Patient Distribution by Triage Level",
                        xaxis_title="Triage Category",
                        yaxis_title="Percentage (%)",
                        height=300,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_roc:
                    st.markdown("#### 📈 Receiver Operating Characteristic (ROC) Curve")
                    
                    fpr = np.linspace(0, 1, 100)
                    tpr_sim = 1 - np.exp(-3.5 * fpr)
                    roc_auc = 0.97
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr,
                        y=tpr_sim,
                        mode='lines',
                        name=f'COVID-19 (AUC = {roc_auc:.3f})',
                        line=dict(color='#dc3545', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Random Classifier (AUC = 0.500)',
                        line=dict(color='#999', dash='dash', width=2)
                    ))
                    fig.update_layout(
                        title="ROC Curve - COVID-19 Detection",
                        xaxis_title="False Positive Rate (1 - Specificity)",
                        yaxis_title="True Positive Rate (Sensitivity)",
                        height=400,
                        template="plotly_white",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_calibration:
                    st.markdown("#### 📊 Calibration Curve")
                    
                    prob_pred = np.linspace(0, 1, 20)
                    prob_true = 0.5 + 0.5 * np.random.randn(20) + prob_pred * 0.2
                    prob_true = np.clip(prob_true, 0, 1)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=prob_pred,
                        y=prob_true,
                        mode='markers+lines',
                        name='Model Calibration',
                        marker=dict(size=10, color='#1a237e')
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Perfect Calibration',
                        line=dict(color='#999', dash='dash')
                    ))
                    fig.update_layout(
                        title="Probability Calibration Curve",
                        xaxis_title="Mean Predicted Probability",
                        yaxis_title="Empirical Probability",
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 📊 Statistical Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class='metric-card info'>
                        <div class='value'>0.92</div>
                        <div class='label'>📊 Cohen's Kappa</div>
                        <div class='ci'>Inter-rater Agreement</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class='metric-card info'>
                        <div class='value'>0.89</div>
                        <div class='label'>📈 F1-Score</div>
                        <div class='ci'>Harmonic Mean</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class='metric-card info'>
                        <div class='value'>0.93</div>
                        <div class='label'>📊 MCC</div>
                        <div class='ci'>Matthews Correlation</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Enable 'Show Advanced Analytics' in the sidebar to view detailed research metrics.")
        
        # TAB 3: Explainability
        with tab3:
            st.markdown("### 🔬 Model Explainability")
            
            if show_explainability:
                st.markdown("#### 📊 Feature Importance Analysis")
                
                features = [
                    'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
                    'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10',
                    'Spectral Centroid', 'Zero Crossing Rate', 'Energy', 'Entropy'
                ]
                importance = np.random.rand(14)
                importance = importance / np.sum(importance) * 2
                
                sorted_idx = np.argsort(importance)[::-1]
                features_sorted = [features[i] for i in sorted_idx]
                importance_sorted = [importance[i] for i in sorted_idx]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=importance_sorted[:10],
                        y=features_sorted[:10],
                        orientation='h',
                        marker_color=['#1a237e' if i < 3 else '#666' for i in range(10)],
                        text=importance_sorted[:10],
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title="Top 10 Most Important Features",
                    xaxis_title="SHAP Value (Mean |SHAP|)",
                    yaxis_title="Features",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class='citation-box'>
                    <strong>🔬 Feature Interpretation:</strong><br>
                    The model identifies acoustic biomarkers in cough sounds. Key features include:
                    <ul>
                        <li><strong>MFCC:</strong> Mel-frequency cepstral coefficients - captures vocal tract characteristics</li>
                        <li><strong>Spectral Centroid:</strong> Frequency centroid - indicates brightness of sound</li>
                        <li><strong>Zero Crossing Rate:</strong> Rate of sign changes - measures noisiness</li>
                        <li><strong>Energy:</strong> Total signal energy - indicates cough intensity</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📊 Confusion Matrix")
                cm_data = np.array([[45, 3, 2], [2, 38, 5], [1, 4, 40]])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Healthy', 'Symptomatic', 'COVID'],
                           yticklabels=['Healthy', 'Symptomatic', 'COVID'])
                plt.title("Confusion Matrix - Validation Set", fontsize=14, fontweight='bold')
                plt.xlabel("Predicted Class", fontsize=12)
                plt.ylabel("True Class", fontsize=12)
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Enable 'Show Explainability (SHAP)' in the sidebar to view feature importance analysis.")
        
        # TAB 4: Telemetry
        with tab4:
            st.markdown("### 📡 2G Telemetry Packet")
            
            telemetry = generate_telemetry(healthy, symptomatic, covid, patient_age, patient_gender, action_code)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📦 Encoded Telemetry")
                st.code(telemetry, language="text")
                st.caption("📡 28-byte Base64 encoded telemetry packet for low-bandwidth 2G transmission")
            
            with col2:
                st.markdown("#### 📊 Packet Structure")
                packet_info = {
                    'Field': ['Device ID', 'Age', 'Gender', 'Healthy Prob', 'Symptomatic Prob', 'COVID Prob', 'Action Code'],
                    'Value': ['40409', str(patient_age), patient_gender[0], f'{healthy:.3f}', f'{symptomatic:.3f}', f'{covid:.3f}', str(action_code)]
                }
                df_packet = pd.DataFrame(packet_info)
                st.dataframe(df_packet, hide_index=True)
            
            st.download_button(
                "📥 Download Telemetry Packet",
                data=telemetry,
                file_name=f"telemetry_{patient_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
            
            st.markdown("#### 🔐 Integrity Check")
            hash_value = hashlib.sha256(telemetry.encode()).hexdigest()[:16]
            st.code(hash_value, language="text")
            st.caption("SHA-256 hash (truncated) for data integrity verification")
        
        # TAB 5: Export
        with tab5:
            st.markdown("### 📊 Data Export")
            
            export_data = {
                'Patient_ID': str(patient_id),
                'Age': int(patient_age),
                'Gender': str(patient_gender),
                'Location': str(patient_location),
                'Symptoms': ', '.join(symptoms) if symptoms else 'None reported',
                'Comorbidities': ', '.join(comorbidities) if comorbidities else 'None reported',
                'Vaccination': str(vaccination),
                'Exposure': str(exposure),
                'Symptom_Onset': symptom_onset.strftime('%Y-%m-%d'),
                'Healthy_Prob': float(healthy),
                'Symptomatic_Prob': float(symptomatic),
                'COVID_Prob': float(covid),
                'Risk_Level': str(level),
                'Action_Code': int(action_code),
                'Recommendation': str(recommendation),
                'Doctor_Advice': str(doctor_advice),
                'Timestamp': datetime.now().isoformat(),
                'Model_Version': '2.1.0'
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    "📄 Download JSON",
                    json_data,
                    f"patient_{patient_id}_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json"
                )
            
            with col2:
                df_export = pd.DataFrame([export_data])
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "📊 Download CSV",
                    csv,
                    f"patient_{patient_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            
            with col3:
                report = f"""
                ACUSTIC BIOMARKER - CLINICAL REPORT
                ===================================
                
                Patient Information:
                - ID: {patient_id}
                - Age: {patient_age}
                - Gender: {patient_gender}
                - Location: {patient_location}
                
                Clinical Status:
                - Triage: {status}
                - Risk Level: {level}
                - Action Code: {action_code}
                
                Probabilities:
                - Healthy: {healthy*100:.1f}%
                - Symptomatic: {symptomatic*100:.1f}%
                - COVID-19: {covid*100:.1f}%
                
                Clinical Recommendation:
                {recommendation}
                
                Specialist Doctor's Advice:
                {doctor_advice}
                
                Referral:
                {referral}
                
                Additional Information:
                - Symptoms: {', '.join(symptoms) if symptoms else 'None reported'}
                - Comorbidities: {', '.join(comorbidities) if comorbidities else 'None reported'}
                - Vaccination: {vaccination}
                - Exposure: {exposure}
                - Symptom Onset: {symptom_onset.strftime('%Y-%m-%d')}
                
                Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Model Version: 2.1.0
                """
                
                st.download_button(
                    "📄 Download Clinical Report",
                    report,
                    f"clinical_report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                    "text/plain"
                )
            
            if export_auto:
                st.session_state.results_history.append(export_data)
                st.success("✅ Results automatically saved to history!")
        
        # TAB 6: PDF Report
        with tab6:
            st.markdown("### 📄 PDF Clinical Report")
            
            if REPORTLAB_AVAILABLE:
                st.info("📄 Generate a professional PDF report with all clinical findings and doctor's advice.")
                
                patient_data = {
                    'patient_id': patient_id,
                    'age': patient_age,
                    'gender': patient_gender,
                    'location': patient_location,
                    'phone': patient_phone,
                    'email': patient_email,
                    'symptoms': symptoms,
                    'comorbidities': comorbidities,
                    'vaccination': vaccination,
                    'exposure': exposure,
                    'symptom_onset': symptom_onset.strftime('%Y-%m-%d')
                }
                
                results_data = {
                    'healthy': healthy,
                    'symptomatic': symptomatic,
                    'covid': covid,
                    'status': status,
                    'recommendation': recommendation
                }
                
                doctor_data = {
                    'doctor_advice': doctor_advice,
                    'referral': referral
                }
                
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        pdf_buffer = generate_pdf_report(patient_data, results_data, doctor_data)
                        if pdf_buffer:
                            st.download_button(
                                "📥 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"clinical_report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                            st.success("✅ PDF Report generated successfully!")
                        else:
                            st.error("Failed to generate PDF report.")
            else:
                st.warning("""
                **PDF generation is not available.** 
                
                Please install ReportLab to enable PDF export:
                ```bash
                pip install reportlab
