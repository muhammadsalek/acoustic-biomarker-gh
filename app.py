import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import base64
import struct
import matplotlib.pyplot as plt

st.set_page_config(page_title="AcousticBiomarker-GH", layout="centered", page_icon="🌊")

st.markdown("<h1 style='text-align:center;color:#008080;'>🌊 AcousticBiomarker-GH</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Edge-AI Cough Screening</h4>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.lite.Interpreter(model_path="acoustic_biomarker_quantized.tflite")

interpreter = load_model()
interpreter.allocate_tensors()

uploaded = st.file_uploader("Upload Cough (.wav)", type=["wav"])

if uploaded:
    st.audio(uploaded)
    with st.spinner("Processing..."):
        y, sr = librosa.load(uploaded, sr=16000, duration=3.0)
        if len(y) < 48000:
            y = np.pad(y, (0, 48000 - len(y)))
        else:
            y = y[:48000]
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        X = np.expand_dims(np.repeat(np.expand_dims(log_mel, -1), 3, -1), 0).astype(np.float32)
        
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        interpreter.set_tensor(inp['index'], X)
        interpreter.invoke()
        probs = interpreter.get_tensor(out['index'])[0]
        
        healthy, symptomatic, covid = probs
        
        if covid >= 0.70:
            status, color = "🔴 CRITICAL - EVACUATE", "#B22222"
        elif covid >= 0.35 or symptomatic > 0.50:
            status, color = "🟠 ALERT - TELEMEDICINE", "#E67E22"
        else:
            status, color = "🟢 STABLE - MONITOR", "#27AE60"
        
        st.markdown(f"### Result: <span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Healthy", f"{healthy*100:.1f}%")
        c2.metric("Symptomatic", f"{symptomatic*100:.1f}%")
        c3.metric("COVID-19", f"{covid*100:.1f}%")
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(log_mel, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        payload = struct.pack('>IBBfffB', 40409, 45, ord('F'), float(healthy), float(symptomatic), float(covid), 1)
        b64 = base64.b64encode(payload).decode()
        st.code(b64, language="text")
        st.download_button("Download Telemetry", data=b64, file_name="telemetry.txt")
