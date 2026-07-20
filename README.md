<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a237e,50:0d47a1,100:00838f&height=280&section=header&text=AcousticBiomarker-GH&fontSize=56&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Cough-Acoustic%20Screening%20for%20Respiratory%20Pathogens%2C%20On-Device&descAlignY=58&descSize=18" width="100%"/>

<br/>

<a href="https://acoustic-biomarker-gh-salek05.streamlit.app/">
  <img src="https://readme-typing-svg.demolab.com/?lines=Cough+%E2%86%92+Log-Mel+Spectrogram+%E2%86%92+TFLite+MobileNetV2;Clinical+Triage+%2B+PDF%2FJSON%2FCSV+Export;Built+for+Low-Bandwidth%2C+Low-Resource+Screening;Sylhet+%F0%9F%87%A7%F0%9F%87%A9+%E2%86%92+The+World+%F0%9F%8C%8D&font=Fira+Code&center=true&width=780&height=50&duration=3200&pause=900&color=0D47A1&vCenter=true&size=25&weight=700" alt="Typing SVG" />
</a>

<br/><br/>

[![Live App](https://img.shields.io/badge/🚀_LIVE_APP-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://acoustic-biomarker-gh-salek05.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-orange?style=for-the-badge)]()

<br/>

![GitHub stars](https://img.shields.io/github/stars/muhammadsalek/acoustic-biomarker-gh?style=social)
![GitHub forks](https://img.shields.io/github/forks/muhammadsalek/acoustic-biomarker-gh?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/muhammadsalek/acoustic-biomarker-gh?color=0d47a1)
![GitHub repo size](https://img.shields.io/github/repo-size/muhammadsalek/acoustic-biomarker-gh?color=1a237e)
![Profile views](https://komarev.com/ghpvc/?username=muhammadsalek-acousticbiomarker&color=0d47a1&style=flat)

<br/>

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="480">

</div>

<br/>

<p align="center">
  <i>"A cough carries acoustic information about the airway that a questionnaire cannot capture — the question is whether a phone microphone and a 2.3M-parameter model can responsibly listen for it."</i>
</p>

<br/>

> ⚠️ **Research status.** AcousticBiomarker-GH is a **research prototype and interface demonstrator**, not a validated diagnostic device. Inference on uploaded audio is real (TFLite forward pass through the bundled model). The **Advanced Analytics** and **Explainability** tabs currently render illustrative/placeholder statistics — designed to show what a fully externally-validated version of this dashboard *would* report — pending prospective validation against ground-truth clinical labels. See [Validation Status](#-validation-status--limitations) before citing any performance figure from this repository.

<br/>

---

## 📌 Table of Contents

<table align="center">
<tr>
<td width="33%" valign="top">

**🎯 Overview**
- [What is this?](#-what-is-acousticbiomarker-gh)
- [Motivation](#-motivation)
- [Live demo](#-live-demo)
- [Validation status](#-validation-status--limitations)

</td>
<td width="33%" valign="top">

**🛠️ System**
- [Signal pipeline](#-signal-processing-pipeline)
- [Model card](#-model-card)
- [Clinical triage logic](#-clinical-triage-logic)
- [Dashboard tabs](#-dashboard-tabs)

</td>
<td width="33%" valign="top">

**⚙️ Engineering**
- [Architecture](#-architecture)
- [Tech stack](#-tech-stack)
- [Installation](#-installation)
- [Roadmap](#-roadmap)
- [Citation](#-citation)

</td>
</tr>
</table>

<br/>

---

## 🎯 What is AcousticBiomarker-GH?

<img align="right" width="320" src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385e21e2c9.gif">

**AcousticBiomarker-GH** is a single-page Streamlit clinical-decision-support interface that takes a **3-second cough recording** (`.wav`), converts it to a **128-band log-mel spectrogram**, and runs it through a **quantized MobileNetV2 model in TensorFlow Lite** to produce a 3-way probability distribution over:

```
Healthy   ·   Symptomatic   ·   COVID-19
```

Those probabilities are fed into a rule-based triage layer that maps continuous risk into a discrete clinical action (monitor / telemedicine / escalate), alongside patient metadata (age, comorbidities, vaccination status, exposure history) captured through a structured intake sidebar.

The design goal is not just "run a model" — it is to demonstrate what a **deployable, low-bandwidth, exportable screening interface** looks like end-to-end: spectrogram visualization, statistical reporting, structured PDF/CSV/JSON export, and a compact binary telemetry packet intended for 2G-class connectivity in low-resource settings.

<br clear="right"/>

### Motivation

| Constraint in respiratory screening | How this system is designed to respond |
|---|---|
| PCR/lab testing has latency and cost | Instant, on-device inference from a 3-second recording |
| Rural/low-resource clinics often lack broadband | 19-byte binary telemetry packet, base64-encoded for narrowband transmission |
| Screening tools need a human-readable action, not just a probability | Rule-based triage → plain-language recommendation + specialist-style advice |
| Clinical audit trails require structured records | SHA-256–stamped export bundle: JSON, CSV, plain-text, and PDF report |
| Non-specialist users need to see *why* a class was flagged | Log-mel spectrogram + waveform view, feature-importance panel, confusion-matrix view |

<br/>

---

## 🌐 Live Demo

<div align="center">

### 👉 [**acoustic-biomarker-gh-salek05.streamlit.app**](https://acoustic-biomarker-gh-salek05.streamlit.app/) 👈

Upload a `.wav` cough recording to see the full pipeline run: preprocessing → inference → triage → multi-tab report.

</div>

<br/>

---

## ✅ Validation Status & Limitations

<div align="center">

| Component | Status |
|---|:---:|
| Audio preprocessing (`librosa` log-mel extraction) | ✅ Live, deterministic |
| TFLite MobileNetV2 inference on uploaded audio | ✅ Live model forward pass |
| Triage thresholds (probability → action code) | ✅ Live, rule-based (see below) |
| AUC-ROC / Sensitivity / Specificity / MCC table | 🟡 **Placeholder** — fixed demonstration values, not measured on a held-out set |
| ROC curve | 🟡 **Simulated** curve for interface design, not fit to real labels |
| SHAP-style feature importance | 🟡 **Randomly generated** each run (`np.random`) — illustrates the intended panel layout, not real attribution |
| Confusion matrix | 🟡 **Static placeholder matrix**, not from a validation run |
| Training corpus (COUGHVID + Virufy, per in-app system info) | 🟡 Stated in-app; not independently verifiable from this repository alone — recommend documenting the exact training/eval split, preprocessing parity, and class balance before citing |

</div>

This distinction matters for anyone citing this work academically: the **pipeline and interface are real and reproducible**; the **reported clinical performance metrics are currently placeholders describing the target reporting format**, not results from an external validation study. A rigorous next step (tracked in the [Roadmap](#-roadmap)) is to replace every placeholder panel with metrics computed on a documented, held-out clinical dataset — with the model card updated to match.

The in-app footer likewise states this directly: *"All clinical decisions should be validated by healthcare professionals. This is a research tool and not a substitute for clinical diagnosis."*

<br/>

---

## 🔬 Signal Processing Pipeline

<div align="center">

```mermaid
graph LR
    A[🎤 .wav Upload] --> B[librosa.load<br/>16 kHz, 3.0 s, mono]
    B --> C{Pad / Truncate<br/>to 48,000 samples}
    C --> D[Peak Normalization]
    D --> E[Log-Mel Spectrogram<br/>128 mels · n_fft 2048 · hop 512]
    E --> F[Stack ×3 channels<br/>128×94×3 tensor]
    F --> G[TFLite Interpreter<br/>MobileNetV2, INT8]
    G --> H[Softmax: Healthy / Symptomatic / COVID]
    H --> I{Triage Rule}
    I --> J[📋 Clinical Report + Export]
    style A fill:#1a237e,color:#fff
    style E fill:#0d47a1,color:#fff
    style G fill:#00838f,color:#fff
    style I fill:#dc3545,color:#fff
    style J fill:#28a745,color:#fff
```

</div>

| Stage | Parameter |
|---|---|
| Sample rate | 16,000 Hz |
| Clip duration | 3.0 s (padded/truncated to 48,000 samples) |
| Normalization | Peak amplitude normalization |
| Mel bands | 128 |
| FFT size | 2,048 |
| Hop length | 512 |
| Frequency range | 0–8,000 Hz |
| Model input tensor | 128 × 94 × 3 (log-mel replicated across 3 channels) |

<br/>

---

## 🧠 Model Card

<div align="center">

| Field | Value |
|---|---|
| **Architecture** | MobileNetV2 (quantized) |
| **Parameters** | ~2.3M |
| **Framework** | TensorFlow Lite 2.15.0 |
| **Quantization** | INT8 |
| **Input shape** | 128 × 94 × 3 |
| **Output classes** | 3 — Healthy / Symptomatic / COVID-19 |
| **Reported inference latency** | ~10.6 ms per clip (in-app) |
| **Stated training corpora** | COUGHVID + Virufy (per in-app system panel — see [Validation Status](#-validation-status--limitations)) |
| **Model file** | `acoustic_biomarker_quantized.tflite` |

</div>

> 💡 For an academic audience, the strongest version of this section replaces "stated" with a linked, versioned data card: exact COUGHVID/Virufy subset sizes, class balance, train/val/test split strategy, and whether any deduplication was done across the two source datasets (both are cough-audio corpora and can overlap in acoustic conditions if not handled carefully).

<br/>

---

## 🚦 Clinical Triage Logic

The model's `COVID-19` and `Symptomatic` probabilities are mapped to a discrete action via fixed thresholds:

<div align="center">

| Condition | Triage Level | Action Code | Suggested Action |
|---|---|:---:|---|
| `P(COVID) ≥ 0.70` | 🔴 Critical | 1 | Immediate clinical evaluation, isolation, urgent care escalation |
| `P(COVID) ≥ 0.35` **or** `P(Symptomatic) > 0.50` | 🟠 Moderate | 2 | Telemedicine consult, PCR testing within 24h, self-isolation |
| Otherwise | 🟢 Stable | 3 | Routine monitoring, standard precautions |

</div>

Each level renders a plain-language recommendation, a specialist-style clinical note, and a referral pathway. Because these thresholds are fixed constants rather than learned or calibrated cut-points, they should be treated as an interface convention to be tuned against a real cost-sensitivity/decision-curve analysis, not as clinically-validated cutoffs.

<br/>

---

## 📊 Dashboard Tabs

<table>
<tr><th>Tab</th><th>Contents</th></tr>
<tr>
<td>🌊 <b>Spectrogram</b></td>
<td>Log-mel spectrogram (matplotlib) and raw time-domain waveform of the uploaded cough clip</td>
</tr>
<tr>
<td>📈 <b>Advanced Analytics</b></td>
<td>Performance metrics table, triage distribution chart, ROC curve, calibration curve, Cohen's Kappa / F1 / MCC — see <a href="#-validation-status--limitations">Validation Status</a> for which of these are live vs. placeholder</td>
</tr>
<tr>
<td>🔍 <b>Explainability</b></td>
<td>SHAP-style feature-importance bar chart (MFCCs, spectral centroid, zero-crossing rate, energy) and a confusion matrix panel</td>
</tr>
<tr>
<td>📡 <b>Telemetry</b></td>
<td>19-byte binary packet (<code>struct.pack('>IBBfffB', ...)</code>) — device ID, age, gender, class probabilities, action code — base64-encoded for low-bandwidth transmission</td>
</tr>
<tr>
<td>💾 <b>Export</b></td>
<td>JSON, CSV, and plain-text clinical report, each stamped with a SHA-256 integrity hash</td>
</tr>
<tr>
<td>📄 <b>PDF Report</b></td>
<td>Formatted multi-section clinical PDF via <code>reportlab</code> — patient info, results table with 95% CIs, recommendation, referral</td>
</tr>
</table>

<br/>

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Streamlit Runtime                          │
│                                                                     │
│  Sidebar (intake form)          Main panel                        │
│  ├─ Patient demographics   ┌──▶ File uploader (.wav)               │
│  ├─ Symptoms / comorbid.   │    ├─ preprocess_audio()               │
│  ├─ Vaccination / exposure │    │   └─ librosa → log-mel tensor    │
│  └─ Research toggles       │    ├─ TFLite Interpreter.invoke()      │
│         (SHAP/ROC/calib.)  │    ├─ get_triage_level()               │
│                             │    ├─ get_recommendation()            │
│                             │    └─ Tabs: Spectrogram · Analytics · │
│                             │             Explainability · Telemetry│
│                             │             · Export · PDF Report     │
└────────────────────────────────────────────────────────────────────┘
```

**Key design decision:** the TFLite interpreter is loaded once via `@st.cache_resource`, and inference happens only after a file is uploaded — spectrogram, triage, and every downstream tab are derived from that single forward pass, keeping the report internally consistent across tabs.

<br/>

---

## 🧰 Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **UI Framework** | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| **Audio Processing** | ![librosa](https://img.shields.io/badge/-librosa-000000?style=flat-square&logo=python&logoColor=white) |
| **Model Runtime** | ![TensorFlow Lite](https://img.shields.io/badge/-TensorFlow_Lite-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) |
| **Numerics / Stats** | ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=python&logoColor=white) ![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=flat-square) |
| **Reporting** | ![ReportLab](https://img.shields.io/badge/-ReportLab-2b2b2b?style=flat-square) (PDF) · `pandas` (CSV/JSON) |
| **Language / Runtime** | ![Python](https://img.shields.io/badge/-Python_3.10-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Hosting** | ![Streamlit Cloud](https://img.shields.io/badge/-Streamlit_Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |

</div>

<br/>

---

## 📁 Project Structure

```
acoustic-biomarker-gh/
│
├── app.py                             # Single-file Streamlit application (v2.1)
│   ├── load_model()                    # Cached TFLite interpreter loader
│   ├── preprocess_audio()              # librosa → log-mel tensor
│   ├── get_triage_level()              # Probability → clinical action
│   ├── get_recommendation()            # Action → plain-language advice
│   ├── generate_telemetry()            # struct.pack → base64 packet
│   ├── generate_pdf_report()           # reportlab clinical PDF
│   └── main render flow (tabs 1–6)
│
├── acoustic_biomarker_quantized.tflite # Quantized MobileNetV2 model
├── requirements.txt                    # streamlit, librosa, tensorflow, reportlab, ...
├── runtime.txt                         # python-3.10
└── README.md                           # You are here
```

<br/>

---

## ⚙️ Installation

<table>
<tr><th>Step</th><th>Command</th></tr>
<tr>
<td>1. Clone</td>
<td>

```bash
git clone https://github.com/muhammadsalek/acoustic-biomarker-gh.git
cd acoustic-biomarker-gh
```

</td>
</tr>
<tr>
<td>2. Create environment</td>
<td>

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

</td>
</tr>
<tr>
<td>3. Install dependencies</td>
<td>

```bash
pip install -r requirements.txt
```

</td>
</tr>
<tr>
<td>4. Run locally</td>
<td>

```bash
streamlit run app.py
```

</td>
</tr>
</table>

**`requirements.txt`**

```txt
streamlit>=1.30.0
numpy>=1.23.5
librosa>=0.10.1
tensorflow>=2.15.0
matplotlib>=3.7.0
seaborn>=0.13.0
scikit-learn>=1.2.2
pandas>=2.0.0
plotly>=5.18.0
scipy>=1.11.0
reportlab>=4.0.0
```

**`runtime.txt`**
```
python-3.10
```

<br/>

---

## 🗺️ Roadmap

- [x] End-to-end audio → spectrogram → TFLite inference pipeline
- [x] Rule-based clinical triage layer
- [x] JSON / CSV / plain-text / PDF export with SHA-256 integrity stamp
- [x] Low-bandwidth binary telemetry encoding
- [ ] **Replace placeholder Advanced Analytics table with metrics computed on a documented, held-out validation split**
- [ ] **Replace simulated ROC/calibration curves with curves fit to real predicted probabilities and ground-truth labels**
- [ ] **Replace randomly-generated SHAP panel with real SHAP/Integrated-Gradients attribution over the log-mel input**
- [ ] Publish a formal model & data card (training/eval split, class balance, demographic coverage, known failure modes)
- [ ] External validation on a prospective, geographically distinct cohort
- [ ] Multi-language UI (English / Bengali)

<div align="center">

![Progress](https://progress-bar.dev/45/?title=Toward+Externally-Validated+Release&width=420&color=0d47a1)

</div>

<br/>

---

## 📖 Citation

If this pipeline or interface informs your work, a citation is appreciated:

```bibtex
@software{miah_acoustic_biomarker_gh,
  author  = {Miah, Md Salek},
  title   = {AcousticBiomarker-GH: A TensorFlow Lite Cough-Acoustic Screening Interface},
  year    = {2026},
  url     = {https://github.com/muhammadsalek/acoustic-biomarker-gh},
  note    = {Research prototype; see Validation Status section for scope of verified vs. illustrative components}
}
```

<br/>

---

## 🔒 Privacy & Data Handling

<div align="center">

| Aspect | Behavior |
|---|---|
| Where is patient data stored? | In `st.session_state`, in-memory, for the duration of the browser session |
| Is audio uploaded to an external API? | No — inference runs locally via the bundled TFLite interpreter |
| What happens on page refresh? | Session state resets; export anything needed beforehand |

</div>

<br/>

---

## 📬 Contact

<div align="center">

[![Email](https://img.shields.io/badge/Email-saleksta%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:saleksta@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-muhammadsalek-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/muhammadsalek)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--5973--461X-A6CE39?style=for-the-badge&logo=orcid&logoColor=white)](https://orcid.org/0009-0005-5973-461X)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Md_Salek_Miah-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-salek-miah-b34309329/)

</div>

<br/>

## 📄 License

Released under the **MIT License**.

<br/>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00838f,50:0d47a1,100:1a237e&height=150&section=footer" width="100%"/>

</div>
