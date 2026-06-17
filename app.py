import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io, base64, datetime

# ─── Konfigurasi ────────────────────────────────────────────────────────────
MODEL_PATH = "model_patched.keras"

CLASS_NAMES = [
    "100000_kertas",
    "10000_kertas",
    "1000_kertas",
    "20000_kertas",
    "2000_kertas",
    "50000_kertas",
    "5000_kertas",
    "75000_kertas",
]

LABEL_DISPLAY = {
    "1000_kertas":   "Rp 1.000",
    "2000_kertas":   "Rp 2.000",
    "5000_kertas":   "Rp 5.000",
    "10000_kertas":  "Rp 10.000",
    "20000_kertas":  "Rp 20.000",
    "50000_kertas":  "Rp 50.000",
    "75000_kertas":  "Rp 75.000",
    "100000_kertas": "Rp 100.000",
}

IMG_SIZE = (224, 224)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Uang Kertas",
    page_icon="💜",
    layout="centered",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #F5F3FF; }
    .stAppToolbar { display: none; }

    .header-wrap {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 60%, #4C1D95 100%);
        border-radius: 20px;
        padding: 36px 40px 32px;
        margin-bottom: 28px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(124,58,237,0.25);
    }
    .header-wrap h1 { color:#fff; font-size:2rem; font-weight:800; margin:0 0 6px; letter-spacing:-0.5px; }
    .header-wrap p  { color:#DDD6FE; font-size:0.95rem; margin:0; }
    .header-badge {
        display:inline-block; background:rgba(255,255,255,0.15); color:#EDE9FE;
        font-size:0.75rem; font-weight:600; padding:4px 12px; border-radius:99px;
        margin-bottom:14px; letter-spacing:0.5px;
    }

    .card {
        background:#fff; border-radius:16px; padding:24px;
        box-shadow:0 2px 12px rgba(109,40,217,0.08);
        border:1px solid #EDE9FE; margin-bottom:20px;
    }
    .card-title {
        font-size:0.8rem; font-weight:600; color:#7C3AED;
        text-transform:uppercase; letter-spacing:0.8px; margin-bottom:14px;
    }
    .result-card {
        background:linear-gradient(135deg,#7C3AED,#6D28D9);
        border-radius:16px; padding:28px 24px; text-align:center;
        box-shadow:0 8px 24px rgba(124,58,237,0.3); margin-bottom:12px;
    }
    .result-label { color:#DDD6FE; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px; }
    .result-value { color:#fff; font-size:2.4rem; font-weight:800; letter-spacing:-1px; line-height:1.1; }
    .result-conf  { color:#C4B5FD; font-size:0.9rem; margin-top:8px; font-weight:500; }

    .pill { display:inline-block; padding:8px 18px; border-radius:99px; font-size:0.85rem; font-weight:600; width:100%; text-align:center; margin-top:10px; }
    .pill-success { background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; }
    .pill-warning { background:#FFFBEB; color:#92400E; border:1px solid #FCD34D; }
    .pill-error   { background:#FEF2F2; color:#991B1B; border:1px solid #FECACA; }

    [data-testid="stFileUploader"] { background:#fff; border-radius:16px; padding:8px; border:2px dashed #C4B5FD; }
    [data-testid="stAlert"] { border-radius:12px; }
    .stSpinner > div { border-top-color:#7C3AED !important; }
    .streamlit-expanderHeader { background:#F5F3FF; border-radius:10px; color:#6D28D9; font-weight:600; }

    /* ── Riwayat ── */
    .history-wrap {
        background:#fff; border-radius:16px; padding:24px;
        box-shadow:0 2px 12px rgba(109,40,217,0.08);
        border:1px solid #EDE9FE;
    }
    .history-header {
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom:18px;
    }
    .history-title {
        font-size:0.8rem; font-weight:600; color:#7C3AED;
        text-transform:uppercase; letter-spacing:0.8px;
    }
    .history-count {
        background:#EDE9FE; color:#6D28D9; font-size:0.75rem;
        font-weight:700; padding:3px 10px; border-radius:99px;
    }
    .history-row {
        display:flex; align-items:center; gap:14px;
        padding:10px 0; border-bottom:1px solid #F5F3FF;
    }
    .history-row:last-child { border-bottom:none; padding-bottom:0; }
    .history-thumb {
        width:52px; height:52px; border-radius:10px;
        object-fit:cover; border:2px solid #EDE9FE; flex-shrink:0;
    }
    .history-nominal {
        font-size:1rem; font-weight:700; color:#1E1B4B; margin-bottom:2px;
    }
    .history-conf { font-size:0.78rem; color:#8B5CF6; font-weight:500; }
    .history-time { font-size:0.72rem; color:#94A3B8; margin-top:2px; }
    .history-badge-ok  { background:#ECFDF5; color:#065F46; font-size:0.7rem; font-weight:600; padding:2px 8px; border-radius:99px; }
    .history-badge-warn{ background:#FFFBEB; color:#92400E; font-size:0.7rem; font-weight:600; padding:2px 8px; border-radius:99px; }
    .history-badge-err { background:#FEF2F2; color:#991B1B; font-size:0.7rem; font-weight:600; padding:2px 8px; border-radius:99px; }
    .history-empty { text-align:center; padding:32px 0; color:#C4B5FD; font-size:0.875rem; }

    .stat-box {
        background:linear-gradient(135deg,#EDE9FE,#F5F3FF);
        border-radius:12px; padding:14px 18px; text-align:center;
        border:1px solid #DDD6FE;
    }
    .stat-val { font-size:1.5rem; font-weight:800; color:#6D28D9; line-height:1; }
    .stat-lbl { font-size:0.72rem; color:#8B5CF6; font-weight:600; margin-top:4px; text-transform:uppercase; letter-spacing:0.5px; }

    .footer { text-align:center; color:#A78BFA; font-size:0.78rem; padding:20px 0 8px; }
    .block-container { padding-top:2rem; padding-bottom:2rem; max-width:780px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dict
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import keras, os
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"File '{MODEL_PATH}' tidak ditemukan. Jalankan dulu: python patch_model.py"
        )
    keras.config.enable_unsafe_deserialization()
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

# ─── Helper ─────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE, Image.BICUBIC)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)

def image_to_b64(image: Image.Image, size=(80, 80)) -> str:
    thumb = image.copy()
    thumb.thumbnail(size, Image.BICUBIC)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()

def plot_confidence(probs):
    labels   = [LABEL_DISPLAY[c] for c in CLASS_NAMES]
    top_idx  = int(np.argmax(probs))
    colors   = ["#7C3AED" if i == top_idx else "#DDD6FE" for i in range(len(probs))]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bars = ax.barh(labels, probs * 100, color=colors, height=0.55, zorder=3)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=9,
                color="#7C3AED" if prob == probs[top_idx] else "#94A3B8",
                fontweight="600" if prob == probs[top_idx] else "400")

    ax.set_xlim(0, 118)
    ax.tick_params(colors="#374151", labelsize=9)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.xaxis.set_visible(False)
    ax.set_title("Probabilitas Semua Kelas", fontsize=11, fontweight="700",
                 color="#1E1B4B", pad=14, loc="left")
    ax.yaxis.set_tick_params(length=0)
    ax.grid(axis="x", color="#F3F4F6", zorder=0)
    plt.tight_layout()
    return fig

# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-wrap">
    <div class="header-badge">💜 Machine Learning Demo</div>
    <h1>Deteksi Uang Kertas Indonesia</h1>
    <p>Upload foto uang kertas — model akan mengenali nominalnya secara otomatis</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat model..."):
    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.code("python patch_model.py", language="bash")
        st.stop()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

st.success("Model siap digunakan!", icon="✅")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">📁 Upload Gambar</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Seret & lepas gambar di sini, atau klik untuk memilih",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file:
    image = Image.open(uploaded_file)

    # Prediksi hanya jika file baru (hindari re-run tanpa upload baru)
    file_id = uploaded_file.file_id
    if file_id != st.session_state.last_file_id:
        with st.spinner("Menganalisis..."):
            input_arr = preprocess(image)
            preds     = model.predict(input_arr, verbose=0)[0]

        top_idx   = int(np.argmax(preds))
        top_label = CLASS_NAMES[top_idx]
        top_conf  = float(preds[top_idx])

        # Simpan ke riwayat
        st.session_state.history.insert(0, {
            "thumb_b64": image_to_b64(image),
            "label":     top_label,
            "conf":      top_conf,
            "time":      datetime.datetime.now().strftime("%H:%M:%S"),
            "preds":     preds.tolist(),
        })
        st.session_state.last_file_id = file_id
    else:
        # Ambil dari riwayat terakhir
        entry     = st.session_state.history[0]
        preds     = np.array(entry["preds"])
        top_idx   = int(np.argmax(preds))
        top_label = entry["label"]
        top_conf  = entry["conf"]

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🖼️ Gambar Input</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Nominal Terdeteksi</div>
            <div class="result-value">{LABEL_DISPLAY[top_label]}</div>
            <div class="result-conf">Confidence: {top_conf * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if top_conf >= 0.85:
            st.markdown('<div class="pill pill-success">✅ Model sangat yakin</div>', unsafe_allow_html=True)
        elif top_conf >= 0.60:
            st.markdown('<div class="pill pill-warning">⚠️ Cukup yakin, ada kemungkinan salah</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pill pill-error">📷 Confidence rendah — coba foto lebih jelas</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig = plot_confidence(preds)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 Lihat semua probabilitas"):
        for cls, prob in sorted(zip(CLASS_NAMES, preds), key=lambda x: -x[1]):
            bar_pct = int(prob * 100)
            color   = "#7C3AED" if cls == top_label else "#C4B5FD"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <div style="width:90px;font-size:0.85rem;font-weight:{'700' if cls==top_label else '400'};
                            color:{'#1E1B4B' if cls==top_label else '#64748B'}">{LABEL_DISPLAY[cls]}</div>
                <div style="flex:1;background:#F5F3FF;border-radius:99px;height:8px;">
                    <div style="width:{bar_pct}%;background:{color};height:8px;border-radius:99px;"></div>
                </div>
                <div style="width:48px;text-align:right;font-size:0.82rem;color:{color};font-weight:600">
                    {prob*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:48px 24px;color:#A78BFA;">
        <div style="font-size:3rem;margin-bottom:12px;">💜</div>
        <div style="font-size:1rem;font-weight:600;color:#6D28D9;margin-bottom:6px;">Belum ada gambar</div>
        <div style="font-size:0.875rem;color:#8B5CF6;">Upload foto uang kertas di atas untuk memulai prediksi</div>
    </div>
    """, unsafe_allow_html=True)

# ── Riwayat Prediksi ─────────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
history = st.session_state.history

# Stat bar (hanya tampil jika ada riwayat)
if history:
    total      = len(history)
    avg_conf   = np.mean([h["conf"] for h in history]) * 100
    top_akurat = sum(1 for h in history if h["conf"] >= 0.85)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{total}</div><div class="stat-lbl">Total Prediksi</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{avg_conf:.0f}%</div><div class="stat-lbl">Rata-rata Confidence</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{top_akurat}</div><div class="stat-lbl">Sangat Yakin (≥85%)</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# Tabel riwayat
st.markdown('<div class="history-wrap">', unsafe_allow_html=True)
st.markdown(f"""
<div class="history-header">
    <div class="history-title">🕓 Riwayat Prediksi</div>
    <div class="history-count">{len(history)} gambar</div>
</div>
""", unsafe_allow_html=True)

if not history:
    st.markdown('<div class="history-empty">Belum ada prediksi — upload gambar untuk memulai</div>', unsafe_allow_html=True)
else:
    for entry in history:
        conf = entry["conf"]
        if conf >= 0.85:
            badge = '<span class="history-badge-ok">✅ Sangat Yakin</span>'
        elif conf >= 0.60:
            badge = '<span class="history-badge-warn">⚠️ Cukup Yakin</span>'
        else:
            badge = '<span class="history-badge-err">❌ Rendah</span>'

        st.markdown(f"""
        <div class="history-row">
            <img class="history-thumb" src="data:image/jpeg;base64,{entry['thumb_b64']}" />
            <div style="flex:1;">
                <div class="history-nominal">{LABEL_DISPLAY[entry['label']]}</div>
                <div class="history-conf">{conf*100:.1f}% confidence &nbsp; {badge}</div>
                <div class="history-time">🕐 {entry['time']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Tombol reset
if history:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Hapus Riwayat", use_container_width=False):
        st.session_state.history = []
        st.session_state.last_file_id = None
        st.rerun()

# Footer
st.markdown("""
<div class="footer">
    Model: Hybrid CNN &nbsp;·&nbsp; Input: 224×224 RGB &nbsp;·&nbsp; 8 kelas uang kertas Indonesia
</div>
""", unsafe_allow_html=True)