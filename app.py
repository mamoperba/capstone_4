"""
app.py
======
Construction Safety Monitor — Streamlit Application

Fitur:
  ✅ Upload gambar → deteksi APD dengan YOLOv8
  ✅ Bounding box berwarna per class
  ✅ Safety Compliance Score & Risk Level
  ✅ IoU-based person matching (lebih akurat)
  ✅ Detail pelanggaran per pekerja
  ✅ Confidence & IoU threshold slider
  ✅ Statistik deteksi lengkap + avg confidence
  ✅ Perbandingan original vs deteksi
  ✅ Download hasil annotasi
  ✅ Upload multiple images (batch mode)

Cara jalankan:
  streamlit run app.py
"""

import os
import io
try:
    import cv2
except ImportError as e:
    import streamlit as st
    st.error(
        f"""❌ OpenCV gagal diimport: {e}

Kemungkinan penyebab:
1. Python version tidak kompatibel (butuh Python 3.11)
2. System library kurang (libgl1-mesa-glx)

Pastikan file berikut ada di repo:
- runtime.txt berisi: python-3.11.9
- packages.txt berisi: libgl1-mesa-glx
"""
    )
    st.stop()
import numpy as np
import streamlit as st
from PIL import Image
from collections import Counter
from ultralytics import YOLO

try:
    import gdown
except ImportError:
    gdown = None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Safety Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 28px 32px; color: white; margin-bottom: 20px;
}
.hero-title { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
.hero-sub   { font-size: 13px; opacity: .85; }
.hero-badges { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
.hero-badge  {
    background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
    border-radius: 20px; padding: 3px 12px; font-size: 11px; font-weight: 500;
}

.metric-row  { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 18px; }
.metric-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; text-align: center; }
.metric-icon  { font-size: 20px; margin-bottom: 5px; }
.metric-value { font-size: 24px; font-weight: 700; }
.metric-label { font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: .5px; margin-top: 3px; }
.mv-blue   { color: #1e40af; }
.mv-green  { color: #15803d; }
.mv-red    { color: #dc2626; }
.mv-amber  { color: #d97706; }

.risk-badge  {
    display: inline-block; padding: 6px 18px; border-radius: 20px;
    font-size: 15px; font-weight: 700; letter-spacing: .5px;
}
.risk-low    { background: #dcfce7; color: #15803d; }
.risk-medium { background: #fef9c3; color: #a16207; }
.risk-high   { background: #fee2e2; color: #dc2626; }

.violation-box {
    background: #fff5f5; border: 1px solid #fecaca; border-left: 4px solid #dc2626;
    border-radius: 8px; padding: 12px 14px; margin: 8px 0;
    font-size: 13px; color: #7f1d1d;
}
.safe-box {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a;
    border-radius: 8px; padding: 12px 14px; margin: 8px 0;
    font-size: 13px; color: #14532d;
}
.person-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 8px 12px; margin: 4px 0; font-size: 12px;
    display: flex; align-items: center; gap: 8px;
}
.compliance-bar-wrap {
    background: #f1f5f9; border-radius: 20px; height: 20px;
    overflow: hidden; margin: 8px 0;
}
.compliance-bar {
    height: 100%; border-radius: 20px;
    display: flex; align-items: center; padding-left: 10px;
    font-size: 12px; font-weight: 600; color: white;
    transition: width .5s ease;
}
.class-row   {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px;
}
.class-dot   { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.class-count { font-weight: 700; font-size: 16px; }
.section-title {
    font-size: 14px; font-weight: 600; color: #1e293b;
    margin: 16px 0 10px; padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
}
.batch-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px; margin-bottom: 8px;
}
.batch-filename { font-size: 11px; color: #64748b; margin-bottom: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']

CLASS_COLORS_BGR = {
    'helmet':    (46, 204, 113),
    'no-helmet': (231,  76,  60),
    'no-vest':   (230, 126,  34),
    'person':    ( 52, 152, 219),
    'vest':      ( 39, 174,  96),
}
CLASS_COLORS_HEX = {
    'helmet':    '#2ecc71',
    'no-helmet': '#e74c3c',
    'no-vest':   '#e67e22',
    'person':    '#3498db',
    'vest':      '#27ae60',
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
MODEL_PATH    = "construction_safety_best.pt"

# ── Konfigurasi Google Drive untuk auto-download model ────────────────────────
# Setelah training selesai di Colab:
# 1. Upload best.pt ke Google Drive
# 2. Klik kanan file → Get link → ubah ke "Anyone with the link"
# 3. Copy FILE_ID dari URL:
#    https://drive.google.com/file/d/FILE_ID_DISINI/view
# 4. Paste FILE_ID di bawah ini
GDRIVE_FILE_ID = "GANTI_DENGAN_FILE_ID_DARI_GOOGLE_DRIVE"


def download_model_from_gdrive(file_id: str, dest_path: str) -> bool:
    """
    Download model .pt dari Google Drive menggunakan gdown.
    Diperlukan agar Streamlit Cloud bisa mengakses model
    (file .pt tidak bisa diupload ke GitHub karena ukurannya besar).

    Cara mendapatkan file_id:
    - Buka Google Drive → klik kanan file .pt → Share → Copy link
    - Ambil bagian ID dari URL: drive.google.com/file/d/[FILE_ID]/view
    """
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Mengunduh model dari Google Drive... (hanya sekali)"):
            gdown.download(url, dest_path, quiet=False)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            st.success("Model berhasil diunduh!")
            return True
        else:
            st.error("Download gagal atau file terlalu kecil. Cek FILE_ID Google Drive.")
            return False
    except Exception as e:
        st.error(f"Error saat download model: {e}")
        return False


@st.cache_resource
def load_model():
    """
    Load YOLOv8 model dengan strategi bertingkat:
    1. Cek apakah model sudah ada secara lokal (local testing)
    2. Jika tidak ada dan GDRIVE_FILE_ID sudah diisi → auto-download dari Drive
    3. Jika FILE_ID belum diisi → tampilkan instruksi setup

    Cached dengan @st.cache_resource agar tidak reload setiap interaksi.
    """
    # Strategi 1: Model sudah ada lokal (local testing / sudah pernah download)
    if os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            return None

    # Strategi 2: Download dari Google Drive (Streamlit Cloud)
    if GDRIVE_FILE_ID and GDRIVE_FILE_ID != "GANTI_DENGAN_FILE_ID_DARI_GOOGLE_DRIVE":
        success = download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
        if success:
            try:
                return YOLO(MODEL_PATH)
            except Exception as e:
                st.error(f"Gagal load model setelah download: {e}")
                return None

    # Strategi 3: Model tidak ada, FILE_ID belum dikonfigurasi
    st.error("Model belum tersedia. Ikuti langkah berikut:")
    st.markdown("""
    **Setup untuk Local Testing:**
    ```
    1. Jalankan notebook Capstone4_ConstructionSafety.ipynb di Google Colab
    2. Download file construction_safety_best.pt dari Google Drive
    3. Letakkan file .pt di folder yang sama dengan app.py
    4. Jalankan: streamlit run app.py
    ```

    **Setup untuk Streamlit Cloud:**
    ```
    1. Upload best.pt ke Google Drive
    2. Share file → "Anyone with the link can view"
    3. Copy FILE_ID dari URL Google Drive
    4. Edit app.py baris GDRIVE_FILE_ID = "paste_file_id_disini"
    5. Push ke GitHub → Streamlit Cloud akan auto-download model
    ```
    """)
    return None


model = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_iou(box_a: list, box_b: list) -> float:
    """
    Hitung IoU antara dua bounding box format [x1, y1, x2, y2].
    Digunakan untuk mencocokkan posisi person dengan APD secara spasial.
    """
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])

    inter_w = max(0, xa2 - xa1)
    inter_h = max(0, ya2 - ya1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def run_detection(image: np.ndarray, conf: float, iou: float,
                  iou_match_thresh: float = 0.25) -> dict:
    """
    Jalankan deteksi + analisis safety compliance (IoU-based person matching).

    IoU-based matching lebih akurat dibanding counting sederhana karena:
    - Setiap 'person' diperiksa secara individual
    - Menghindari double-count ketika 1 orang melanggar 2 aturan sekaligus
    - Bisa mendeteksi 'siapa' yang melanggar (bukan hanya 'berapa banyak')

    iou_match_thresh=0.15 digunakan karena bounding box helm (kecil)
    secara natural memiliki IoU rendah dengan bounding box person (besar).
    """
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)[0]

    # ── Kumpulkan semua box per class ─────────────────────────────────────
    counts       = Counter()
    confidences  = {}
    person_boxes    = []
    no_helmet_boxes = []
    no_vest_boxes   = []
    vest_boxes      = []   # tambahan untuk vest-override logic

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id   = int(box.cls.item())
            cls_name = CLASS_NAMES[cls_id]
            conf_val = float(box.conf.item())
            xyxy     = box.xyxy[0].tolist()

            counts[cls_name] += 1
            confidences.setdefault(cls_name, []).append(conf_val)

            if cls_name == 'person':
                person_boxes.append(xyxy)
            elif cls_name == 'no-helmet':
                no_helmet_boxes.append(xyxy)
            elif cls_name == 'no-vest':
                no_vest_boxes.append(xyxy)
            elif cls_name == 'vest':
                vest_boxes.append(xyxy)   # kumpulkan box vest

    # ── Gambar bounding box dengan warna custom ───────────────────────────
    annotated = image.copy()
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id   = int(box.cls.item())
            cls_name = CLASS_NAMES[cls_id]
            conf_val = float(box.conf.item())
            color    = CLASS_COLORS_BGR[cls_name]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf_val:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-lh-8), (x1+lw+6, y1), color, -1)
            cv2.putText(annotated, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── IoU-based person matching ─────────────────────────────────────────
    person_status = []
    for i, p_box in enumerate(person_boxes):
        has_no_helmet = any(
            compute_iou(p_box, nh) >= iou_match_thresh
            for nh in no_helmet_boxes
        )
        has_no_vest = any(
            compute_iou(p_box, nv) >= iou_match_thresh
            for nv in no_vest_boxes
        )

        # ── Vest-override logic ───────────────────────────────────────────
        # Jika model mendeteksi BOTH vest DAN no-vest untuk person yang sama,
        # prioritaskan vest → anggap pekerja COMPLIANT.
        # Ini mencegah false positive ketika model ragu antara vest dan no-vest
        # (misalnya vest berwarna oranye terang yang mirip pakaian biasa).
        has_vest = any(
            compute_iou(p_box, v) >= iou_match_thresh
            for v in vest_boxes
        )
        if has_vest and has_no_vest:
            has_no_vest = False   # vest override: deteksi vest lebih dipercaya

        person_status.append({
            'idx':           i + 1,
            'has_no_helmet': has_no_helmet,
            'has_no_vest':   has_no_vest,
            'has_vest':      has_vest,
            'is_violating':  has_no_helmet or has_no_vest,
        })

    # ── Hitung ringkasan compliance ───────────────────────────────────────
    n_persons    = len(person_boxes)
    n_violations = sum(1 for p in person_status if p['is_violating'])
    n_compliant  = n_persons - n_violations

    # Fallback: jika tidak ada 'person' terdeteksi tapi ada no-helmet/no-vest
    if n_persons == 0:
        n_violations = counts.get('no-helmet', 0) + counts.get('no-vest', 0)
        n_compliant  = 0

    compliance_pct = round(
        (n_compliant / n_persons * 100) if n_persons > 0 else 100.0, 1
    )

    if compliance_pct >= 80:
        risk_level, risk_class = 'LOW',    'risk-low'
    elif compliance_pct >= 50:
        risk_level, risk_class = 'MEDIUM', 'risk-medium'
    else:
        risk_level, risk_class = 'HIGH',   'risk-high'

    violations = []
    n_nh = sum(1 for p in person_status if p['has_no_helmet'])
    n_nv = sum(1 for p in person_status if p['has_no_vest'])
    if n_nh > 0:
        violations.append(f"{n_nh} pekerja tidak memakai helm")
    if n_nv > 0:
        violations.append(f"{n_nv} pekerja tidak memakai safety vest")

    avg_conf = {
        cls_name: round(sum(confs) / len(confs), 3)
        for cls_name, confs in confidences.items()
    }

    return {
        'annotated_img':  annotated,
        'counts':         dict(counts),
        'avg_conf':       avg_conf,
        'n_persons':      n_persons,
        'n_helmet':       counts.get('helmet',    0),
        'n_no_helmet':    counts.get('no-helmet', 0),
        'n_vest':         counts.get('vest',      0),
        'n_no_vest':      counts.get('no-vest',   0),
        'n_compliant':    n_compliant,
        'n_violations':   n_violations,
        'compliance_pct': compliance_pct,
        'risk_level':     risk_level,
        'risk_class':     risk_class,
        'violations':     violations,
        'person_status':  person_status,
        'total_objects':  sum(counts.values()),
    }


def render_analysis_panel(c: dict):
    """Render panel analisis safety compliance (compliance bar, risk, violations, classes)."""

    # Compliance bar
    bar_color = '#16a34a' if c['compliance_pct'] >= 80 else \
                '#d97706' if c['compliance_pct'] >= 50 else '#dc2626'

    st.markdown(f"""
    <div style="margin-bottom:12px">
        <div style="font-size:13px;color:#64748b;margin-bottom:4px">Compliance Rate</div>
        <div class="compliance-bar-wrap">
            <div class="compliance-bar" style="width:{c['compliance_pct']}%;background:{bar_color}">
                {c['compliance_pct']}%
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8">
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
    </div>
    <div style="margin-bottom:14px">
        <span style="font-size:13px;color:#64748b;">Risk Level: </span>
        <span class="risk-badge {c['risk_class']}">{c['risk_level']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Violations / Safe
    if c['violations']:
        st.markdown('<div class="section-title">🚨 Pelanggaran Terdeteksi</div>',
                    unsafe_allow_html=True)
        for v in c['violations']:
            st.markdown(f'<div class="violation-box">⚠️ {v}</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="safe-box">✅ Tidak ada pelanggaran APD terdeteksi!</div>',
            unsafe_allow_html=True,
        )

    # Detail per pekerja (IoU-based)
    if c['person_status']:
        st.markdown('<div class="section-title">👷 Detail per Pekerja (IoU-matched)</div>',
                    unsafe_allow_html=True)
        for ps in c['person_status']:
            icon   = "❌" if ps['is_violating'] else "✅"
            status = "MELANGGAR" if ps['is_violating'] else "COMPLIANT"
            detail_parts = []
            if ps['has_no_helmet']: detail_parts.append("no-helmet")
            if ps['has_no_vest']:   detail_parts.append("no-vest")
            detail_str = f" · {', '.join(detail_parts)}" if detail_parts else ""
            color = "#dc2626" if ps['is_violating'] else "#15803d"
            st.markdown(
                f'<div class="person-card">'
                f'{icon} <strong>Pekerja #{ps["idx"]}</strong>'
                f'<span style="color:{color};font-weight:600"> {status}</span>'
                f'<span style="color:#94a3b8">{detail_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Deteksi per class
    st.markdown('<div class="section-title">📊 Deteksi per Class</div>',
                unsafe_allow_html=True)
    for cls_name in CLASS_NAMES:
        count   = c['counts'].get(cls_name, 0)
        color   = CLASS_COLORS_HEX[cls_name]
        avg_c   = c['avg_conf'].get(cls_name, 0)
        conf_txt = f"avg conf: {avg_c:.2f}" if count > 0 else ""
        st.markdown(f"""
        <div class="class-row">
            <div>
                <span class="class-dot" style="background:{color}"></span>
                <span>{cls_name}</span>
                <span style="font-size:11px;color:#94a3b8;margin-left:6px">{conf_txt}</span>
            </div>
            <span class="class-count" style="color:{color}">{count}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 16px">
        <div style="font-size:40px">🦺</div>
        <div style="font-size:15px;font-weight:700;color:#1e3a5f">Safety Monitor</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px">Construction PPE Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("⚙️ Detection Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90,
        value=0.40, step=0.05,
        help="Semakin rendah = lebih banyak deteksi tapi lebih banyak false positive. Disarankan 0.35-0.45 untuk mengurangi false positive."
    )
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.30, max_value=0.80,
        value=0.45, step=0.05,
        help="Threshold untuk Non-Maximum Suppression"
    )

    st.divider()
    st.subheader("🏷️ Class Legend")
    for cls_name, color in CLASS_COLORS_HEX.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px">'
            f'<div style="width:14px;height:14px;border-radius:3px;background:{color};flex-shrink:0"></div>'
            f'{cls_name}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("YOLOv8s · Construction Safety Dataset · 5 Classes")
    st.caption("IoU-based person matching untuk akurasi compliance")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# Hero banner
st.markdown("""
<div class="hero">
    <div class="hero-title">🦺 Construction Safety Monitor</div>
    <div class="hero-sub">
        Deteksi kelengkapan Alat Pelindung Diri (APD) pekerja konstruksi
        menggunakan AI — YOLOv8 Object Detection dengan IoU-based matching
    </div>
    <div class="hero-badges">
        <span class="hero-badge">🤖 YOLOv8s</span>
        <span class="hero-badge">🪖 Helm Detection</span>
        <span class="hero-badge">🦺 Vest Detection</span>
        <span class="hero-badge">📊 Compliance Score</span>
        <span class="hero-badge">⚠️ Risk Level</span>
        <span class="hero-badge">🔁 Batch Upload</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Mode Upload",
    options=["📸 Single Image", "🗂️ Multiple Images (Batch)"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════

if mode == "📸 Single Image":

    st.subheader("📤 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Upload gambar (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        help="Upload gambar area konstruksi untuk dianalisis",
        key="single_uploader",
    )

    if uploaded_file is None:
        st.info("👆 Upload gambar untuk memulai deteksi APD pekerja konstruksi.")
        st.markdown("""
        **Cara penggunaan:**
        1. Upload gambar area konstruksi menggunakan tombol di atas
        2. Atur confidence threshold di sidebar (default: 0.25)
        3. Model mendeteksi: helm, vest, dan pelanggarannya
        4. Lihat Safety Compliance Score, Risk Level & detail per pekerja
        """)
        st.stop()

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if model is None:
        st.stop()

    with st.spinner("🔍 Menjalankan deteksi..."):
        result = run_detection(image_rgb, conf=conf_threshold, iou=iou_threshold)

    # ── Metric cards ──────────────────────────────────────────────────────
    c = result
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-icon">👷</div>
            <div class="metric-value mv-blue">{c['n_persons']}</div>
            <div class="metric-label">Pekerja</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">✅</div>
            <div class="metric-value mv-green">{c['n_compliant']}</div>
            <div class="metric-label">Compliant</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⚠️</div>
            <div class="metric-value mv-red">{c['n_violations']}</div>
            <div class="metric-label">Violations</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📦</div>
            <div class="metric-value mv-amber">{c['total_objects']}</div>
            <div class="metric-label">Total Deteksi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gambar & Analisis ─────────────────────────────────────────────────
    col_img, col_analysis = st.columns([1.2, 1])

    with col_img:
        st.markdown('<div class="section-title">🖼️ Hasil Deteksi</div>',
                    unsafe_allow_html=True)
        st.image(result['annotated_img'],
                 caption="Gambar dengan bounding box deteksi",
                 use_container_width=True)

        annotated_pil = Image.fromarray(result['annotated_img'])
        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="⬇️ Download Hasil Deteksi",
            data=buf.getvalue(),
            file_name=f"detection_{uploaded_file.name}",
            mime="image/jpeg",
            use_container_width=True,
        )

    with col_analysis:
        st.markdown('<div class="section-title">🛡️ Safety Compliance</div>',
                    unsafe_allow_html=True)
        render_analysis_panel(c)

    # ── Perbandingan original vs deteksi ──────────────────────────────────
    st.markdown('<div class="section-title">🔍 Perbandingan: Original vs Deteksi</div>',
                unsafe_allow_html=True)
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.image(image_rgb, caption="Gambar Original", use_container_width=True)
    with col_det:
        st.image(result['annotated_img'], caption="Hasil Deteksi",
                 use_container_width=True)

    # ── Safety Report ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Safety Compliance Report</div>',
                unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Total Pekerja",    c['n_persons'])
        st.metric("Pakai Helm",       c['n_helmet'])
        st.metric("Tidak Pakai Helm", c['n_no_helmet'],
                  delta=f"-{c['n_no_helmet']}" if c['n_no_helmet'] > 0 else None,
                  delta_color="inverse")
    with r2:
        st.metric("Pakai Vest",       c['n_vest'])
        st.metric("Tidak Pakai Vest", c['n_no_vest'],
                  delta=f"-{c['n_no_vest']}" if c['n_no_vest'] > 0 else None,
                  delta_color="inverse")
        st.metric("Pekerja Compliant", c['n_compliant'])
    with r3:
        st.metric("Compliance Rate",    f"{c['compliance_pct']}%")
        st.metric("Risk Level",         c['risk_level'])
        st.metric("Total Objek Deteksi", c['total_objects'])

    if c['risk_level'] == 'HIGH':
        st.error("🚨 **RISIKO TINGGI** — Hentikan pekerjaan dan pastikan semua pekerja memakai APD lengkap.")
    elif c['risk_level'] == 'MEDIUM':
        st.warning("⚠️ **RISIKO SEDANG** — Ada pekerja yang tidak memakai APD lengkap. Segera lakukan pengecekan.")
    else:
        st.success("✅ **RISIKO RENDAH** — Area konstruksi dalam kondisi aman.")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — BATCH / MULTIPLE IMAGES
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.subheader("🗂️ Upload Multiple Images (Batch)")
    uploaded_files = st.file_uploader(
        "Upload beberapa gambar sekaligus (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload beberapa gambar area konstruksi untuk dianalisis sekaligus",
        key="batch_uploader",
    )

    if not uploaded_files:
        st.info("👆 Upload beberapa gambar untuk analisis batch APD.")
        st.stop()

    if model is None:
        st.stop()

    st.write(f"**{len(uploaded_files)} gambar** siap dianalisis.")

    batch_results = []
    progress_bar  = st.progress(0, text="Memproses gambar...")

    for i, uf in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(uf.read()), dtype=np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result     = run_detection(image_rgb, conf=conf_threshold, iou=iou_threshold)
        result['filename']  = uf.name
        result['image_rgb'] = image_rgb
        batch_results.append(result)
        progress_bar.progress((i + 1) / len(uploaded_files),
                              text=f"Memproses: {uf.name}")

    progress_bar.empty()

    # ── Ringkasan batch ───────────────────────────────────────────────────
    st.subheader("📊 Ringkasan Batch")

    total_persons    = sum(r['n_persons']    for r in batch_results)
    total_compliant  = sum(r['n_compliant']  for r in batch_results)
    total_violations = sum(r['n_violations'] for r in batch_results)
    avg_compliance   = sum(r['compliance_pct'] for r in batch_results) / len(batch_results)

    risk_counts = Counter(r['risk_level'] for r in batch_results)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-icon">🖼️</div>
            <div class="metric-value mv-blue">{len(batch_results)}</div>
            <div class="metric-label">Gambar Dianalisis</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">👷</div>
            <div class="metric-value mv-blue">{total_persons}</div>
            <div class="metric-label">Total Pekerja</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">✅</div>
            <div class="metric-value mv-green">{total_compliant}</div>
            <div class="metric-label">Total Compliant</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⚠️</div>
            <div class="metric-value mv-red">{total_violations}</div>
            <div class="metric-label">Total Violations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_avg, col_risk = st.columns(2)
    with col_avg:
        bar_color = '#16a34a' if avg_compliance >= 80 else \
                    '#d97706' if avg_compliance >= 50 else '#dc2626'
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <div style="font-size:13px;color:#64748b;margin-bottom:4px">
                Rata-rata Compliance Rate (semua gambar)
            </div>
            <div class="compliance-bar-wrap">
                <div class="compliance-bar"
                     style="width:{avg_compliance:.1f}%;background:{bar_color}">
                    {avg_compliance:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_risk:
        st.markdown("**Distribusi Risk Level:**")
        for level, color in [('HIGH','#dc2626'),('MEDIUM','#d97706'),('LOW','#15803d')]:
            count = risk_counts.get(level, 0)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:4px 0;font-size:13px">'
                f'<span style="color:{color};font-weight:600">{level}</span>'
                f'<span>{count} gambar</span></div>',
                unsafe_allow_html=True,
            )

    # ── Detail tiap gambar ────────────────────────────────────────────────
    st.divider()
    st.subheader("🖼️ Detail Per Gambar")

    for r in batch_results:
        with st.expander(
            f"{'🔴' if r['risk_level']=='HIGH' else '🟡' if r['risk_level']=='MEDIUM' else '🟢'} "
            f"{r['filename']}  —  compliance {r['compliance_pct']}%  |  "
            f"risk: {r['risk_level']}  |  "
            f"{r['n_persons']} pekerja, {r['n_violations']} violations",
            expanded=False
        ):
            col_img, col_ana = st.columns([1.2, 1])
            with col_img:
                st.image(r['annotated_img'],
                         caption="Hasil Deteksi",
                         use_container_width=True)

                buf = io.BytesIO()
                Image.fromarray(r['annotated_img']).save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="⬇️ Download",
                    data=buf.getvalue(),
                    file_name=f"detection_{r['filename']}",
                    mime="image/jpeg",
                    key=f"dl_{r['filename']}",
                )

            with col_ana:
                render_analysis_panel(r)
