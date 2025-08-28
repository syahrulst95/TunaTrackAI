# app.py
import logging
import os
import tempfile
from pathlib import Path
import av
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

import ui

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config Streamlit ---
st.set_page_config(page_title="TunaTrackAI", layout="wide")

# --- Terapkan Custom UI ---
ui.apply_custom_style()
ui.render_header()
conf = ui.render_sidebar()

# --- Path project ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Running on: %s", DEVICE)

# --- Load Model ---
@st.cache_resource
def load_model() -> YOLO:
    try:
        model = YOLO("https://huggingface.co/syahrulst95/tuna_yolov9m/resolve/main/best_model_exp2_yolov9.pt")
        return model
    except Exception as e:
        logger.error("Gagal memuat model: %s", e)
        st.error("❌ Model gagal dimuat. Pastikan file weight tersedia.")
        raise


MODEL = load_model()


# --- Helper Function ---
def save_temp_file(uploaded_file) -> str:
    """Simpan file upload ke file sementara dan return path-nya."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def show_class_distribution(results, model: YOLO):
    """Tampilkan distribusi kelas hasil deteksi."""
    if not results.boxes:
        return

    labels, counts = np.unique(results.boxes.cls.cpu().numpy(), return_counts=True)
    st.write("📊 Distribusi per kelas:")
    for label, count in zip(labels, counts):
        class_name = model.names.get(int(label), f"Class {label}")
        st.write(f"- **{class_name}**: {count}")


# --- UI Streamlit ---
st.write("Unggah image, video, atau gunakan **camera** untuk mendeteksi spesies tuna.")
st.write(f"Hasil deteksi akan disimpan di folder: `{OUTPUT_DIR}`")

tab1, tab2, tab3 = st.tabs(["📷 Image", "🎥 Video", "📹 Live Camera"])


# --- Inference Image ---
with tab1:
    uploaded_file = st.file_uploader("Unggah image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        tmp_path = save_temp_file(uploaded_file)
        try:
            results = MODEL.predict(tmp_path, conf=conf, device=DEVICE)
            for r in results:
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                save_path = OUTPUT_DIR / f"result_{uploaded_file.name}"
                im.save(save_path)

                num_fish = len(r.boxes)
                st.success(f"🎯 Jumlah ikan tuna terdeteksi: **{num_fish}**")

                if num_fish > 0:
                    show_class_distribution(r, MODEL)

                st.image(im, caption=f"Hasil Deteksi (tersimpan di {save_path.name})", width='stretch')
        finally:
            os.unlink(tmp_path)  # hapus file sementara


# --- Inference Video ---
with tab2:
    uploaded_file = st.file_uploader("Unggah video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tmp_path = save_temp_file(uploaded_file)
        try:
            st.info("Sedang memproses video, harap tunggu...")
            progress_bar = st.progress(0)

            results = MODEL.predict(
                source=tmp_path,
                conf=conf,
                device=DEVICE,
                stream=True
            )

            cap = cv2.VideoCapture(tmp_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = str(OUTPUT_DIR / f"{Path(uploaded_file.name).stem}_result.mp4")
            out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed, last_result = 0, None

            for r in results:
                frame = r.plot()
                out.write(frame)
                processed += 1
                progress_bar.progress(min(processed / total_frames, 1.0))
                last_result = r

            cap.release()
            out.release()

            if last_result:
                num_fish = len(last_result.boxes)
                st.success(f"🎯 Jumlah ikan tuna pada frame terakhir: **{num_fish}**")
                if num_fish > 0:
                    show_class_distribution(last_result, MODEL)

            st.video(out_path)
            st.success(f"Hasil video tersimpan di {out_path}")
        finally:
            os.unlink(tmp_path)


# --- Inference Webcam ---
with tab3:
    st.info("Menggunakan webcam/camera yang terhubung untuk real-time detection")

    class YOLOCam(VideoTransformerBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            scale = min(1280 / w, 720 / h, 1.0)
            img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))

            results = MODEL.predict(img_resized, conf=conf, device=DEVICE, verbose=False, stream=True)
            img_overlay = img_resized

            for r in results:
                img_overlay = r.plot()
                num_fish = len(r.boxes)
                cv2.putText(img_overlay, f"Tuna: {num_fish}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                labels, counts = np.unique(r.boxes.cls.cpu().numpy(), return_counts=True)
                y_offset = 60
                for label, count in zip(labels, counts):
                    class_name = MODEL.names.get(int(label), f"Class {label}")
                    cv2.putText(img_overlay, f"{class_name}: {count}",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 30

            return av.VideoFrame.from_ndarray(img_overlay, format="bgr24")

    webrtc_streamer(
        key="tuna-detection",
        video_processor_factory=YOLOCam,
        media_stream_constraints={
            "video": {"frameRate": {"ideal": 15, "max": 15}, "width": 640, "height": 480},
            "audio": False
        }
    )

