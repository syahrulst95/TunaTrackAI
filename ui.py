import streamlit as st

def apply_custom_style():
    st.markdown(
        """
        <style>
        /* --- Background & font --- */
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f4f9fb;
        }

        /* --- Header Title --- */
        .main-title {
            font-size: 38px;
            font-weight: 800;
            color: #0f4c75; /* biru laut */
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #1abc9c; /* hijau laut */
            text-align: center;
            margin-bottom: 25px;
        }

        /* --- Divider --- */
        hr {
            border: 1px solid #dceef5;
            margin: 20px 0;
        }

        /* --- Radio button styling --- */
        div[data-baseweb="radio"] > div {
            background: #ffffff;
            padding: 12px 18px;
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(15, 76, 117, 0.15);
            margin: 10px 0;
            border-left: 5px solid #1abc9c;
        }

        /* --- File uploader --- */
        .uploadedFile {
            border: 2px dashed #0f4c75 !important;
            border-radius: 12px;
            padding: 12px;
            background-color: #f9fdfd;
        }

        /* --- Success & Info box --- */
        .stAlert {
            border-radius: 10px;
        }
        .stAlert[data-baseweb="alert"] {
            background-color: #e8f8f5 !important;
            border-left: 6px solid #1abc9c;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
    st.markdown('<h1 class="main-title">🐟 TunaTrackAI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Sistem Deteksi dan Klasifikasi Ikan Tuna secara Real-Time Berbasis Kecerdasan Buatan</p>',
        unsafe_allow_html=True
    )
    st.write("<hr>", unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.title("⚙️ Pengaturan")
    st.sidebar.info(
        "Gunakan aplikasi ini untuk mendeteksi dan mengklasifikasikan "
        "spesies tuna dari **image, video, atau live camera**."
    )

    # --- Slider Confidence Threshold ---
    conf = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Atur tingkat keyakinan (confidence threshold) minimal deteksi YOLO"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 Dirancang dengan ❤️ oleh **Tim Riset**")

    return conf


