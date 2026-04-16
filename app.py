import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Object Counter AI",
    page_icon="🤖",
    layout="wide"
)

# ======================
# LANGUAGE SELECT
# ======================
lang = st.selectbox("Language / Хэл сонгох", ["Монгол", "English"])

# ======================
# TEXTS
# ======================
text = {
    "Монгол": {
        "title": "🤖 Объект Тоологч AI",
        "upload": "📤 Зураг оруулна уу",
        "total": "Нийт объект",
        "result": "📊 Үр дүн",
        "image": "🖼️ Илрүүлсэн зураг"
    },
    "English": {
        "title": "🤖 Object Counter AI",
        "upload": "📤 Upload image",
        "total": "Total objects",
        "result": "📊 Results",
        "image": "🖼️ Detected image"
    }
}

t = text[lang]

# ======================
# DESIGN
# ======================
st.markdown(f"<h1 style='text-align:center;color:#00ffcc'>{t['title']}</h1>", unsafe_allow_html=True)

# ======================
# MODEL
# ======================
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader(t["upload"], type=["jpg","png","jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    results = model(image, conf=0.25, imgsz=1280)

    r = results[0]
    boxes = r.boxes

    counts = Counter()
    output = image.copy()

    for box in boxes:
        cls = int(box.cls[0])
        name = model.names[cls]

        counts[name] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(output, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(output, name, (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

    col1, col2 = st.columns([3,1])

    # ======================
    # BIG IMAGE
    # ======================
    with col1:
        st.image(output, caption=t["image"], use_container_width=True)

    # ======================
    # STATS
    # ======================
    with col2:
        st.markdown(f"### {t['result']}")

        total = sum(counts.values())

        st.markdown(f"""
        <div style="
            background:#1c1f26;
            padding:15px;
            border-radius:15px;
            text-align:center;
            font-size:22px;
            color:white;
            margin-bottom:10px;
        ">
        {t['total']}: {total}
        </div>
        """, unsafe_allow_html=True)

        for k, v in counts.items():
            st.markdown(f"""
            <div style="
                background:#2c2f36;
                padding:10px;
                border-radius:10px;
                margin-bottom:5px;
                color:white;
            ">
            🔹 {k} : {v}
            </div>
            """, unsafe_allow_html=True)