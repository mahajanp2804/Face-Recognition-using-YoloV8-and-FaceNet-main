"""Streamlit web app: recognise faces from the camera or a photo, enroll people, view attendance.

Run from the repository folder:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from facerec.attendance import HEADER
from facerec.config import load_config
from facerec.enroll import IMAGE_EXTENSIONS, clean_person_name, enroll_person
from facerec.matcher import FaceMatcher, load_embeddings
from facerec.recognize import draw_results, recognize_faces, record_attendance

st.set_page_config(page_title="Face Attendance", page_icon="🧑‍💼", layout="wide")

config = load_config()


@st.cache_resource(show_spinner="Loading YOLOv8, MTCNN and FaceNet...")
def load_models():
    from facerec.detector import FaceDetector
    from facerec.embedder import FaceEmbedder

    return FaceDetector(config.yolo_weights), FaceEmbedder()


def load_db() -> dict:
    # Re-read whenever the file changes (e.g. after enrolling someone in this app)
    mtime = config.embeddings_path.stat().st_mtime if config.embeddings_path.exists() else 0
    return _load_db_cached(str(config.embeddings_path), mtime)


@st.cache_data(show_spinner=False)
def _load_db_cached(path: str, mtime: float) -> dict:
    return load_embeddings(Path(path))


def decode_image(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


detector, embedder = load_models()
db = load_db()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Face Attendance")
    st.caption("YOLOv8 detection · MTCNN alignment · FaceNet embeddings")
    threshold = st.slider(
        "Match threshold (L2 distance)",
        min_value=0.4,
        max_value=1.2,
        value=float(config.recognition_threshold),
        step=0.05,
        help="A face matches an enrolled person when its embedding distance is below this.",
    )
    st.metric("Enrolled people", len(db))
    st.metric("Enrolled face embeddings", sum(len(v) for v in db.values()))
    st.caption(f"Device: `{embedder.device}`")
    st.caption(f"Liveness check: {'on' if config.liveness_enabled else 'off'} (config.yaml)")

matcher = FaceMatcher(db, threshold)

tab_recognize, tab_enroll, tab_attendance = st.tabs(["Recognize", "Enroll", "Attendance"])

# ------------------------------------------------------------------ recognize
with tab_recognize:
    if not db:
        st.info("Nobody is enrolled yet. Add people in the **Enroll** tab first.")

    source = st.radio("Input", ["Camera", "Upload a photo"], horizontal=True)
    if source == "Camera":
        upload = st.camera_input("Look at the camera and take a photo")
    else:
        upload = st.file_uploader("Photo", type=[e.lstrip(".") for e in IMAGE_EXTENSIONS])
    if config.liveness_enabled:
        # A single photo can't show a head turn, and an uploaded photo is exactly the spoof
        # the liveness check exists to stop, so only the live CLI may mark attendance.
        st.warning(
            "Liveness checking is on, so attendance isn't marked from photos here. "
            "Use `python -m facerec run` at the camera, which asks for a head turn."
        )
        auto_mark = False
    else:
        auto_mark = st.checkbox("Mark attendance for recognised people", value=True)

    if upload is not None:
        data = upload.getvalue()
        image = decode_image(data)
        if image is None:
            st.error("Could not read that image.")
        else:
            results = recognize_faces(image, detector, embedder, matcher)
            col_img, col_info = st.columns([3, 2])
            col_img.image(to_rgb(draw_results(image.copy(), results)), width="stretch")

            with col_info:
                if not results:
                    st.warning("No face detected.")
                else:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Name": r.name,
                                    "Distance": round(r.distance, 3),
                                    "Match": "✅" if r.known else "❌",
                                }
                                for r in results
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )

                known = [r.name for r in results if r.known]
                if auto_mark and known:
                    # Streamlit re-runs this script on every interaction; only record a photo once.
                    key = (hashlib.sha1(data).hexdigest(), round(threshold, 2))
                    marked = st.session_state.setdefault("marked", {})
                    if key not in marked:
                        marked[key] = record_attendance(known, config.attendance_file)
                    for name, new in marked[key].items():
                        if new:
                            st.success(f"Attendance marked: **{name}**")
                        else:
                            st.info(f"Already marked today: **{name}**")

# ------------------------------------------------------------------ enroll
with tab_enroll:
    st.subheader("Enroll a person")
    st.caption(
        "Upload 3–5 clear photos with only this person's face. They are saved to "
        f"`{config.faces_dir.name}/<name>/` (git-ignored) and the person's embeddings are rebuilt."
    )
    with st.form("enroll", clear_on_submit=True):
        name = st.text_input("Full name")
        photos = st.file_uploader(
            "Photos", type=[e.lstrip(".") for e in IMAGE_EXTENSIONS], accept_multiple_files=True
        )
        submitted = st.form_submit_button("Enroll", type="primary")

    if submitted:
        try:
            person = clean_person_name(name)
        except ValueError as e:
            st.error(str(e))
        else:
            if not photos:
                st.error("Add at least one photo.")
            else:
                person_dir = config.faces_dir / person
                person_dir.mkdir(parents=True, exist_ok=True)
                for photo in photos:
                    digest = hashlib.sha1(photo.getvalue()).hexdigest()[:12]
                    suffix = Path(photo.name).suffix.lower() or ".jpg"
                    (person_dir / f"upload_{digest}{suffix}").write_bytes(photo.getvalue())
                with st.spinner("Embedding faces..."):
                    n = enroll_person(person_dir, config.embeddings_path, detector, embedder)
                if n:
                    st.success(f"Enrolled **{person}** with {n} face embedding(s).")
                    st.rerun()
                else:
                    st.error("No face was found in those photos. Try clearer, front-facing ones.")

    if db:
        st.subheader("Enrolled people")
        st.dataframe(
            pd.DataFrame([{"Name": n, "Face embeddings": len(v)} for n, v in sorted(db.items())]),
            hide_index=True,
            width="stretch",
        )

# ------------------------------------------------------------------ attendance
with tab_attendance:
    path = config.attendance_file
    if not path.exists():
        st.info("No attendance recorded yet.")
    else:
        df = pd.read_csv(path)
        if list(df.columns) != HEADER or df.empty:
            st.info("No attendance recorded yet.")
        else:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["Date"] = df["Timestamp"].dt.date

            today = df[df["Date"] == date.today()]
            c1, c2, c3 = st.columns(3)
            c1.metric("Present today", today["Name"].nunique())
            c2.metric("Enrolled", len(db))
            c3.metric("Days recorded", df["Date"].nunique())

            day = st.date_input("Show day", value=df["Date"].max())
            st.dataframe(
                df[df["Date"] == day][["Name", "Timestamp"]].sort_values("Timestamp"),
                hide_index=True,
                width="stretch",
            )

            st.subheader("Attendance per day")
            per_day = df.groupby("Date")["Name"].nunique().rename("People present")
            per_day.index = per_day.index.astype(str)  # one labelled bar per day
            st.bar_chart(per_day, x_label="Date", y_label="People present")

            st.download_button(
                "Download attendance.csv",
                path.read_bytes(),
                file_name="attendance.csv",
                mime="text/csv",
            )
