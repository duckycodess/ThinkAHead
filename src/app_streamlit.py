#!/usr/bin/env python3

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys

# toss src on the path so imports chill
sys.path.insert(0, str(Path(__file__).parent))

from infer_yolo import load_model, YOLODetector
from analyze import analyze_frame


# basic streamlit page setup
st.set_page_config(
    page_title="ThinkAHead - Helmet Detection",
    page_icon="🏍️",
    layout="wide"
)


@st.cache_resource
def get_model():
    """Load model (cached)"""
    model_path = Path(__file__).parent.parent / 'models' / 'trained' / 'thinkahead_best.pt'
    
    if not model_path.exists():
        # try a couple other spots
        alt_paths = [
            Path('models/trained/thinkahead_best.pt'),
            Path('../models/trained/thinkahead_best.pt'),
            Path('outputs/runs/thinkahead/weights/best.pt'),
        ]
        for p in alt_paths:
            if p.exists():
                model_path = p
                break
    
    if not model_path.exists():
        st.error(f"Model not found! Please ensure the model is at: models/trained/thinkahead_best.pt")
        st.stop()
    
    return load_model(str(model_path))


def process_uploaded_image(uploaded_file, model, conf, iou):
    # pull the upload into a numpy array
    image = Image.open(uploaded_file)
    frame = np.array(image)
    
    # flip rgb to bgr because opencv is picky
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # run the detector
    annotated, stats = analyze_frame(frame, model, conf, iou, read_plates=True)
    
    # flip it back to rgb for streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return annotated_rgb, stats


def process_video_file(video_path, model, conf, iou, frame_skip, progress_bar, frame_display, stats_display):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_stats = []
    violations_log = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            annotated, stats = analyze_frame(frame, model, conf, iou, read_plates=True)
            
            # show the frame in the ui
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_display.image(annotated_rgb, caption=f"Frame {frame_idx}/{total_frames}", use_container_width=True)
            
            # stash stats
            all_stats.append(stats)
            
            # jot down violations
            for plate in stats.get('plate_readings', []):
                violations_log.append({
                    'Frame': frame_idx,
                    'Plate': plate['text'],
                    'Violations': ', '.join(plate['violation_types']),
                    'Time': datetime.now().strftime('%H:%M:%S')
                })
        
        progress_bar.progress(min(frame_idx / total_frames, 1.0))
        frame_idx += 1
    
    cap.release()
    return all_stats, violations_log


def main():
    # header section
    st.title("🏍️ ThinkAHead")
    st.markdown("**Motorcycle Helmet Violation & Overloading Detection**")
    
    # sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    # pick how we're feeding data
    input_mode = st.sidebar.radio(
        "Input Source",
        ["📷 Image", "🎬 Video", "📹 Webcam"]
    )
    
    # tweak detection knobs
    st.sidebar.subheader("Detection")
    conf_threshold = st.sidebar.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.05)
    
    if input_mode == "🎬 Video":
        frame_skip = st.sidebar.slider("Process every N frames", 1, 30, 5)
    
    # grab the model
    with st.spinner("Loading model..."):
        try:
            model = get_model()
            st.sidebar.success("✅ Model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Model error: {e}")
            st.stop()
    
    # main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📺 Detection View")
        display_area = st.empty()
    
    with col2:
        st.subheader("📊 Statistics")
        stats_area = st.empty()
        
        st.subheader("🚨 Violations")
        violations_area = st.empty()
    
    # image flow
    if input_mode == "📷 Image":
        uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded:
            with st.spinner("Processing..."):
                annotated, stats = process_uploaded_image(uploaded, model, conf_threshold, iou_threshold)
            
            display_area.image(annotated, use_container_width=True)
            
            # stats block
            with stats_area.container():
                st.metric("Compliance Rate", f"{stats['compliance_rate']:.1f}%")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Riders", stats['total_riders'])
                c2.metric("Helmeted", stats['helmeted_riders'])
                c3.metric("No Helmet", stats['unhelmeted_riders'])
                
                c4, c5 = st.columns(2)
                c4.metric("Motorcycles", stats['motorcycles'])
                c5.metric("Overloaded", stats['overloaded_motorcycles'])
            
            # violations table
            if stats['plate_readings']:
                violations_area.dataframe(
                    pd.DataFrame(stats['plate_readings']),
                    use_container_width=True
                )
            else:
                violations_area.info("No violations with readable plates")
    
    # video flow
    elif input_mode == "🎬 Video":
        uploaded = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        
        if uploaded:
            # drop upload into a temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded.read())
            
            if st.button("▶️ Process Video"):
                progress = st.progress(0)
                
                all_stats, violations = process_video_file(
                    tfile.name, model, conf_threshold, iou_threshold,
                    frame_skip, progress, display_area, stats_area
                )
                
                progress.progress(1.0)
                st.success("✅ Video processing complete!")
                
                if violations:
                    df = pd.DataFrame(violations)
                    violations_area.dataframe(df, use_container_width=True)
                    
                    # csv download button
                    csv = df.to_csv(index=False)
                    st.download_button("📥 Download Log", csv, "violations.csv", "text/csv")
    
    # webcam flow
    elif input_mode == "📹 Webcam":
        st.warning("⚠️ Webcam requires local execution")
        
        run = st.checkbox("Start Webcam")
        
        if run:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam")
            else:
                stop = st.button("⏹️ Stop")
                frame_window = st.empty()
                
                while run and not stop:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    annotated, stats = analyze_frame(frame, model, conf_threshold, iou_threshold)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    frame_window.image(annotated_rgb, use_container_width=True)
                    
                    with stats_area.container():
                        st.metric("Compliance", f"{stats['compliance_rate']:.1f}%")
                
                cap.release()
    
    # footer flex
    st.markdown("---")
    st.markdown("**ThinkAHead** - CS 176 Final Project")


if __name__ == "__main__":
    main()
