import cv2
import streamlit as st
import numpy as np
from datetime import datetime
import os
import time

# Set page configuration
st.set_page_config(page_title="Motion Detection App", layout="wide")

# Title and description
st.title("🎥 Motion Detection with OpenCV")
st.write("Detect and visualize motion in videos or webcam feed")

# Sidebar controls
st.sidebar.header("Settings")

# File uploader or webcam selection
input_source = st.sidebar.selectbox("Select input source", ["Upload a video", "Use webcam"], key="input_source")

# Initialize variables
video_file = None

# Session state initialization
if "cap" not in st.session_state:
    st.session_state.cap = None
if "prev_frame" not in st.session_state:
    st.session_state.prev_frame = None
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "saved_count" not in st.session_state:
    st.session_state.saved_count = 0
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "temp_video_path" not in st.session_state:
    st.session_state.temp_video_path = None
if "source_signature" not in st.session_state:
    st.session_state.source_signature = None

if input_source == "Upload a video":
    video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")
else:
    # Use webcam
    pass

# Motion detection parameters
st.sidebar.subheader("Detection Settings")
threshold = st.sidebar.slider("Sensitivity", 10, 100, 30, 5, 
                             help="Adjust the sensitivity of motion detection")
min_contour_area = st.sidebar.slider("Minimum Contour Area", 10, 1000, 20, 10,
                                    help="Minimum area for a contour to be considered as motion")
save_frames = st.sidebar.checkbox("Save frames with motion", value=True)
frame_interval = st.sidebar.slider("Save every N frames", 1, 30, 10, 1,
                                  help="Save frame interval when motion is detected")
fps = st.sidebar.slider("Auto-run FPS", 1, 30, 10, 1,
                        help="Frames per second when Auto-run is enabled")
autorun = st.sidebar.toggle("Auto-run", value=False, help="Process frames continuously")

# Control buttons
col_a, col_b = st.sidebar.columns(2)
start_clicked = col_a.button("Start/Reset")
stop_clicked = col_b.button("Stop", help="Stops Auto-run; you can still step frames")

# Create output directory if it doesn't exist
os.makedirs("motion_detection_output", exist_ok=True)

# Placeholder for the video display
video_placeholder = st.empty()
status_text = st.empty()

def cleanup_temp_file():
    if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
        try:
            os.remove(st.session_state.temp_video_path)
        except Exception:
            pass
        st.session_state.temp_video_path = None

def initialize_capture():
    # Release any prior capture
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

    cleanup_temp_file()

    if input_source == "Upload a video" and video_file is not None:
        # Save uploaded file to a temporary file
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())
        st.session_state.temp_video_path = temp_path
        st.session_state.cap = cv2.VideoCapture(temp_path)
        st.session_state.source_signature = ("file", getattr(video_file, "name", "uploaded"), video_file.size)
    elif input_source == "Use webcam":
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.source_signature = ("webcam", 0)
    else:
        st.session_state.cap = None

    st.session_state.prev_frame = None
    st.session_state.frame_count = 0
    st.session_state.saved_count = 0
    st.session_state.initialized = True

# Initialize/reset requested
if start_clicked:
    initialize_capture()

# Stop requested simply disables Auto-run
if stop_clicked and autorun:
    # Turning off autorun by forcing rerun with toggle off
    st.session_state["Auto-run"] = False  # relies on toggle default key being its label

# Initialize automatically if not yet initialized but conditions allow
if not st.session_state.initialized:
    if input_source == "Use webcam" or (input_source == "Upload a video" and video_file is not None):
        initialize_capture()

def process_one_frame():
    cap = st.session_state.cap
    if cap is None or not cap.isOpened():
        status_text.warning("No video source available. Please Start/Reset with a valid source.")
        return False

    ret, frame = cap.read()
    if not ret:
        status_text.warning("Video ended or unable to read frame")
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if st.session_state.prev_frame is None:
        st.session_state.prev_frame = gray
        # Show the first frame as-is
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, channels="RGB", width='stretch')
        status_text.info(
            f"Frames processed: {st.session_state.frame_count} | Frames saved: {st.session_state.saved_count}"
        )
        return True

    frame_delta = cv2.absdiff(st.session_state.prev_frame, gray)
    thresh_img = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    thresh_img = cv2.dilate(thresh_img, None, iterations=2)

    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if motion_detected and save_frames and st.session_state.frame_count % frame_interval == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"motion_detection_output/motion_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        st.session_state.saved_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(rgb, channels="RGB", width='stretch')

    st.session_state.prev_frame = gray
    st.session_state.frame_count += 1
    status_text.info(
        f"Frames processed: {st.session_state.frame_count} | Frames saved: {st.session_state.saved_count}"
    )
    return True

# Step/auto-run controls (no widgets inside loops)
col1, col2 = st.columns(2)
step_once = col1.button("Step one frame")
show_info = col2.checkbox("Show info messages", value=True)

if show_info:
    st.caption("Use 'Start/Reset' to initialize the selected source. 'Auto-run' will process frames continuously at the chosen FPS. 'Step one frame' advances a single frame.")

processed = False
if st.session_state.initialized:
    # Step once if requested
    if step_once:
        processed = process_one_frame()

    # Auto-run: process one frame, then schedule a rerun
    if autorun and not step_once:
        processed = process_one_frame()
        # Sleep according to FPS and rerun
        time.sleep(max(0.001, 1.0 / float(fps)))
        st.rerun()

# Cleanup when video ends or reset
if st.session_state.initialized and st.session_state.cap is not None:
    # If capture is no longer opened (end of file), release and cleanup temp
    if not st.session_state.cap.isOpened():
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        cleanup_temp_file()
        st.session_state.initialized = False
        st.info("Capture ended. You can Start/Reset to run again.")

# Add some instructions
st.markdown("### Instructions:")
st.markdown("""
1. Select input source (upload a video or use webcam)
2. Adjust detection sensitivity and other parameters in the sidebar
3. Click 'Start/Reset' to initialize the source
4. Enable 'Auto-run' to process continuously, or click 'Step one frame' to advance manually
5. Frames with motion will be automatically saved if enabled
""")
