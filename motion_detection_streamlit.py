import cv2
import streamlit as st
import numpy as np
from datetime import datetime
import os
import time
import io
import zipfile
import shutil

# Optional WebRTC imports (work on Streamlit Cloud with extra deps)
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

# Set page configuration
st.set_page_config(page_title="Motion Detection App", layout="wide")

# Title and description
st.title("🎥 Motion Detection with OpenCV")
st.write("Detect and visualize motion in videos or webcam feed")

# Sidebar controls
st.sidebar.header("Settings")

# File uploader or webcam selection
source_options = ["Upload a video", "Webcam (Browser)", "Use webcam"]
input_source = st.sidebar.selectbox("Select input source", source_options, index=0, key="input_source")

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
if "webcam_unavailable" not in st.session_state:
    st.session_state.webcam_unavailable = False
if "zip_buffer" not in st.session_state:
    st.session_state.zip_buffer = None
if "zip_name" not in st.session_state:
    st.session_state.zip_name = "motion_frames.zip"

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
display_every = st.sidebar.slider("Display every N frames", 1, 10, 1, 1,
                                 help="Reduce UI rendering frequency while processing all frames")
fps = st.sidebar.slider("Auto-run FPS", 1, 30, 10, 1,
                        help="Frames per second when Auto-run is enabled")
autorun = st.sidebar.toggle("Auto-run", value=False, help="Process frames continuously")

st.sidebar.subheader("Output")
auto_zip_on_end = st.sidebar.checkbox("Auto-create ZIP when capture ends", value=False)
save_annotated = st.sidebar.checkbox("Save annotated frames", value=True, help="Save frames with bounding boxes")
save_raw = st.sidebar.checkbox("Save raw frames", value=False, help="Save original frames without boxes")
zip_include = st.sidebar.selectbox(
    "ZIP includes",
    options=["Annotated", "Raw", "Both"],
    index=0,
    help="Choose which saved frames to include when creating the ZIP",
)

# Control buttons
col_a, col_b = st.sidebar.columns(2)
start_clicked = col_a.button("Start/Reset")
stop_clicked = col_b.button("Stop", help="Stops Auto-run; you can still step frames")

# Create output directories if they don't exist
os.makedirs("motion_detection_output/annotated", exist_ok=True)
os.makedirs("motion_detection_output/raw", exist_ok=True)

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
    # reset flags on each (re)initialization
    st.session_state.webcam_unavailable = False

    if input_source == "Upload a video" and video_file is not None:
        # Save uploaded file to a temporary file with progress
        temp_path = "temp_video.mp4"
        total_size = getattr(video_file, "size", None)
        progress_bar = st.sidebar.progress(0, text="Saving uploaded file...")
        bytes_written = 0
        with open(temp_path, "wb") as f:
            # Read in chunks if size is known; otherwise read all
            chunk_size = 4 * 1024 * 1024  # 4MB
            if total_size and hasattr(video_file, "read"):
                video_file.seek(0)
                while True:
                    data = video_file.read(chunk_size)
                    if not data:
                        break
                    f.write(data)
                    bytes_written += len(data)
                    if total_size:
                        progress = min(100, int(bytes_written * 100 / max(1, total_size)))
                        progress_bar.progress(progress, text=f"Saving uploaded file... {progress}%")
            else:
                f.write(video_file.read())
                bytes_written = total_size or 0
        progress_bar.progress(100, text="Upload saved")
        st.session_state.temp_video_path = temp_path
        st.session_state.cap = cv2.VideoCapture(temp_path)
        st.session_state.source_signature = ("file", getattr(video_file, "name", "uploaded"), video_file.size)
    elif input_source == "Use webcam":
        # Attempt to open default webcam. This is not supported on Streamlit Cloud.
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap or not st.session_state.cap.isOpened():
            # Mark webcam as unavailable to avoid repeated retries
            try:
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
            except Exception:
                pass
            st.session_state.cap = None
            st.session_state.webcam_unavailable = True
            st.session_state.initialized = False
            st.error("Webcam is not available in this environment. Please upload a video instead.")
            return
        st.session_state.source_signature = ("webcam", 0)
    else:
        st.session_state.cap = None

    st.session_state.prev_frame = None
    st.session_state.frame_count = 0
    st.session_state.saved_count = 0
    st.session_state.initialized = True

# Initialize/reset requested (skip for WebRTC mode)
if start_clicked and input_source != "Webcam (Browser)":
    initialize_capture()

# Stop requested simply disables Auto-run
if stop_clicked and autorun:
    # Turning off autorun by forcing rerun with toggle off
    st.session_state["Auto-run"] = False  # relies on toggle default key being its label

# Initialize automatically if not yet initialized but conditions allow (for non-WebRTC modes)
if input_source != "Webcam (Browser)":
    if not st.session_state.initialized:
        if (input_source == "Use webcam" and not st.session_state.webcam_unavailable) or (
            input_source == "Upload a video" and video_file is not None
        ):
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
        try:
            video_placeholder.image(
                rgb,
                channels="RGB",
                use_column_width=True,
                output_format="JPEG",
            )
        except Exception:
            # Transient media storage error; ignore this frame
            pass
        status_text.info(
            f"Frames processed: {st.session_state.frame_count} | Frames saved: {st.session_state.saved_count}"
        )
        return True

    frame_delta = cv2.absdiff(st.session_state.prev_frame, gray)
    thresh_img = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    thresh_img = cv2.dilate(thresh_img, None, iterations=2)

    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    # Keep a copy of the original frame for potential raw saving
    original_frame = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if motion_detected and save_frames and st.session_state.frame_count % frame_interval == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if save_annotated:
            annotated_path = f"motion_detection_output/annotated/motion_{timestamp}.jpg"
            try:
                cv2.imwrite(annotated_path, frame)
                st.session_state.saved_count += 1
            except Exception:
                pass
        if save_raw:
            raw_path = f"motion_detection_output/raw/raw_{timestamp}.jpg"
            try:
                cv2.imwrite(raw_path, original_frame)
                st.session_state.saved_count += 1
            except Exception:
                pass

    # Display only every N frames to throttle UI
    if st.session_state.frame_count % max(1, int(display_every)) == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            video_placeholder.image(
                rgb,
                channels="RGB",
                use_column_width=True,
                output_format="JPEG",
            )
        except Exception:
            # Transient media storage error; skip displaying this frame
            pass

    st.session_state.prev_frame = gray
    st.session_state.frame_count += 1
    status_text.info(
        f"Frames processed: {st.session_state.frame_count} | Frames saved: {st.session_state.saved_count}"
    )
    return True

# If using browser webcam, run WebRTC path; else use VideoCapture controls
if input_source == "Webcam (Browser)":
    st.subheader("Webcam (Browser) via WebRTC")
    if not WEBRTC_AVAILABLE:
        st.error("'streamlit-webrtc' is not installed. Please add 'streamlit-webrtc', 'av', and 'aiortc' to requirements.txt for Streamlit Cloud.")
    else:
        class MotionProcessor:
            def __init__(self, threshold_val, min_area, save_flag, interval, save_annotated_flag, save_raw_flag):
                self.threshold_val = int(threshold_val)
                self.min_area = int(min_area)
                self.save_flag = bool(save_flag)
                self.interval = int(interval)
                self.save_annotated = bool(save_annotated_flag)
                self.save_raw = bool(save_raw_flag)
                self.prev_gray = None
                self.frame_count = 0
                self.saved_count = 0

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                original = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if self.prev_gray is None:
                    self.prev_gray = gray
                    self.frame_count += 1
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

                frame_delta = cv2.absdiff(self.prev_gray, gray)
                thresh_img = cv2.threshold(frame_delta, self.threshold_val, 255, cv2.THRESH_BINARY)[1]
                thresh_img = cv2.dilate(thresh_img, None, iterations=2)

                contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                motion_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) < self.min_area:
                        continue
                    motion_detected = True
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if motion_detected and self.save_flag and self.frame_count % max(1, self.interval) == 0:
                    try:
                        os.makedirs("motion_detection_output/annotated", exist_ok=True)
                        os.makedirs("motion_detection_output/raw", exist_ok=True)
                    except Exception:
                        pass
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    if self.save_annotated:
                        annotated_path = f"motion_detection_output/annotated/motion_{timestamp}.jpg"
                        try:
                            cv2.imwrite(annotated_path, img)
                            self.saved_count += 1
                        except Exception:
                            pass
                    if self.save_raw:
                        raw_path = f"motion_detection_output/raw/raw_{timestamp}.jpg"
                        try:
                            cv2.imwrite(raw_path, original)
                            self.saved_count += 1
                        except Exception:
                            pass

                self.prev_gray = gray
                self.frame_count += 1
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        def processor_factory():
            return MotionProcessor(threshold, min_contour_area, save_frames, frame_interval, save_annotated, save_raw)

        ctx = webrtc_streamer(
            key="webrtc-motion",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=processor_factory,
        )

        st.info("Grant camera permission in your browser. Processing happens server-side; motion frames are saved to 'motion_detection_output/'.")
else:
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
if input_source != "Webcam (Browser)" and st.session_state.initialized and st.session_state.cap is not None:
    # If capture is no longer opened (end of file), release and cleanup temp
    if not st.session_state.cap.isOpened():
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        # Auto-create ZIP if requested and there are saved files
        try:
            if auto_zip_on_end and os.path.isdir("motion_detection_output"):
                base = "motion_detection_output"
                subdirs = []
                if zip_include in ("Annotated", "Both"):
                    subdirs.append("annotated")
                if zip_include in ("Raw", "Both"):
                    subdirs.append("raw")
                file_list = []
                for sub in subdirs:
                    subdir_path = os.path.join(base, sub)
                    if os.path.isdir(subdir_path):
                        file_list.extend(
                            [
                                os.path.join(subdir_path, f)
                                for f in os.listdir(subdir_path)
                                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                            ]
                        )
                if file_list:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for path in file_list:
                            # arcname include subfolder name to distinguish annotated/raw
                            rel = os.path.relpath(path, start=base)
                            try:
                                zf.write(path, arcname=rel)
                            except Exception:
                                pass
                    buf.seek(0)
                    st.session_state.zip_buffer = buf
                    st.session_state.zip_name = "motion_frames.zip"
        except Exception:
            pass
        cleanup_temp_file()
        st.session_state.initialized = False
        st.info("Capture ended. You can Start/Reset to run again.")

# Download section for saved frames
st.markdown("### Download saved motion frames")
saved_files = []
try:
    base = "motion_detection_output"
    saved_annotated = []
    saved_raw_list = []
    ann_dir = os.path.join(base, "annotated")
    raw_dir = os.path.join(base, "raw")
    if os.path.isdir(ann_dir):
        saved_annotated = [
            os.path.join(ann_dir, f)
            for f in os.listdir(ann_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    if os.path.isdir(raw_dir):
        saved_raw_list = [
            os.path.join(raw_dir, f)
            for f in os.listdir(raw_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    # Combine according to selection for manual ZIP creation
    saved_files = []
    if zip_include in ("Annotated", "Both"):
        saved_files.extend(saved_annotated)
    if zip_include in ("Raw", "Both"):
        saved_files.extend(saved_raw_list)
except Exception:
    saved_files = []

st.write(f"Saved frames - Annotated: {len(saved_annotated) if 'saved_annotated' in locals() else 0} | Raw: {len(saved_raw_list) if 'saved_raw_list' in locals() else 0}")

col_dl1, col_dl2 = st.columns([1, 3])
with col_dl1:
    make_zip = st.button("Create ZIP")

zip_buffer = None
zip_name = "motion_frames.zip"
if make_zip and saved_files:
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in saved_files:
                # Preserve subfolder (annotated/raw) in archive
                rel = os.path.relpath(path, start="motion_detection_output")
                try:
                    zf.write(path, arcname=rel)
                except Exception:
                    # skip unreadable files
                    pass
        zip_buffer.seek(0)
        st.session_state.zip_buffer = zip_buffer
        st.session_state.zip_name = zip_name
    except Exception as e:
        st.error(f"Failed to create ZIP: {e}")

if st.session_state.zip_buffer:
    st.download_button(
        label="Download motion_frames.zip",
        data=st.session_state.zip_buffer,
        file_name=st.session_state.zip_name,
        mime="application/zip",
    )

# Clear saved frames
st.markdown("#### Maintenance")
if st.button("Clear saved frames"):
    try:
        base = "motion_detection_output"
        if os.path.isdir(base):
            shutil.rmtree(base)
        os.makedirs(os.path.join(base, "annotated"), exist_ok=True)
        os.makedirs(os.path.join(base, "raw"), exist_ok=True)
        st.session_state.zip_buffer = None
        st.success("Cleared saved frames.")
    except Exception as e:
        st.error(f"Failed to clear saved frames: {e}")

# Add some instructions
st.markdown("### Instructions:")
st.markdown("""
1. Select input source (upload a video or use webcam)
2. Adjust detection sensitivity and other parameters in the sidebar
3. Click 'Start/Reset' to initialize the source
4. Enable 'Auto-run' to process continuously, or click 'Step one frame' to advance manually
5. Frames with motion will be automatically saved if enabled
""")
