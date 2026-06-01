import streamlit as st
import cv2
import time
import os

from modules.face_detection import FaceDetector
from modules.eye_detection import EyeDetector
from modules.gaze_estimation import GazeEstimator
from modules.analytics import AttentionAnalytics
from modules.attention_graph import AttentionGraph
from modules.attention_heatmap import AttentionHeatmap

# PAGE CONFIG

st.set_page_config(
    page_title="AI Attention Monitoring",
    layout="wide"
)

os.makedirs("outputs", exist_ok=True)

# CUSTOM CSS

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #050816,
        #0f172a,
        #111827
    );
    color: white;
}

.main-title {
    font-size: 54px;
    font-weight: 800;
    color: white;
    text-align: center;
    margin-top: 10px;
}

.subtitle {
    font-size: 20px;
    color: #b0b0b0;
    text-align: center;
    margin-bottom: 40px;
}

section[data-testid="stSidebar"] {
    background-color: #0b1120;
}

.metric-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 20px rgba(0,255,255,0.08);
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# TITLE

st.markdown(
    '<p class="main-title">AI Attention Monitoring System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Real-Time Eye Gaze Tracking and Attention Analytics</p>',
    unsafe_allow_html=True
)

# SIDEBAR

st.sidebar.title("🎛 Session Controls")

start = st.sidebar.button("▶ Start Session")

stop = st.sidebar.button("⏹ End Session")


# SESSION STATES

if "running" not in st.session_state:
    st.session_state.running = False

if "session_ended" not in st.session_state:
    st.session_state.session_ended = False

if "final_stats" not in st.session_state:
    st.session_state.final_stats = None

# INITIALIZE MODULES

if "face_detector" not in st.session_state:
    st.session_state.face_detector = FaceDetector()

if "eye_detector" not in st.session_state:
    st.session_state.eye_detector = EyeDetector()

if "gaze_estimator" not in st.session_state:
    st.session_state.gaze_estimator = GazeEstimator()

if "analytics" not in st.session_state:
    st.session_state.analytics = AttentionAnalytics()

if "attention_graph" not in st.session_state:
    st.session_state.attention_graph = AttentionGraph()

if "attention_heatmap" not in st.session_state:
    st.session_state.attention_heatmap = AttentionHeatmap()

# START SESSION

if start:

    st.session_state.running = True

    st.session_state.session_ended = False

    st.session_state.final_stats = None

    st.session_state.analytics = AttentionAnalytics()

    st.session_state.attention_graph = AttentionGraph()

    st.session_state.attention_heatmap = AttentionHeatmap()

# -----------------------------------
# LOAD OBJECTS
# -----------------------------------

face_detector = st.session_state.face_detector

eye_detector = st.session_state.eye_detector

gaze_estimator = st.session_state.gaze_estimator

analytics = st.session_state.analytics

attention_graph = st.session_state.attention_graph

attention_heatmap = st.session_state.attention_heatmap

# LAYOUT

left, right = st.columns([2.5, 1])

with left:

    st.markdown("""
    <div style="
    padding:20px;
    border-radius:20px;
    background: linear-gradient(
        135deg,
        rgba(15,23,42,0.95),
        rgba(17,24,39,0.95)
    );
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:20px;
    box-shadow:0px 0px 25px rgba(0,255,255,0.08);
    ">
    <h1 style="
    color:white;
    margin-bottom:10px;
    ">
    🎥 Live AI Monitoring
    </h1>

    <p style="
    color:#9ca3af;
    font-size:18px;
    ">
    Real-Time Eye Tracking + Attention Detection
    </p>
    </div>
    """, unsafe_allow_html=True)

    webcam_container = st.empty()

with right:

    st.markdown("## 📊 Live Analytics")

    gaze_box = st.empty()

    attention_box = st.empty()

    attentive_box = st.empty()

    distracted_box = st.empty()

    session_box = st.empty()


# MAIN SESSION

if st.session_state.running:

    cap = cv2.VideoCapture(0)

    start_time = time.time()
    last_graph_update = time.time()

    while st.session_state.running:

        # STOP BUTTON

        if stop:

            st.session_state.running = False

            st.session_state.session_ended = True

            attention_graph.save_graph(
                "outputs/attention_graph.png"
            )

            attention_heatmap.save_heatmap(
                "outputs/final_heatmap.png"
            )

            st.session_state.final_stats = (
                analytics.get_statistics()
            )

            break

        # READ FRAME
        

        ret, frame = cap.read()

        if not ret:

            st.error("Unable to access webcam.")

            break

        frame = cv2.flip(frame, 1)

        
        # FACE DETECTION

        faces = face_detector.detect_faces(frame)

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        
        # EYE DETECTION
        

        eye_data = eye_detector.get_eye_data(frame)

        gaze = "NO FACE"

        attention = "DISTRACTED"

        if eye_data:

            cv2.circle(
                frame,
                eye_data["left_center"],
                3,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                eye_data["right_center"],
                3,
                (0, 0, 255),
                -1
            )

            gaze = gaze_estimator.estimate_gaze(
                eye_data
            )

            if gaze == "CENTER":

                attention = "ATTENTIVE"

            else:

                attention = "DISTRACTED"

            # UPDATE ANALYTICS
            analytics.update(attention)

            current_time = time.time()
            if current_time - last_graph_update > 0.3:

                attention_heatmap.update(gaze)
                last_graph_update = current_time

        # GET STATS
        

        stats = analytics.get_statistics()

        
        # TEXT ON FRAME

        cv2.putText(
            frame,
            f"Gaze: {gaze}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Attention: {attention}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # RGB CONVERSION

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # SHOW FRAME
        
        webcam_container.image(
            frame,
            channels="RGB",
            width="stretch"
        )
        elapsed = int(
            time.time() - start_time
        )

        st.markdown(f"""
        <div style="
        margin-top:15px;
        padding:18px;
        border-radius:18px;
        background:rgba(255,255,255,0.05);
        border:1px solid rgba(255,255,255,0.08);
        display:flex;
        justify-content:space-between;
        align-items:center;
        box-shadow:0px 0px 15px rgba(0,255,255,0.06);
        ">

        <div>
        <h3 style="color:white;">
        🧠 Current Status
        </h3>

        <p style="
        font-size:20px;
        font-weight:bold;
        color:{
            '#00ff99'
            if attention == 'ATTENTIVE'
            else '#ff4444'
        };
        ">
        {attention}
        </p>
        </div>

        <div>
        <h3 style="color:white;">
        👀 Eye Direction
        </h3>

        <p style="
        font-size:20px;
        font-weight:bold;
        color:#00ccff;
        ">
        {gaze}
        </p>
        </div>

        <div>
        <h3 style="color:white;">
        ⏱ Runtime
        </h3>

        <p style="
        font-size:20px;
        font-weight:bold;
        color:white;
        ">
        {elapsed}s
        </p>
        </div>

        </div>
        """, unsafe_allow_html=True)

        # LIVE ANALYTICS

        gaze_box.markdown(f"""
        <div class="metric-box">
        <h2>👀 Gaze: {gaze}</h2>
        </div>
        """, unsafe_allow_html=True)

        attention_box.markdown(f"""
        <div class="metric-box">
        <h2>🧠 Status: {attention}</h2>
        </div>
        """, unsafe_allow_html=True)

        attentive_box.markdown(f"""
        <div class="metric-box">
        <h3>✅ Attention %:
        {stats['attentive_percent']}%</h3>
        </div>
        """, unsafe_allow_html=True)

        distracted_box.markdown(f"""
        <div class="metric-box">
        <h3>❌ Distracted %:
        {stats['distracted_percent']}%</h3>
        </div>
        """, unsafe_allow_html=True)

        elapsed = int(
            time.time() - start_time
        )

        session_box.markdown(f"""
        <div class="metric-box">
        <h3>⏱ Session Time:
        {elapsed}s</h3>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.final_stats = stats

        time.sleep(0.03)

    cap.release()

# DEFAULT SCREEN

else:

    webcam_container.markdown("""
    <div style="
    padding:40px;
    border-radius:25px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    text-align:center;
    box-shadow:0px 0px 20px rgba(0,255,255,0.08);
    ">

    <h1 style="
    font-size:40px;
    color:white;
    margin-bottom:20px;
    ">
    🎥 AI Monitoring Ready
    </h1>

    <p style="
    font-size:22px;
    color:#9ca3af;
    margin-bottom:25px;
    ">
    Start a session to begin real-time
    eye gaze tracking and attention analysis
    </p>

    <div style="
    padding:18px;
    border-radius:15px;
    background:rgba(0,255,150,0.1);
    color:#00ff99;
    font-size:20px;
    font-weight:bold;
    ">
    ✅ System Ready
    </div>

    </div>
    """, unsafe_allow_html=True)

# FINAL REPORT

if st.session_state.session_ended:

    stats = st.session_state.final_stats

    attention = stats["attentive_percent"]
    distracted = stats["distracted_percent"]
    duration = stats["session_duration"]

    # FINAL SCORE LOGIC

    if attention >= 85:

        final_score = "EXCELLENT"
        score_color = "#00ff99"

    elif attention >= 70:

        final_score = "GOOD"
        score_color = "#00ccff"

    elif attention >= 50:

        final_score = "AVERAGE"
        score_color = "#ffaa00"

    else:

        final_score = "POOR"
        score_color = "#ff4444"

    # =====================================
    # MOSTLY STATUS
    # =====================================

    if attention > distracted:

        mostly_status = "MOSTLY ATTENTIVE"
        mostly_color = "#00ff99"

    else:

        mostly_status = "MOSTLY DISTRACTED"
        mostly_color = "#ff4444"

    st.markdown("---")

    
    # HEADER

    st.markdown("""
    <div style="
    padding:25px;
    border-radius:20px;
    background: linear-gradient(135deg,#0f172a,#111827);
    border:1px solid #1e293b;
    margin-top:20px;
    margin-bottom:30px;
    box-shadow:0px 0px 25px rgba(0,255,150,0.15);
    ">
    <h1 style='color:white;'>📋 Final Session Report</h1>
    </div>
    """, unsafe_allow_html=True)

    # METRIC CARDS

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div style="
        background:#111827;
        padding:25px;
        border-radius:20px;
        text-align:center;
        border:1px solid #1f2937;
        box-shadow:0px 0px 15px rgba(0,255,150,0.1);
        ">
        <h3 style='color:#9ca3af;'>Attention %</h3>
        <h1 style='color:#00ff99;font-size:48px;'>
        {attention}%
        </h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div style="
        background:#111827;
        padding:25px;
        border-radius:20px;
        text-align:center;
        border:1px solid #1f2937;
        box-shadow:0px 0px 15px rgba(255,0,0,0.1);
        ">
        <h3 style='color:#9ca3af;'>Distracted %</h3>
        <h1 style='color:#ff4444;font-size:48px;'>
        {distracted}%
        </h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div style="
        background:#111827;
        padding:25px;
        border-radius:20px;
        text-align:center;
        border:1px solid #1f2937;
        ">
        <h3 style='color:#9ca3af;'>Session Time</h3>
        <h1 style='color:white;font-size:48px;'>
        {duration}s
        </h1>
        </div>
        """, unsafe_allow_html=True)

    # FINAL SCORE

    st.markdown(f"""
    <div style="
    margin-top:30px;
    padding:25px;
    border-radius:20px;
    background:rgba(15,23,42,0.95);
    border:2px solid {score_color};
    color:{score_color};
    text-align:center;
    font-size:34px;
    font-weight:bold;
    box-shadow:0px 0px 30px {score_color};
    ">
    🏆 FINAL ATTENTION SCORE: {final_score}
    </div>
    """, unsafe_allow_html=True)

    # MOSTLY STATUS

    st.markdown(f"""
    <div style="
    margin-top:20px;
    padding:20px;
    border-radius:18px;
    background:#111827;
    border-left:8px solid {mostly_color};
    color:white;
    font-size:24px;
    font-weight:bold;
    ">
    📌 USER WAS {mostly_status}
    </div>
    """, unsafe_allow_html=True)

    # TIMELINE

    st.markdown("## 📈 Attention Timeline")

    if os.path.exists("outputs/attention_graph.png"):

        st.image(
            "outputs/attention_graph.png",
            width="stretch"
        )

    else:

        st.warning(
            "Attention graph not generated."
        )

    # HEATMAP

    st.markdown("## 🔥 Attention Heatmap")

    if os.path.exists("outputs/final_heatmap.png"):

        st.image(
            "outputs/final_heatmap.png",
            width="stretch"
        )

    else:

        st.warning(
            "Heatmap not generated."
        )