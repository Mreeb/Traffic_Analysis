import streamlit as st
from pathlib import Path

# Set the page title and layout
st.set_page_config(page_title="Video Upload and Play", layout="centered")

# App title
st.title("📹 Video Upload and Playback App")

# Video Upload Section
st.subheader("Upload a Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

# Define path to the secondary video
default_video_path = "Traffic_flow.mp4"

# Display the uploaded video
if uploaded_file:
    st.video(uploaded_file, format="video/mp4")
    st.success("Uploaded video is displayed above!")

# Display the default video
st.subheader("Play TraffiqFlow Video")
if Path(default_video_path).exists():
    st.video(default_video_path, format="video/mp4")
else:
    st.error(f"File not found: {default_video_path}")

# Footer note
st.markdown("<hr><center>Created with ❤️ by [Your Name]</center>", unsafe_allow_html=True)
