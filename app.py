import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel
import os
import tempfile
import subprocess

# Function to download YouTube video using yt-dlp
def download_youtube_video(url, output_path):
    ydl_cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "-o", output_path,
        url
    ]
    subprocess.run(ydl_cmd, check=True)

# Streamlit interface
st.set_page_config(page_title="ShortsBot", layout="centered")
st.title("⚡ ShortsBot: Fast + Focused Highlights")
st.markdown("Paste a YouTube link and get 25–45 second clips around your keywords.")

keywords_input = st.text_input("🔑 Keywords (comma separated)", "important,summary,highlight")
youtube_url = st.text_input("📺 YouTube URL")
model_choice = st.selectbox("Whisper Model (smaller = faster)", ["tiny", "base", "small"], index=0)

if st.button("✨ Make My Shorts"):
    if not youtube_url:
        st.warning("Paste a YouTube link first!")
        st.stop()

    keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
    whisper_model = WhisperModel(model_choice, device="cpu", compute_type="int8")

    with st.spinner("📥 Downloading video..."):
        temp_video_path = os.path.join(tempfile.gettempdir(), "video.mp4")
        try:
            download_youtube_video(youtube_url, temp_video_path)
        except subprocess.CalledProcessError:
            st.error("🚫 Failed to download video. Is it public?")
            st.stop()

    with st.spinner("🧠 Transcribing..."):
        segments, _ = whisper_model.transcribe(temp_video_path, beam_size=5)

    # Only include segments that match keywords
    filtered = []
    for seg in segments:
        duration = seg.end - seg.start
        if any(kw in seg.text.lower() for kw in keywords):
            if 25 <= duration <= 45:  # keep clips between 25-45 seconds
                filtered.append((seg.start, seg.end, seg.text))

    if not filtered:
        st.warning("😕 No highlights found in the 25–45 sec range.")
    else:
        os.makedirs("clips", exist_ok=True)
        video = VideoFileClip(temp_video_path)
        st.success(f"✅ Found {len(filtered)} short clip(s)!")

        for i, (start, end, _) in enumerate(filtered):
            output = f"clips/clip_{i+1}.mp4"
            video.subclip(start, end).write_videofile(output, codec="libx264", audio_codec="aac", verbose=False)
            st.video(output)
            with open(output, "rb") as f:
                st.download_button(f"⬇️ Download clip_{i+1}.mp4", f, file_name=os.path.basename(output))
        video.close()
