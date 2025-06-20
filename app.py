import streamlit as st
import os
import tempfile
import subprocess
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from faster_whisper import WhisperModel
from datetime import timedelta
import random

# --- CONFIG ---
MODEL_SIZE = "tiny"  # Use "base", "small" or "medium" for better accuracy
MIN_CLIP_LENGTH = 25  # seconds
MAX_CLIP_LENGTH = 60  # seconds
MAX_SEGMENTS = 7  # Max clips per video to keep things snappy
CAPTION_FONT_SIZE = 48

# --- UI ---
st.set_page_config(page_title="🔥 Viral ShortsBot", layout="centered")
st.title("🔥 ShortsBot: Your Viral Clip-Maker")
st.markdown("Turn long videos into catchy 25–60 sec shorts, ready for YouTube, Reels, or TikTok.")

youtube_url = st.text_input("Paste a YouTube link")
keywords_input = st.text_input("Enter target keywords (comma-separated)", "summary, important, amazing, key point")

def download_youtube_video(url, output_path):
    cmd = ["yt-dlp", "-f", "best[ext=mp4]", "-o", output_path, url]
    subprocess.run(cmd, check=True)

def add_captions_to_clip(video_path, start, end, text, output_path):
    video = VideoFileClip(video_path).subclip(start, end)
    caption = TextClip(txt=text, fontsize=CAPTION_FONT_SIZE, color='white', bg_color='black', size=(video.w, None), method='caption')
    caption = caption.set_duration(video.duration).set_position(('center', 'bottom'))
    final = CompositeVideoClip([video, caption])
    final.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)

def get_random_style():
    colors = ['yellow', 'cyan', 'magenta', 'green', 'red']
    return random.choice(colors)

if st.button("🎬 Generate Viral Shorts"):
    if not youtube_url:
        st.warning("Paste a YouTube URL first!")
        st.stop()

    keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
    st.info("Downloading and processing video...")
    whisper = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    temp_path = os.path.join(tempfile.gettempdir(), "shortsbot_video.mp4")
    try:
        download_youtube_video(youtube_url, temp_path)
    except Exception as e:
        st.error("Failed to download video.")
        st.stop()

    st.info("Transcribing audio (this may take a few mins)...")
    segments, _ = whisper.transcribe(temp_path, beam_size=5)

    highlights = []
    for seg in segments:
        seg_text = seg.text.strip()
        seg_duration = seg.end - seg.start
        if any(kw in seg_text.lower() for kw in keywords) and MIN_CLIP_LENGTH <= seg_duration <= MAX_CLIP_LENGTH:
            highlights.append((seg.start, seg.end, seg_text))
        if len(highlights) >= MAX_SEGMENTS:
            break

    if not highlights:
        st.warning("No good highlights found in that video with your keywords.")
        st.stop()

    st.success(f"Creating {len(highlights)} awesome clips with captions...")
    os.makedirs("viral_clips", exist_ok=True)
    for i, (start, end, text) in enumerate(highlights):
        output_file = f"viral_clips/clip_{i+1}.mp4"
        try:
            add_captions_to_clip(temp_path, start, end, text, output_file)
            st.video(output_file)
            with open(output_file, "rb") as f:
                st.download_button(f"⬇️ Download Clip {i+1}", f, file_name=os.path.basename(output_file))
        except Exception as e:
            st.error(f"Failed to create clip {i+1}")

    st.balloons()
    st.success("Done! Now upload these clips to YouTube Shorts, Insta Reels, TikTok and go viral 🔥")
