import streamlit as st
import os
import tempfile
import ffmpeg
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from openai import OpenAI
import re

# --- Config ---
MODEL_SIZE = "tiny"
MIN_CLIP_LENGTH = 25
MAX_CLIP_LENGTH = 60
MAX_SEGMENTS = 7

# --- UI ---
st.set_page_config(page_title="🔥 ShortsBot", layout="centered")
st.title("🔥 ShortsBot: AI-Powered Viral Clip Generator")
st.markdown("Paste a YouTube video (public). Let me find standout 25–60 sec highlights!")

youtube_url = st.text_input("📻 Paste YouTube link")
keywords_input = st.text_input("🔑 Target keywords (optional)", "summary, important, key point")
use_gpt = st.checkbox("🤖 Use GPT to detect highlights (slower but smarter)")

def download_youtube_video(url, output_dir):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(output_dir, 'video.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'user_agent': 'Mozilla/5.0'
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

def cut_clip_ffmpeg(input_path, start, end, output_path):
    ffmpeg.input(input_path, ss=start, to=end).output(
        output_path, codec='libx264', acodec='aac', loglevel='error'
    ).run(overwrite_output=True)

if st.button("🎨 Generate Shorts"):
    if not youtube_url:
        st.warning("Please paste a YouTube link first!")
        st.stop()

    temp_dir = tempfile.mkdtemp()
    try:
        download_youtube_video(youtube_url, temp_dir)
        video_file = next((f for f in os.listdir(temp_dir) if f.endswith(".mp4")), None)
        if not video_file:
            st.error("🚫 Video file not found after download.")
            st.stop()
        video_path = os.path.join(temp_dir, video_file)
    except Exception as e:
        st.error(f"🚫 Failed to download video: {e}")
        st.stop()

    st.info("🧠 Transcribing video...")
    whisper = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(video_path, beam_size=5)

    highlights = []

    if use_gpt:
        st.info("🧪 Asking GPT to find highlights...")
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        except KeyError:
            st.error("🚫 OPENAI_API_KEY missing from secrets. Set it in Streamlit → Manage App → Secrets.")
            st.stop()

        full_text = "\n".join([seg.text for seg in segments])[:6000]

        prompt = (
            "You are a video editor. Identify 3–7 highlight clips (25–60 seconds) "
            "from the transcript that are emotional, funny, or impactful. "
            "Format: '01:23 - 01:45: summary sentence'.\n\nTranscript:\n" + full_text
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            gpt_text = response.choices[0].message.content
            times = re.findall(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*:?:?\s*(.+)", gpt_text)
            def to_sec(ts): return sum(int(x) * 60**i for i, x in enumerate(reversed(ts.split(":"))))
            for start_ts, end_ts, _ in times:
                s, e = to_sec(start_ts), to_sec(end_ts)
                if MIN_CLIP_LENGTH <= e - s <= MAX_CLIP_LENGTH:
                    highlights.append((s, e))
        except Exception as e:
            st.error(f"❌ GPT failed: {e}")
            st.stop()
    else:
        keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
        for seg in segments:
            dur = seg.end - seg.start
            if any(k in seg.text.lower() for k in keywords) and MIN_CLIP_LENGTH <= dur <= MAX_CLIP_LENGTH:
                highlights.append((seg.start, seg.end))
            if len(highlights) >= MAX_SEGMENTS:
                break

    if not highlights:
        st.warning("😕 No good highlights found. Try different keywords or turn on GPT.")
        st.stop()

    st.success(f"🎉 Creating {len(highlights)} highlight clip(s)...")
    os.makedirs("viral_clips", exist_ok=True)

    for i, (start, end) in enumerate(highlights, start=1):
        out_path = f"viral_clips/short_{i}.mp4"
        cut_clip_ffmpeg(video_path, start, end, out_path)
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button(f"⬇️ Download Short {i}", f, file_name=os.path.basename(out_path))

    st.balloons()
    st.success("✅ Done! Your viral clips are ready to post!")
