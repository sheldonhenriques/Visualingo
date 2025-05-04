import os
import json
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from pose import generate_mp4
from video import get_transcript
from pose import generate_gif
from video import describe_video_by_frames, get_transcript
from moviepy import VideoFileClip, concatenate_videoclips
import concurrent.futures

load_dotenv()

client = OpenAI(
    base_url="https://api.llama.com/compat/v1/",
    api_key=os.getenv("LLAMA_API_KEY"),
)

def summarize_segments(video_path):
    segments = get_transcript(video_path)
    try:
        text = json.dumps(segments, indent=2)
        prompt = f"""Analyze the following transcript segments. Each segment contains:
        - start_time, end_time
        - original_text
        For each segment, return:
        - glossy_text (expressive ASL-friendly rewrite)
        - dominant emotion
        {text}"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates ASL-enhanced transcripts."},
                {"role": "user", "content": prompt},
            ],
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            stream=False,
            temperature=0.4,
            top_p=1,
            max_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "segments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_time": {"type": "number"},
                                        "end_time": {"type": "number"},
                                        "original_text": {"type": "string"},
                                        "glossy_text": {"type": "string"},
                                        "emotion": {"type": "string"}
                                    },
                                    "required": ["start_time", "end_time", "original_text", "glossy_text", "emotion"]
                                }
                            }
                        },
                        "required": ["segments"]
                    }
                }
            }
        )

        output = json.loads(response.choices[0].message.content)
        all_segments = output["segments"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_segment_for_video, segment) for segment in all_segments]
            enhanced_segments = [f.result() for f in concurrent.futures.as_completed(futures)]

        enhanced_segments = sorted(enhanced_segments, key=lambda s: s["start_time"])
        video_paths = [seg["video_path"] for seg in enhanced_segments]

        # Load each GIF as a VideoFileClip
        clips = [VideoFileClip(path) for path in video_paths]

        # Concatenate them
        final_clip = concatenate_videoclips(clips, method="compose")

        # Export as a video or GIF
        final_clip.write_videofile("stitched_output.mp4")

        return enhanced_segments, final_clip

    except Exception as e:
        print(f"Error from LLAMA: {e}")
        raise

def process_segment_for_video(segment):
    glossy_text = segment["glossy_text"]
    words = glossy_text.split()

    # Limit overly long glosses
    if len(words) > 25:
        glossy_text = " ".join(words[:25]) + "..."

    duration = segment["end_time"] - segment["start_time"]

    try:
        video_path = generate_mp4(glossy_text, duration=duration)
    except Exception as e:
        print(f"[Pose API error] {e} for text: {glossy_text}")
        video_path = ""  # Optional fallback

    return {
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "original_text": segment["original_text"],
        "glossy_text": glossy_text,
        "emotion": segment["emotion"],
        "video_path": video_path
    }

def summarize_segments_with_html(video_path):
    segments, video_paths = summarize_segments(video_path)

    # Sort the transcript for chronological display
    transcript_text = "\n".join([
        f"[{s['start_time']:.2f}-{s['end_time']:.2f}] {s['original_text']} ({s['emotion']})"
        for s in sorted(segments, key=lambda s: s["start_time"])
    ])

    html = generate_asl_display_html(segments)
    return transcript_text, video_paths, html

def generate_asl_display_html(segments):
    segments = sorted(segments, key=lambda s: s["start_time"])

    for seg in sorted(segments, key=lambda s: s["start_time"]):
        if not seg["video_path"].startswith("/file="):
            seg["video_path"] = f"/file={os.path.basename(seg['video_path'])}"

    segments_json = json.dumps(segments)

    html = f"""
    <div id="asl-container" style="text-align:center;">
        <video id="asl-video" width="256" height="256" muted playsinline style="display:none;">
            <source id="asl-video-src" src="" type="video/mp4">
        </video>
        <div id="asl-text" style="margin-top: 5px; font-size: 16px; font-weight: bold;"></div>
        <div id="asl-emotion" style="margin-top: 3px; font-style: italic;"></div>
    </div>
    <script>
    const segments = {segments_json};
    const videoEl = document.getElementById("asl-video");
    const srcEl = document.getElementById("asl-video-src");
    const textDiv = document.getElementById("asl-text");
    const emotionDiv = document.getElementById("asl-emotion");

    function findMainVideo() {{
        return document.querySelector("video:not(#asl-video)");
    }}

    function sync() {{
        const main = findMainVideo();
        if (!main) {{
            setTimeout(sync, 1000);
            return;
        }}

        main.addEventListener("timeupdate", () => {{
            const t = main.currentTime;
            const seg = segments.find(s => t >= s.start_time && t <= s.end_time);

            if (seg && videoEl.dataset.segment !== String(seg.start_time)) {{
                srcEl.src = seg.video_path;
                videoEl.load();
                videoEl.dataset.segment = seg.start_time;
                videoEl.currentTime = 0;
                videoEl.play();
                videoEl.style.display = "block";
                textDiv.textContent = seg.glossy_text;
                emotionDiv.textContent = "Emotion: " + seg.emotion;
            }} else if (!seg) {{
                videoEl.style.display = "none";
                videoEl.pause();
                textDiv.textContent = "";
                emotionDiv.textContent = "";
                videoEl.dataset.segment = "";
            }}
        }});
    }}

    sync();
    </script>
    """
    return html
