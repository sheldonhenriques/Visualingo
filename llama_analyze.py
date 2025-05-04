import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from video import get_transcript

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
        return output["segments"], None, None

    except Exception as e:
        print(f"Error from LLAMA: {e}")
        raise

def summarize_segments_with_html(video_path):
    segments, _, _ = summarize_segments(video_path)

    transcript_text = "\n".join([
        f"[{s['start_time']:.2f}-{s['end_time']:.2f}] {s['original_text']} ({s['emotion']})"
        for s in sorted(segments, key=lambda s: s["start_time"])
    ])

    stitched_asl_path = os.path.join("stitched_output.mp4")
    if not os.path.exists(stitched_asl_path):
        raise FileNotFoundError(f"{stitched_asl_path} not found. Make sure ASL video is generated.")

    html = generate_asl_sync_html("/file=stitched_output.mp4")
    return transcript_text, [stitched_asl_path], html

def generate_asl_sync_html(asl_video_url):
    html = f"""
    <div id="asl-container" style="text-align:center;">
        <video id="asl-video" width="256" height="256" muted playsinline style="display:block;">
            <source src="{asl_video_url}" type="video/mp4">
        </video>
        <div style="margin-top: 5px; font-size: 12px; color: #666;">ASL Animation</div>
    </div>
    <script>
    (function() {{
        const aslVideo = document.getElementById("asl-video");
        if (!aslVideo) return;

        function findMainVideo() {{
            const mainContainer = document.getElementById("main-video");
            return mainContainer ? mainContainer.querySelector("video") : null;
        }}

        function setupSync() {{
            const mainVideo = findMainVideo();
            if (!mainVideo) {{
                setTimeout(setupSync, 500);
                return;
            }}

            mainVideo.addEventListener("play", () => {{
                aslVideo.play().catch(e => console.warn("ASL autoplay blocked", e));
            }});
            mainVideo.addEventListener("pause", () => aslVideo.pause());
            mainVideo.addEventListener("seeking", () => {{
                aslVideo.currentTime = mainVideo.currentTime;
            }});
            mainVideo.addEventListener("timeupdate", () => {{
                if (Math.abs(aslVideo.currentTime - mainVideo.currentTime) > 0.1) {{
                    aslVideo.currentTime = mainVideo.currentTime;
                }}
            }});
            mainVideo.addEventListener("ended", () => aslVideo.pause());
        }}

        setupSync();
    }})();
    </script>
    """
    return html
