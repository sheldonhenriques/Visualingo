import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pose import generate_gif
from video import describe_video_by_frames, get_transcript
import concurrent.futures

load_dotenv()

client = OpenAI(
    base_url="https://api.llama.com/compat/v1/",
    api_key=os.getenv("LLAMA_API_KEY"),
)

def load_segments_from_file(path):
    with open(path, "r") as f:
        return json.load(f)
    
def summarize_segments(video_path):
    segments = get_transcript(video_path)
    try:
        text = json.dumps(segments, indent=2)
        prompt = f"""Analyze the following transcript segments. Each segment contains:
            - start_time, end_time
            - original_text (spoken text)
            - 4 associated images (visual expression during the segment)

            For each segment:
            - Rewrite the original_text into a polished, expressive "glossy_text"
            - Identify the dominant emotion (from facial/body cues and text)
            - Use the following emotion categories: ["happy", "sad", "angry", "surprised", "confused", "excited", "bored", "nervous", "confident", "frustrated", "grateful", "disgusted", "neutral"]

            Return a JSON object with:
            - start_time
            - end_time
            - original_text
            - glossy_text
            - emotion

            Ignore >>>>>>

            Segment:
            {text}
            """

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that enhances transcript segments. For each segment, analyze the text and accompanying images to generate a 'glossy_text' (polished, expressive version of the transcript), and identify the dominant ASL-expressible emotion."
                },
                {"role": "user", "content": prompt},
            ],
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",  # Replace if needed
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
                                        "emotion": {
                                            "type": "string",
                                            "enum": [
                                                "happy", "sad", "angry", "surprised", "confused", "excited",
                                                "bored", "nervous", "confident", "frustrated", "grateful",
                                                "disgusted", "neutral"
                                            ]
                                        }
                                    },
                                    "required": [
                                        "start_time", "end_time", "original_text", "glossy_text", "emotion"
                                    ]
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
        print("\n=== Glossy Segments with Emotion ===\n")
        print(all_segments)

        # Run GIF generation in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_segment_for_gif, segment) for segment in all_segments]
            enhanced_segments = [future.result() for future in concurrent.futures.as_completed(futures)]

        gif_paths = [segment["gif_path"] for segment in enhanced_segments]

        return enhanced_segments, gif_paths

    except Exception as e:
        print(f"Can't get llama: {e}")


def process_segment_for_gif(segment):
    glossy_text = segment["glossy_text"]
    gif_path = generate_gif(glossy_text)
    gradio_url = f"/file={os.path.basename(gif_path)}"
    return {
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "original_text": segment["original_text"],
        "glossy_text": glossy_text,
        "emotion": segment["emotion"],
        "gif_path": gif_path
    }


def summarize_segments_with_html(video_path):
    """
    Process video segments and generate transcript, GIFs, and HTML for ASL display
    
    Args:
        video_path: Path to the video file
        
    Returns:
        tuple: (transcript_text, gif_paths, html_display)
    """
    segments, gif_paths = summarize_segments(video_path)  # This returns a list of dicts with start_time, end_time, gif_path, etc.
    
    # Combine all glossy text segments for full transcript
    transcript_text = "\n".join([f"[{s['start_time']:.2f}-{s['end_time']:.2f}] {s['glossy_text']} ({s['emotion']})" 
                               for s in segments])
    
    # Generate HTML with our enhanced sync function
    html = generate_asl_display_html(segments)
    
    return transcript_text, gif_paths, html


def generate_asl_display_html(segments):
    """
    Generates HTML/JS code to display ASL GIFs synchronized with video playback
    
    Args:
        segments: List of dictionaries containing start_time, end_time, and gif_path
        
    Returns:
        HTML string with embedded JavaScript for synchronization
    """
    # Ensure GIF paths are properly formatted for Gradio
    for seg in segments:
        if not seg["gif_path"].startswith("/file="):
            seg["gif_path"] = f"/file={os.path.basename(seg['gif_path'])}"

    # Convert segments to JSON to embed in the HTML
    segments_json = json.dumps(segments)
    
    html = """
    <div id="asl-container" style="text-align:center; margin-top:10px; position:relative;">
        <img id="asl-gif" src="" width="256" height="256" style="display:none; border: 3px solid #4CAF50;" />
        <div id="asl-text" style="margin-top: 5px; font-size: 16px; font-weight: bold; color: #333;"></div>
        <div id="asl-emotion" style="margin-top: 3px; font-style: italic; color: #666;"></div>
    </div>
    <script>
    // Store segments data
    const segments = """ + segments_json + """;
    let currentSegmentIndex = -1;
    
    // Function to find the video element
    function findVideoElement() {
        // Try to find the video element in various ways
        let video = document.querySelector("video");
        
        // If not found, try looking inside iframes
        if (!video) {
            const iframes = document.querySelectorAll('iframe');
            for (const iframe of iframes) {
                try {
                    video = iframe.contentDocument.querySelector('video');
                    if (video) break;
                } catch (e) {
                    // Cross-origin iframe, can't access
                    console.log("Could not access iframe content");
                }
            }
        }
        
        return video;
    }
    
    // Set up the synchronization
    function setupSync() {
        const gif = document.getElementById("asl-gif");
        const textDiv = document.getElementById("asl-text");
        const emotionDiv = document.getElementById("asl-emotion");
        const video = findVideoElement();
        
        if (!video) {
            console.error("Video element not found, retrying in 1 second...");
            setTimeout(setupSync, 1000);
            return;
        }
        
        console.log("Found video element, setting up synchronization");
        
        // Sync function that runs whenever video time updates
        function syncGifToTime() {
            const time = video.currentTime;
            
            // Find the active segment for current time
            const activeSegment = segments.find(s => 
                time >= s.start_time && time <= s.end_time
            );
            
            if (activeSegment) {
                // If we have an active segment, show the GIF
                const gifURL = activeSegment.gif_path;
                
                // Only update if segment changed
                if (gif.dataset.currentSegment !== String(activeSegment.start_time)) {
                    gif.src = gifURL;
                    gif.style.display = "inline-block";
                    gif.dataset.currentSegment = String(activeSegment.start_time);
                    
                    // Update text displays
                    textDiv.textContent = activeSegment.glossy_text;
                    emotionDiv.textContent = "Emotion: " + activeSegment.emotion;
                    
                    console.log(`Showing ASL for: ${activeSegment.glossy_text}`);
                }
            } else {
                // No active segment, hide the GIF
                gif.style.display = "none";
                textDiv.textContent = "";
                emotionDiv.textContent = "";
                gif.dataset.currentSegment = "";
            }
        }
        
        // Set up event listeners
        video.addEventListener("timeupdate", syncGifToTime);
        video.addEventListener("seeking", syncGifToTime);
        video.addEventListener("play", syncGifToTime);
        
        // Initial sync
        syncGifToTime();
        
        console.log("ASL synchronization setup complete");
    }
    
    // Start trying to set up sync immediately
    setupSync();
    
    // Also set up a backup timer in case the video loads later
    const maxRetries = 20;
    let retryCount = 0;
    const retryInterval = setInterval(() => {
        const video = findVideoElement();
        retryCount++;
        
        if (video || retryCount >= maxRetries) {
            clearInterval(retryInterval);
            if (video) setupSync();
            else console.error("Failed to find video element after multiple retries");
        }
    }, 1000);
    </script>
    """
    return html