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
            futures = [executor.submit(process_segment_for_gif, segment, video_path) for segment in all_segments]
            enhanced_segments = [future.result() for future in concurrent.futures.as_completed(futures)]

        return enhanced_segments

    except Exception as e:
        print(f"Can't get llama: {e}")


def process_segment_for_gif(segment, video_path):
    glossy_text = segment["glossy_text"]
    gif_path = generate_gif(glossy_text, video_path)
    return {
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "original_text": segment["original_text"],
        "glossy_text": glossy_text,
        "emotion": segment["emotion"],
        "gif_path": gif_path
    }