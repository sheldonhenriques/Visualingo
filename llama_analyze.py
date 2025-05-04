import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pose import generate_gif
from video import describe_video_by_frames, get_transcript

load_dotenv()

client = OpenAI(
    base_url="https://api.llama.com/compat/v1/",
    api_key=os.getenv("LLAMA_API_KEY"),
)

def load_segments_from_file(path):
    with open(path, "r") as f:
        return json.load(f)
    
def summarize_segments(video_path):
    # print(segments, video_path)
    segments = get_transcript(video_path)
    try:
        # Attach 4 base64 images per segment
        all_segments = []
        for i, segment in enumerate(segments):
            # frames = describe_video_by_frames(video_path, segment["start"], segment["end"])
            # print(frames)
            # segment["images"] = frames

            # Construct input
            text = json.dumps(segment, indent=2)
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
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
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
                }

            )

            output = json.loads(response.choices[0].message.content)
            print(output)
            print('-----------------------------')
            output['gif_path'] = generate_gif(output["glossy_text"], video_path)
            all_segments.append(output)

        print("\n=== Glossy Segments with Emotion ===\n")
        print(all_segments)

    except Exception as e:
        print(f"Can't get llama: {e}")