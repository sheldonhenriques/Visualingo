import io
import os
import json
import base64
import yt_dlp
from moviepy import VideoFileClip
from PIL import Image
from groq import Groq
from requests.exceptions import RequestException
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def download_youtube_video(url, output_path='videos'):
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_template = os.path.join(output_path, '%(title)s.%(ext)s')

        ydl_opts = {
            'format': 'mp4',
            'outtmpl': output_template,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            print(f"Downloaded: {filename}")
            return filename

    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return None
    
def get_transcript(video_path):
    if not video_path:
        print("Error: No video path provided")
        return []

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    audio_path = None
    try:
        video = VideoFileClip(video_path)
        filename = os.path.splitext(video_path)[0] + ".mp3"
        video.audio.write_audiofile(filename, codec='libmp3lame')
        video.close()

        try:                

            with open(filename, "rb") as file:
                transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word","segment"],
                temperature=0.0,
                )
            
            print("Transcript secured")
            return transcription.segments #json.dumps(transcription, indent=4)
            
        
        except (RequestException, Exception) as e:
            print("transcription failed")
            return []
            
    except Exception as e:
        print(f"Can't get transcription: {e}")
        return []
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"Failed to remove temporary audio file: {e}")

def describe_video_by_frames(video_path, start, end, frame_interval=1.0):
    try:
        video = VideoFileClip(video_path)
        frames = []

        for t in np.arange(start, end, frame_interval):
            frame = video.get_frame(t)
            pil_image = Image.fromarray(frame)

            byte_arr = io.BytesIO()
            pil_image.save(byte_arr, format='JPEG')
            pil_image.save("frame_sample.jpg")
            image_bytes = byte_arr.getvalue()

            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            frame_data = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "auto"
                }
            }

            frames.append(frame_data)

        video.close()
        return frames

    except Exception as e:
        print(f"Error describing video: {e}")
        return f"Unable to generate video description. Error: {str(e)}"