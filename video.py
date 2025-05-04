import os
import json
import yt_dlp
from moviepy import VideoFileClip
from groq import Groq
from requests.exceptions import RequestException
from dotenv import load_dotenv
from llama_analyze import summarize_segments

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
            print(transcription.segments)
            summarize_segments(transcription.segments)
            print("Transcript secured")
            return json.dumps(transcription.words, indent=4)
            
        
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
