import requests
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from PIL import Image
import numpy as np
import os
import re

def fetch_pose_from_api(text):
    url = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"
    params = {'text': text, 'spoken': 'en', 'signed': 'ase'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to fetch pose data: Status code {response.status_code}, {response.text}")

def resize_frames(frames, new_size=(256, 256)):
    return [np.array(Image.fromarray(frame).resize(new_size, Image.Resampling.LANCZOS)) for frame in frames]

def frames_to_gif(frames, gif_path, fps=12):
    pil_images = [Image.fromarray(frame.astype(np.uint8)).convert("P", palette=Image.ADAPTIVE) for frame in frames]
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        format='GIF',
        loop=0,
        duration=int(1000 / fps)
    )

def generate_gif(text, video_path):
    try: 
        pose_data = fetch_pose_from_api(text)
        pose = Pose.read(pose_data)
        v = PoseVisualizer(pose)
        gif_frames = [frame.astype(np.uint8) for frame in v.draw()]

        gif_frames = [frame for i, frame in enumerate(gif_frames) if (i % 3) != 2]
        gif_frames = resize_frames(gif_frames, new_size=(256, 256))

        # Sanitize text for filename
        safe_text = re.sub(r'[^\w\-_\. ]', '_', text)[:50]  # limit length

        # Create gif directory if not exists
        gif_dir = os.path.splitext(video_path)[0] + "_gifs"
        os.makedirs(gif_dir, exist_ok=True)

        gif_path = os.path.join(gif_dir, f"{safe_text}.gif")
        print(gif_path)
        frames_to_gif(gif_frames, gif_path, fps=12)

    except Exception as e:
        print(f"Can't generate gif: {e}")