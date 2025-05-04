import uuid
import requests
import os
import tempfile
import torch
import torchvision.transforms as T
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
import numpy as np
import ffmpeg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def fetch_pose_from_api(text):
    url = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"
    params = {'text': text, 'spoken': 'en', 'signed': 'ase'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to fetch pose data: Status code {response.status_code}, {response.text}")

def resize_frames_gpu(frames, new_size=(256, 256)):
    tensor_frames = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1).float() for f in frames
    ]).to(DEVICE)

    resize = T.Resize(new_size, antialias=True)
    resized = torch.stack([resize(f.unsqueeze(0)).squeeze(0) for f in tensor_frames])
    return [f.permute(1, 2, 0).cpu().numpy().astype(np.uint8) for f in resized]

def resample_frames(frames, target_count):
    if len(frames) == target_count:
        return frames
    indices = np.linspace(0, len(frames)-1, target_count).astype(int)
    return [frames[i] for i in indices]

class GPUPoseVisualizer:
    def __init__(self, pose):
        self.pose = pose
        self.visualizer = PoseVisualizer(pose)

    def draw(self):
        frames = [f.astype(np.uint8) for f in self.visualizer.draw()]
        return [frames[i] for i in range(len(frames)) if i % 3 != 2]

def frames_to_mp4(frames, fps=12, output_path=None):
    height, width, _ = frames[0].shape
    output_path = output_path or os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp4")

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
        .output(output_path, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in frames:
        process.stdin.write(frame.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()
    return output_path

def generate_mp4(text, output_size=(256, 256), fps=12, duration=None):
    pose_data = fetch_pose_from_api(text)
    pose = Pose.read(pose_data)
    visualizer = GPUPoseVisualizer(pose)
    frames = visualizer.draw()
    resized_frames = resize_frames_gpu(frames, new_size=output_size)

    if duration is not None:
        target_frame_count = max(1, int(duration * fps))
        resized_frames = resample_frames(resized_frames, target_frame_count)

    return frames_to_mp4(resized_frames, fps=fps)
