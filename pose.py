# import uuid
# import requests
# from pose_format import Pose
# from pose_format.pose_visualizer import PoseVisualizer
# from PIL import Image
# import numpy as np
# import os
# import re
# import tempfile

# def fetch_pose_from_api(text):
#     url = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"
#     params = {'text': text, 'spoken': 'en', 'signed': 'ase'}
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         return response.content
#     else:
#         raise Exception(f"Failed to fetch pose data: Status code {response.status_code}, {response.text}")

# def resize_frames(frames, new_size=(256, 256)):
#     return [np.array(Image.fromarray(frame).resize(new_size, Image.Resampling.LANCZOS)) for frame in frames]

# def frames_to_gif(frames, gif_path, fps=12):
#     pil_images = [Image.fromarray(frame.astype(np.uint8)).convert("P", palette=Image.ADAPTIVE) for frame in frames]
#     pil_images[0].save(
#         gif_path,
#         save_all=True,
#         append_images=pil_images[1:],
#         format='GIF',
#         loop=0,
#         duration=int(1000 / fps)
#     )


# def generate_gif(text):
#     pose_data = fetch_pose_from_api(text)
#     pose = Pose.read(pose_data)
#     v = PoseVisualizer(pose)
#     gif_frames = [frame.astype(np.uint8) for frame in v.draw()]
#     gif_frames = [frame for i, frame in enumerate(gif_frames) if (i % 3) != 2]
#     gif_frames = resize_frames(gif_frames, new_size=(256, 256))

#     # Save to temp directory or current dir
#     gif_filename = f"{uuid.uuid4().hex}.gif"
#     gif_path = os.path.join(tempfile.gettempdir(), gif_filename)
#     frames_to_gif(gif_frames, gif_path, fps=12)

#     return gif_path

# import uuid
# import requests
# import os
# import re
# import tempfile
# import time
# from concurrent.futures import ThreadPoolExecutor

# # GPU-accelerated libraries
# import torch
# import torchvision.transforms as T
# from pose_format import Pose
# from pose_format.pose_visualizer import PoseVisualizer
# from PIL import Image
# import numpy as np

# # Check if CUDA is available
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {DEVICE}")

# def fetch_pose_from_api(text):
#     """Fetch pose data from API with error handling and retries"""
#     url = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"
#     params = {'text': text, 'spoken': 'en', 'signed': 'ase'}
    
#     max_retries = 3
#     retry_delay = 1
    
#     for attempt in range(max_retries):
#         try:
#             response = requests.get(url, params=params, timeout=30)
#             response.raise_for_status()  # Raise exception for 4XX/5XX responses
#             return response.content
#         except requests.exceptions.RequestException as e:
#             if attempt < max_retries - 1:
#                 print(f"Retry {attempt+1}/{max_retries} after error: {e}")
#                 time.sleep(retry_delay)
#                 retry_delay *= 2  # Exponential backoff
#             else:
#                 raise Exception(f"Failed to fetch pose data after {max_retries} attempts: {e}")

# def resize_frames_gpu(frames, new_size=(256, 256)):
#     """Resize frames using GPU acceleration"""
#     # Convert frames to torch tensors and move to GPU
#     tensor_frames = [torch.from_numpy(frame).permute(2, 0, 1).float().to(DEVICE) for frame in frames]
    
#     # Define resizing transform
#     resize = T.Resize(new_size, antialias=True)
    
#     # Apply resizing transform to each frame
#     resized_frames = []
#     for tensor in tensor_frames:
#         resized = resize(tensor.unsqueeze(0)).squeeze(0)
#         # Convert back to numpy and move to CPU
#         resized_frames.append(resized.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    
#     return resized_frames

# def frames_to_gif(frames, gif_path, fps=12):
#     """Convert frames to GIF with optimizations"""
#     # Convert to PIL images with adaptive palette
#     pil_images = []
    
#     # Process frames in parallel
#     def convert_frame(frame):
#         return Image.fromarray(frame.astype(np.uint8)).convert("P", palette=Image.ADAPTIVE)
    
#     with ThreadPoolExecutor() as executor:
#         pil_images = list(executor.map(convert_frame, frames))
    
#     # Optimize GIF creation
#     pil_images[0].save(
#         gif_path,
#         save_all=True,
#         append_images=pil_images[1:],
#         format='GIF',
#         loop=0,
#         duration=int(1000 / fps),
#         optimize=True  # Apply optimization
#     )

# class GPUPoseVisualizer:
#     """GPU-accelerated pose visualization wrapper"""
#     def __init__(self, pose):
#         self.pose = pose
#         self.visualizer = PoseVisualizer(pose)
    
#     def draw(self):
#         # Get frames from standard visualizer
#         frames = [frame.astype(np.uint8) for frame in self.visualizer.draw()]
        
#         # Process frames in batches if needed for large poses
#         batch_frames = []
#         for i, frame in enumerate(frames):
#             # Skip every third frame to reduce GIF size (as in original code)
#             if (i % 3) != 2:
#                 batch_frames.append(frame)
        
#         return batch_frames

# def generate_gif(text, output_size=(256, 256), fps=12):
#     """Generate a GIF from text using GPU acceleration"""
#     try:
#         start_time = time.time()
        
#         # Fetch pose data
#         print("Fetching pose data...")
#         pose_data = fetch_pose_from_api(text)
#         pose = Pose.read(pose_data)
        
#         # Generate frames using GPU-accelerated visualization
#         print("Generating frames...")
#         visualizer = GPUPoseVisualizer(pose)
#         frames = visualizer.draw()
        
#         # Resize frames using GPU
#         print("Resizing frames...")
#         resized_frames = resize_frames_gpu(frames, new_size=output_size)
        
#         # Save to temp directory or current dir
#         gif_filename = f"{uuid.uuid4().hex}.gif"
#         gif_path = os.path.join(tempfile.gettempdir(), gif_filename)
        
#         # Create GIF
#         print("Creating GIF...")
#         frames_to_gif(resized_frames, gif_path, fps=fps)
        
#         print(f"GIF generated in {time.time() - start_time:.2f} seconds")
#         return gif_path
    
#     except Exception as e:
#         print(f"Error generating GIF: {e}")
#         raise

# def batch_generate_gifs(texts, output_size=(256, 256), fps=12):
#     """Generate multiple GIFs in parallel"""
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(generate_gif, text, output_size, fps) for text in texts]
#         return [future.result() for future in futures]

# if __name__ == "__main__":
#     # Example usage
#     sample_text = "Hello, how are you?"
#     gif_path = generate_gif(sample_text)
#     print(f"GIF saved to: {gif_path}")
# import uuid
# import requests
# import os
# import re
# import tempfile
# import time
# from concurrent.futures import ProcessPoolExecutor
# from io import BytesIO

# # GPU-accelerated libraries
# import torch
# import torchvision.transforms as T
# from pose_format import Pose
# from pose_format.pose_visualizer import PoseVisualizer
# from PIL import Image
# import numpy as np

# # Device configuration
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {DEVICE}")
# def fetch_pose_from_api(text):
#     url = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"
#     params = {'text': text, 'spoken': 'en', 'signed': 'ase'}
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         return response.content
#     else:
#         raise Exception(f"Failed to fetch pose data: Status code {response.status_code}, {response.text}")

# def resize_frames_gpu(frames, new_size=(256, 256)):
#     """Resize frames using GPU in batch"""
#     # Stack all frames into a 4D tensor (N, C, H, W)
#     tensor_frames = torch.stack([
#         torch.from_numpy(f).permute(2, 0, 1).float() for f in frames
#     ]).to(DEVICE)

#     resize = T.Resize(new_size, antialias=True)
#     resized = torch.stack([resize(f.unsqueeze(0)).squeeze(0) for f in tensor_frames])

#     return [f.permute(1, 2, 0).cpu().numpy().astype(np.uint8) for f in resized]

# def frames_to_gif(frames, fps=12, save_path=None):
#     """Convert frames to an in-memory or saved GIF"""
#     pil_images = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    
#     gif_bytes = BytesIO()
#     pil_images[0].save(
#         gif_bytes,
#         save_all=True,
#         append_images=pil_images[1:],
#         format='GIF',
#         loop=0,
#         duration=int(1000 / fps),
#         optimize=True
#     )
#     gif_bytes.seek(0)

#     if save_path:
#         with open(save_path, 'wb') as f:
#             f.write(gif_bytes.read())
#         return save_path
#     else:
#         return gif_bytes  # Return in-memory GIF

# class GPUPoseVisualizer:
#     """GPU-accelerated pose visualization wrapper"""
#     def __init__(self, pose):
#         self.pose = pose
#         self.visualizer = PoseVisualizer(pose)

#     def draw(self):
#         frames = [f.astype(np.uint8) for f in self.visualizer.draw()]
#         return [frames[i] for i in range(len(frames)) if i % 3 != 2]  # Frame skipping

# def generate_gif(text, output_size=(256, 256), fps=12, save_to_disk=True):
#     """Generate a GIF from text using GPU acceleration"""
#     try:
#         start_time = time.time()
#         print(f"Processing: {text}")

#         pose_data = fetch_pose_from_api(text)
#         pose = Pose.read(pose_data)

#         visualizer = GPUPoseVisualizer(pose)
#         frames = visualizer.draw()

#         resized_frames = resize_frames_gpu(frames, new_size=output_size)

#         if save_to_disk:
#             gif_filename = f"{uuid.uuid4().hex}.gif"
#             gif_path = os.path.join(tempfile.gettempdir(), gif_filename)
#             frames_to_gif(resized_frames, fps=fps, save_path=gif_path)
#             print(f"Generated GIF in {time.time() - start_time:.2f}s → {gif_path}")
#             return gif_path
#         else:
#             gif_bytes = frames_to_gif(resized_frames, fps=fps)
#             print(f"Generated in-memory GIF in {time.time() - start_time:.2f}s")
#             return gif_bytes

#     except Exception as e:
#         print(f"Error generating GIF for '{text}': {e}")
#         raise

# def batch_generate_gifs(texts, output_size=(256, 256), fps=12, save_to_disk=True):
#     """Generate multiple GIFs in parallel using multiprocessing"""
#     with ProcessPoolExecutor() as executor:
#         futures = [
#             executor.submit(generate_gif, text, output_size, fps, save_to_disk)
#             for text in texts
#         ]
#         return [f.result() for f in futures]

# # Example usage
# if __name__ == "__main__":
#     sample_texts = [
#         "Hello, how are you?",
#         "Nice to meet you",
#         "What time is it?"
#     ]
#     gif_paths = batch_generate_gifs(sample_texts)
#     for path in gif_paths:
#         print(f"GIF saved to: {path}")

import uuid
import requests
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import torchvision.transforms as T
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from PIL import Image
import numpy as np
import ffmpeg

# Device configuration
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
    """Resize frames using GPU in batch"""
    tensor_frames = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1).float() for f in frames
    ]).to(DEVICE)

    resize = T.Resize(new_size, antialias=True)
    resized = torch.stack([resize(f.unsqueeze(0)).squeeze(0) for f in tensor_frames])

    return [f.permute(1, 2, 0).cpu().numpy().astype(np.uint8) for f in resized]

class GPUPoseVisualizer:
    """GPU-accelerated pose visualization wrapper"""
    def __init__(self, pose):
        self.pose = pose
        self.visualizer = PoseVisualizer(pose)

    def draw(self):
        frames = [f.astype(np.uint8) for f in self.visualizer.draw()]
        return [frames[i] for i in range(len(frames)) if i % 3 != 2]  # Skip every 3rd frame

def frames_to_mp4(frames, fps=12, output_path=None):
    """Convert frames to an MP4 file using ffmpeg"""
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

def generate_mp4(text, output_size=(256, 256), fps=12):
    """Generate an MP4 from text using GPU acceleration"""
    try:
        start_time = time.time()
        print(f"Processing: {text}")

        pose_data = fetch_pose_from_api(text)
        pose = Pose.read(pose_data)

        visualizer = GPUPoseVisualizer(pose)
        frames = visualizer.draw()

        resized_frames = resize_frames_gpu(frames, new_size=output_size)

        video_path = frames_to_mp4(resized_frames, fps=fps)
        print(f"Generated MP4 in {time.time() - start_time:.2f}s → {video_path}")
        return video_path

    except Exception as e:
        print(f"Error generating MP4 for '{text}': {e}")
        raise

def batch_generate_mp4s(texts, output_size=(256, 256), fps=12):
    """Generate multiple MP4s in parallel using multiprocessing"""
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(generate_mp4, text, output_size, fps)
            for text in texts
        ]
        return [f.result() for f in futures]

# Example usage
if __name__ == "__main__":
    sample_texts = [
        "Hello, how are you?",
        "Nice to meet you",
        "What time is it?"
    ]
    video_paths = batch_generate_mp4s(sample_texts)
    for path in video_paths:
        print(f"MP4 saved to: {path}")