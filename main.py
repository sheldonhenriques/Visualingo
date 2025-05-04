import gradio as gr
import os
from video import download_youtube_video
from llama_analyze import summarize_segments_with_html

def process_youtube_video(youtube_url):
    return process_video(youtube_url=youtube_url)

def process_uploaded_video(uploaded_video):
    return process_video(uploaded_video=uploaded_video)

def process_video(youtube_url=None, uploaded_video=None):
    """Process either a YouTube URL or uploaded video"""
    if youtube_url:
        video_path = download_youtube_video(youtube_url)
    elif uploaded_video:
        video_path = uploaded_video
    else:
        return None, None, None, "Please provide either a YouTube URL or upload a video."
    
    try:
        transcript_text, gif_paths, html = summarize_segments_with_html(video_path)
        return video_path, transcript_text, gif_paths, html
    except Exception as e:
        return None, None, None, f"Error processing video: {str(e)}"

with gr.Blocks(css="""
    #asl-container { 
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .gradio-container {
        max-width: 1200px !important;
    }
""") as demo:
    gr.Markdown("# Visualingo")
    gr.Markdown("### ASL Visualization for Videos")
    
    with gr.Tabs():
        with gr.TabItem("YouTube Video"):
            with gr.Row():
                youtube_url_input = gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL")
            youtube_process_btn = gr.Button("Process YouTube Video", variant="primary")
        
        with gr.TabItem("Upload Video"):
            with gr.Row():
                uploaded_video_input = gr.Video(label="Upload Video")
            upload_process_btn = gr.Button("Process Uploaded Video", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=2):
            video_output = gr.Video(label="Video")
            asl_display = gr.HTML(label="ASL Animation", elem_id="asl-display-area")
        
        with gr.Column(scale=1):
            transcript_output = gr.Textbox(label="Transcript", lines=10)
            asl_gallery = gr.Gallery(label="All ASL Segments", columns=3, height="auto")
    
    error_output = gr.Textbox(label="Status", visible=False)
    
    # Set up event handlers
    youtube_process_btn.click(
        fn=process_youtube_video,
        inputs=[youtube_url_input],
        outputs=[video_output, transcript_output, asl_gallery, asl_display]
    )

    upload_process_btn.click(
        fn=process_uploaded_video,
        inputs=[uploaded_video_input],
        outputs=[video_output, transcript_output, asl_gallery, asl_display]
    )


    gr.Markdown("""
    ## How it works
    1. Provide a YouTube URL or upload your own video
    2. The app will analyze the speech in the video
    3. For each speech segment, it generates ASL animations
    4. The ASL animations will play synchronized with the video
    
    The system uses AI to enhance the transcription and determine appropriate emotional context.
    """)

# Launch the app
if __name__ == "__main__":
    # Ensure the allowed paths exist
    temp_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    demo.launch(share=True, allowed_paths=[temp_dir])
