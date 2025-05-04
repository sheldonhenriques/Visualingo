import gradio as gr
from video import download_youtube_video
from llama_analyze import summarize_segments_with_html  # New version returning transcript + HTML

with gr.Blocks() as demo:
    gr.Markdown("# Visualingo")

    with gr.Row():
        with gr.Column():
            youtube_url_input = gr.Textbox(label="YouTube URL")
            download_youtube_button = gr.Button("Download YouTube Video")
            summarize_button = gr.Button("Add ASL")
            asl_display = gr.HTML(label="ASL Display Area")
            asl_gallery = gr.Gallery(label="Generated ASL GIFs", show_label=True)

        with gr.Column():
            downloaded_video_output = gr.Video(label="Downloaded Video")

    with gr.Row():
        transcript_output = gr.Textbox(label="Output")

    download_youtube_button.click(
        download_youtube_video,
        inputs=[youtube_url_input],
        outputs=[downloaded_video_output]
    )

    summarize_button.click(
        summarize_segments_with_html,
        inputs=[downloaded_video_output],
        outputs=[transcript_output, asl_gallery, asl_display]
    )

demo.launch(share=True, allowed_paths=["C:/static/gifs"])