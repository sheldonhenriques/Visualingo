import gradio as gr
from video import download_youtube_video


with gr.Blocks() as demo:
    gr.Markdown("#Testing demo")

    with gr.Row():
        with gr.Column():
            youtube_url_input = gr.Textbox(label="YouTube URL")
            download_youtube_button = gr.Button("Download YouTube Video")
        with gr.Column():
            downloaded_video_output = gr.Video(label="Downloaded Video")

    download_youtube_button.click(download_youtube_video, inputs=[youtube_url_input], outputs=[downloaded_video_output])

demo.launch(share=True)