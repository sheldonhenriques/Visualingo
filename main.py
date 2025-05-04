import gradio as gr
from video import download_youtube_video, get_transcript


with gr.Blocks() as demo:
    gr.Markdown("#Visualingo")

    with gr.Row():
        with gr.Column():
            youtube_url_input = gr.Textbox(label="YouTube URL")
            download_youtube_button = gr.Button("Download YouTube Video")
            transcript_button = gr.Button("Transcript the video")
        with gr.Column():
            downloaded_video_output = gr.Video(label="Downloaded Video")

    with gr.Row():
        transcript_output = gr.Textbox(label="Output")

    download_youtube_button.click(download_youtube_video, inputs=[youtube_url_input], outputs=[downloaded_video_output])
    transcript_button.click(get_transcript, inputs=[downloaded_video_output], outputs=[transcript_output])

demo.launch(share=True)