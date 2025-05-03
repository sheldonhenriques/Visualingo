import os
import yt_dlp

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
