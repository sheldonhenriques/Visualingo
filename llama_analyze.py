import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.llama.com/compat/v1/",
    api_key=os.getenv("LLAMA_API_KEY"),
)

def load_segments_from_file(path):
    with open(path, "r") as f:
        return json.load(f)

def summarize_segments(segments):
    try:
        text = json.dumps(segments, indent=2) #" ".join([seg.get("text", "") for seg in segments])
        prompt = f"Please summarize this transcript:\n\n{text}"

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
                {"role": "user", "content": prompt},
            ],
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            stream=True,
            temperature=0.6,
            max_completion_tokens=2048,
            top_p=0.9,
            frequency_penalty=1
        )

        print("\n=== Summary ===\n")
        print(response)
        for chunk in response:
            print(chunk.choices[0].delta.content)
    except Exception as e:
        print(f"Can't get llama: {e}")
    
