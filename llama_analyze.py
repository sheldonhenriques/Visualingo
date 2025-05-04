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
        text = json.dumps(segments, indent=2) 
        prompt = f"""Analyze the following transcript JSON. For each segment, return:
        - start_time
        - end_time
        - text
        - sentiment (positive, neutral, or negative)

        Return in the specified JSON schema format.

        Transcript:
        {text}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes transcripts. For each segment, extract the start and end time, the text, and provide sentiment analysis (positive, neutral, or negative). Return a JSON array of segment objects."},
                {"role": "user", "content": prompt},
            ],
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            stream=False,
            temperature=0.3,
            top_p=1,
            max_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema":{
                        "type": "object",
                        "properties":{
                            "segments":{
                                "type": "array",
                                "items":{
                                    "type": "object",
                                    "properties": {
                                        "start_time": {"type": "string"},
                                        "end_time": {"type": "string"},
                                        "text": {"type": "string"},
                                        "sentiment": {
                                            "type": "string",
                                            "enum": ["positive", "neutral", "negative"]
                                        }
                                    },
                                    "required": ["start_time", "end_time", "text", "sentiment"]
                                }
                            }
                        },
                        "required": ["segments"]
                    }
                }
            }
        )

        print("\n=== Sentiment by Segment ===\n")
        message_content = response.choices[0].message.content
        print(message_content)
    except Exception as e:
        print(f"Can't get llama: {e}")
    
