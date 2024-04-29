import base64
import pathlib

import requests
import show_img
import argparse

import creds


# Add arguments for latitude and longitude
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path", type=str, help="Path to the image to analyze."
    )
    args = parser.parse_args()
    return args

def encode_image(image_path: str):
    """Encodes the image at image_path to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_with_openai(image_path:str):
    """Analyzes the image for road conditions using OpenAI's Vision API."""
    print(f"Analyzing image for road conditions: {image_path}")
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {creds.OPENAI_API_KEY}",
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Rate the road quality on a scale from 0 to 100. Your response should be in the form {ROAD_QUALITY: N}, where N is a number between 0 and 100. If the image does not contain a road, please enter 'NO_ROAD'. If the image is indoors, please enter 'INDOOR'. Start at a score of N=100. If the pavement quality looks rough lower your score by 20. If the road contains some cracks, lower your score by 40. If the road contains a lot of cracks, lower your score by 50. If a road contains a pothole, lower your score by 80."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ],
            },
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Failed to analyze image. Response: {response.text}")
        return "Failed to analyze image."


if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    response = analyze_with_openai(image_path)
    print(response)