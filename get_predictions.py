import argparse
import ast
import base64
import json

import requests

import creds


# Add arguments for latitude and longitude
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image to analyze.")
    args = parser.parse_args()
    return args


def encode_image(image_path: str):
    """Encodes the image at image_path to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_vision_model_value(response: str, key: str) -> int:
    """
    Extracts a numerical value from a string output by a vision language model.

    :param response: A string that mimics a dictionary, output from a vision language model.
    :param key: The key (case-sensitive) whose corresponding value is to be extracted.
    :return: The numerical value associated with the key.
    """
    try:
        # Parse the JSON string into a dictionary
        data = json.loads(response)
        # Access and return the value for the specified key
        return data[key]
    except Exception as e:
        print(f"Failed to extract value from response. Response: {response}")
        raise e


def get_predictions(image_paths: list[str]):
    """Labels the images at the given paths using OpenAI's Vision API."""
    pcis = []
    for image_path in image_paths:
        response = analyze_with_openai(image_path)
        pci = extract_vision_model_value(response, key="ROAD_QUALITY")
        pcis.append(pci)
    return pcis


def analyze_with_openai(image_path: str, verbose=False):
    """Analyzes the image for road conditions using OpenAI's Vision API."""
    if verbose:
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
                        "text": "Rate the road quality on a scale from 0 to 100. Your response should be in the form {ROAD_QUALITY: N}, where N is a number between 0 and 100. If the image does not contain a road, please enter 'NO_ROAD'. If the image is indoors, please enter 'INDOOR'. Start at a score of N=100. If the pavement quality looks rough, lower your score by 20. If the road contains some cracks, lower your score by 30. If the road contains a lot of cracks, lower your score by 40. If a road contains a pothole, lower your score by 80.",
                    }
                ],
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
