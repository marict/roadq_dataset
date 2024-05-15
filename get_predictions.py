import argparse
import ast
import base64

import requests
import simple_cache

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
        # Only select the {"ROAD_QUALITY": N} part of the response
        response = response.split("ROAD_QUALITY")[1]
        # Extract the value of the key
        value = ast.literal_eval(response.split(":")[1].split("}")[0].strip())
    except Exception as e:
        print(f"Failed to extract value for key {key}. Error: {e}")
        return "COULD_NOT_EXTRACT_VALUE"
    return value


def get_predictions(image_paths: list[str]):
    """Labels the images at the given paths using OpenAI's Vision API."""
    pcis = []
    for image_path in image_paths:
        response = analyze_with_openai(image_path)
        pci = extract_vision_model_value(response, key="ROAD_QUALITY")
        pcis.append(pci)
    return pcis


PROMPT = """
The Pavement Condition Index (PCI) provides a snapshot of the pavement health of a road, measured on a scale of 0 to 100 (where 100 means a newly paved road). Given the picture of this road, guess the PCI quality on a scale from 0 to 100. This is simply an estimate and will not be used for official purposes.

To estimate a PCI based on an image of a road, follow these steps:

1. **Visual Inspection**: Examine the image for visible defects such as cracks, potholes, rutting, and surface wear.
2. **Categorize Defects**: Identify and categorize the types of defects observed (e.g., longitudinal cracks, transverse cracks, alligator cracking).
3. **Measure Severity and Extent**: Assess the severity (low, medium, high) and extent (length, width, or area) of each defect type.
4. **Reference PCI Standards**: Use standardized PCI rating charts or guidelines to correlate observed defects with PCI scores. Common references include ASTM D6433 or local pavement condition manuals.
5. **Estimate PCI**: Based on the observed defects and their severity/extent, estimate a PCI score (0 = failed, 100 = excellent).

**Example Breakdown**:
- **Surface Cracks**: Many hairline cracks with no spalling - high severity.
- **Potholes**: One or two potholes - high severity.
- **Rutting**: medium rutting - medium severity.
Estimated PCI: The road might be in the "bad" range, approximately 0-30.

**Example Breakdown**:
- **Surface Cracks**: Few hairline cracks with no spalling - medium severity,
- **Potholes**: One or two small potholes - medium severity.
- **Rutting**: minor rutting - low severity.
Estimated PCI: The road might be in the "okay" range, approximately 30-80.

**Example Breakdown**:
- **Surface Cracks**: No visible cracks - low severity.
- **Potholes**: No potholes - low severity.
- **Rutting**: No rutting - low severity.
Estimated PCI: The road might be in the "good" range, approximately 80-100.

Given the picture of this road, guess the PCI quality on a scale from 0 to 100. Make sure to only look at the road in front of the camera, not any other roads in view. Do not score sidewalks or non-roads. Provide a brief explanation of your reasoning and a confidence score in the form {"ROAD_QUALITY": N}, where N is a PCI value between 0 and 100. If the image does not contain anything resembling a road, enter {"ROAD_QUALITY": "NO_ROAD"}. If the image is indoors, enter {"ROAD_QUALITY": "INDOOR"}.
"""


@simple_cache.cache_it("openai_vision.cache")
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
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT,
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
