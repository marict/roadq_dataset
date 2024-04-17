import base64
import pathlib

import requests
import show_img

import creds

def get_street_view_details(lat, lon):
    """Fetches the capture date and image URL for the closest Street View image to the given coordinates."""
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    image_url_base = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": f"{lat},{lon}",
        "key": creds.GOOGLE_API_KEY,
        "size": "600x300",  # Example size, can be adjusted
    }
    print(f"Retrieving Street View image metadata for location: {lat},{lon}")
    # Fetch metadata
    metadata_response = requests.get(metadata_url, params=params)
    if metadata_response.status_code == 200:
        metadata = metadata_response.json()
        # Construct image URL
        image_url = requests.Request("GET", image_url_base, params=params).prepare().url
        return metadata, image_url
    else:
        return "Failed to retrieve metadata", None

def download_street_view_image(metadata, image_url):
    """Downloads the Street View image from the given URL."""
    # Get filename from timestamp and lat/lon
    date = metadata["date"]
    lat = metadata["location"]["lat"]
    lon = metadata["location"]["lng"]
    filename = f"images/streetview_{date}_{lat}_{lon}.jpg"
    # Check if file already has been downloaded
    if pathlib.Path(filename).exists():
        print(f"Image already downloaded. Skipping download.")
        return filename
    # Add API key as query parameter to image_url
    image_url += f"&key={creds.GOOGLE_API_KEY}"
    # Get local filename from image_url
    print(f"Retrieving Street View image from URL: {image_url}")
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        return filename
    print("Failed to download image.")
    print(f"Response: {response.text}")
    return None


def encode_image(image_path):
    """Encodes the image at image_path to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_openai_api(image_path):
    """Analyzes the image for weather conditions using OpenAI's Vision API."""
    print(f"Analyzing image for weather conditions: {image_path}")
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
                        "text": "Whats in this image? Check for snow, rain, fog or cloud cover. The last characters of your response should be the weather conditions in the form of {'snow': [milimeter_guess], 'rain': [milimeter_guess], 'fog': [percent_visiblity_guess], 'cloud_cover':[percent_cover_guess]}.",
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
    # Not snowy.
    (lat, lon) = (47.5763831, -122.4211769)
    metadata, image_url = get_street_view_details(lat, lon)
    if image_url:
        print(f"Image capture date: {metadata['date']}")
        print(f"Image URL: {image_url}")
        # Download image
        local_url = download_street_view_image(metadata, image_url)
        if local_url:
            print("Image downloaded successfully.")
            show_img.show_image(local_url)
        else:
            exit()
    # Analyze image
    analysis = analyze_image_with_openai_api(local_url)
    print(analysis)
