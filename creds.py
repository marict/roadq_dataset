"""Contains code for loading in credentials, stored in credentials.json."""

import json
import pathlib


def load_credentials():
    """Loads the credentials from credentials.json."""
    credentials_path = pathlib.Path("credentials.json")
    if not credentials_path.exists():
        print(f"Credentials file not found at {credentials_path}.")
        return None
    with open(credentials_path) as file:
        return json.load(file)


credentials = load_credentials()
OPENAI_API_KEY = credentials.get("openai-api-key")
GOOGLE_API_KEY = credentials.get("google-api-key")
