"""Contains code for showing images in the terminal."""
import subprocess
import pathlib
import os
import pathlib
import argparse
import pathlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path", type=pathlib.Path, help="Path to the image file."
    )
    args = parser.parse_args()
    return args

def can_use_imgcat():
    """Check if imgcat is available."""
    # Check if the imgcat.sh file is at root
    if not pathlib.Path("imgcat.sh").exists():
        print("imgcat.sh is not available.")
        return False
    # Check if we are using iterm2
    if 'TERM_SESSION_ID' not in os.environ:
        return False
    return True

def show_image(image_path: pathlib.Path):
    """Show an image in the terminal."""
    if not can_use_imgcat():
        print("imgcat is not available.")
        return
    subprocess.run(["./imgcat.sh", str(image_path)])


if __name__ == "__main__":
    args = parse_args()
    show_image(args.image_path)
