import pandas as pd
import os
import click
import random
import sys
import pickle

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the sys.path
sys.path.insert(0, parent_dir)

import get_images
import get_predictions

import matplotlib.pyplot as plt

from score_utils import ValSegment
from osm_utils import GeoPoint

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

VALIDATION_SAMPLES_PATH = f"{CURRENT_PATH}/validation_samples.csv"
SEGMENTS_BY_PCI_PATH = f"{CURRENT_PATH}/segments_by_pci.pkl"

ALL_SEGMENTS: dict[int, ValSegment] = {}

@click.group()
def main():
    pass

@main.command("score-segments")
def score_segments():
    # lat = 47.652442
    # lon = -122.329200
    # image, metadata = get_images.get_image_and_save(
    #                 lat, lon,
    #                 heading=0,
    #                 pitch=-30,
    #                 show_image=True)

    if not os.path.exists(SEGMENTS_BY_PCI_PATH):
        df = pd.read_csv(VALIDATION_SAMPLES_PATH)
        for i, row in df.iterrows():
            segid = row["SEGID"]
            segment = ALL_SEGMENTS.get(segid, None)
            if not segment:
                segment = ValSegment(
                    id=segid,
                    points=[],
                    pci=row["PCI"],
                    headings=[]
                )
                ALL_SEGMENTS[segid] = segment
            segment.points.append(GeoPoint(row["LATITUDE"], row["LONGITUDE"]))
        print(f"Loaded segments: {len(ALL_SEGMENTS)} ")
        segments_by_length = {}
        for segment in ALL_SEGMENTS.values():
            segment.calculate_headings()
            segments = segments_by_length.setdefault(len(segment.points), [])
            segments.append(segment)

        lengths = sorted(segments_by_length.keys())
        for length in lengths:
            print(f"{length}: {len(segments_by_length[length])}")

        segments_by_pci = {}
        for length in range(3, 6):
            for segment in segments_by_length[length]:
                if segment.pci <= 0:
                    continue
                pci_group = int(segment.pci/10)
                segments = segments_by_pci.setdefault(pci_group, [])
                segments.append(segment)

        LIMIT_PER_PCI = 12
        for pci in segments_by_pci.keys():
            segments = segments_by_pci[pci]
            if len(segments) > LIMIT_PER_PCI:
                segments_by_pci[pci] = random.sample(segments, LIMIT_PER_PCI)

        # plt.figure(figsize=(10, 6))
        # plt.bar(segments_by_pci.keys(), [len(segments) for segments in segments_by_pci.values()])
        # plt.show()
        with open(SEGMENTS_BY_PCI_PATH, "wb") as f:
            pickle.dump(segments_by_pci, f)
    else:
        with open(SEGMENTS_BY_PCI_PATH, "rb") as f:
            segments_by_pci = pickle.load(f)

    for pci in segments_by_pci.keys():
        segments = segments_by_pci[pci]
        for segment in segments:
            segment.calculate_headings()
            for i, p in enumerate(segment.points[:-1]):
                image_path, metadata = get_images.get_image_and_save(
                    p.lat, p.lon,
                    heading=segment.headings[i],
                    pitch=-50,
                    show_image=True)
                if metadata['date'] < '2022':
                    continue
                print(image_path)
                plt.show()
                prediction = get_predictions.analyze_with_openai(image_path, verbose=True)
                # print(metadata)
                print()



if __name__ == "__main__":
    main()