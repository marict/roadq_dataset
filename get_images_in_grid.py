
import get_images
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_points", type=str, nargs='?',
        help="Path to list of lat/lon coordinates as a CSV file (see sample-points.csv)"
    )
    args = parser.parse_args()
    return args

def get_images_in_grid(filename):
    points = []
    stop_early = -1
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            points.append([float(row[0]), float(row[1])])
            if stop_early > 5:
                break
            elif stop_early != -1:
                stop_early += 1

    for point in points:
        print(f'Pulling images for point {point[0]} and {point[1]}')
        lat = point[0]
        lon = point[1]
        num_images = 3
        show_image = False
        get_images.get_images(lat, lon, num_images, show_image)


if __name__ == "__main__":
    args = parse_args()
    filename = './sample-points.csv'
    if args.image_points != None:
        get_images_in_grid(args.image_points)
    else:
        get_images_in_grid(filename)
    print("Done!")


