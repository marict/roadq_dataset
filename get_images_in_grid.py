
import get_images
import csv


if __name__ == "__main__":
    points = []
    stop_early = -1
    with open('./sample-points.csv') as file:
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



