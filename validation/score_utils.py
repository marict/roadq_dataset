from dataclasses import dataclass
import math

from osm_utils import GeoPoint

@dataclass
class ValSegment:
    id: int
    points: list[GeoPoint]
    headings: list[float]
    pci: float

    def calculate_headings(self):
        self.headings = []
        for i in range(1, len(self.points)):
            prev = self.points[i - 1]
            p = self.points[i]
            # p = self.points[i].to_point(prev.lat, prev.lon)
            # heading = int(math.degrees(math.atan2(p.y, p.x)) + 360 + 90) % 360
            heading = calculate_bearing(prev.lat, prev.lon, p.lat, p.lon)
            self.headings.append(heading)
        if self.headings:
            self.headings.append(self.headings[-1])
        # else:
        #     raise ValueError("No headings calculated for segment")


def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Calculate the difference in longitude
    delta_lon = lon2 - lon1

    # Calculate the bearing
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

    initial_bearing = math.atan2(x, y)

    # Convert the bearing from radians to degrees
    initial_bearing = math.degrees(initial_bearing)

    # Normalize the bearing to be within the range 0 to 360 degrees
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
