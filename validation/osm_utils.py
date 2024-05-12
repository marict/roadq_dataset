import math

SPACING_METERS = 10

# Earth radius in meters
R = 6371000.0


class GeoPoint:
    def __init__(self, lat, lon):
        if lon > 0:
            print('Longitude is positive', lon)
        self.lat = lat
        self.lon = lon
    
    def to_point(self, origin_lat, origin_lon):
        # Calculate the cartesian coordinates from origin in meters
        x = math.radians(self.lon - origin_lon) * R * math.cos(math.radians(origin_lat))
        y = math.radians(self.lat - origin_lat) * R
        return Point(x, y)
    

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def to_geo_point(self, origin_lat, origin_lon):
        # Convert the origin latitude and longitude to radians
        origin_lat_rad = math.radians(origin_lat)
        origin_lon_rad = math.radians(origin_lon)

        # Calculate the latitude and longitude in radians
        lat_rad = origin_lat_rad + self.y / R
        lon_rad = origin_lon_rad + self.x / R / math.cos(origin_lat_rad)

        # Convert the latitude and longitude to degrees
        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)

        return GeoPoint(lat_deg, lon_deg)


# def p_dist(p1: GeoPoint, p2: GeoPoint):
#     # Converting degrees to radians
#     lat1, lon1, lat2, lon2 = map(math.radians, [p1.lat, p1.lon, p2.lat, p2.lon])

#     # Difference in coordinates
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1

#     # Haversine formula
#     a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

#     # Distance in kilometers
#     distance_km = R * c

#     # Convert to meters
#     return distance_km * 1000


# class Street:
#     def __init__(self, name: str):
#         self.name = name
#         self.points = []
    
#     def add_point(self, point: GeoPoint):
#         min_dist = float('inf')
#         # for p in self.points:
#         #     min_dist = min(min_dist, p_dist(p, point))
#         # if min_dist > 0.1:
#         self.points.append(point)
        
#         # else:
#         #     print('Too close', min_dist)

#     def add_segment(self, p1: GeoPoint, p2: GeoPoint):
#         d = p_dist(p1, p2)
#         n = int(d / SPACING_METERS) + 1 # every 20 meters
#         dx = (p2.lat - p1.lat) / n
#         dy = (p2.lon - p1.lon) / n
#         # self.add_point(p1)
#         for i in range(1, n):
#             self.add_point(GeoPoint(p1.lat + i * dx, p1.lon + i * dy))
#         self.add_point(p2)


class Node:
    def __init__(self, p: GeoPoint):
        self.p = p
        self.edges: list[Edge] = []
        self.names: set[str] = None
    
    def get_names(self):
        if self.names:
            return self.names
        self.names = set(e.on_street for e in self.edges)
        return self.names
    

class Edge:
    def __init__(self, n1: Node, n2: Node, geometry: any, on_street: str):
        self.n1 = n1
        self.n2 = n2
        self.on_street = on_street
        self.warning = None
        self.segments = []
        self.line: list[GeoPoint] = []
        if geometry:
            self.warning = f'No geometry in {on_street}'
            print(self.warning)

            if geometry.geom_type == 'LineString':
                self.warning = f'Invalid geometry type {geometry.geom_type}'
                lats = geometry.xy[1]
                lons = geometry.xy[0]
                for lat, lon in zip(lats, lons):
                    self.line.append(GeoPoint(lat, lon))
    
    def set_segments(self, segments: list["StreetSegment"]):
        self.segments = segments

    def get_segment(self):
        return self.segments[0] if self.segments else None
    
    def get_line(self):
        if self.line:
            return self.line
        return [self.n1.p, self.n2.p]

    def get_possible_junctions(self):
        result = []
        for r1 in self.n1.get_names():
            for r2 in self.n2.get_names():
                if r1 == r2 or r1 == self.on_street or r2 == self.on_street:
                    continue
                result.append([r1, r2])
        return result    
    
    def get_possible_keys(self):
        return [get_segment_key(self.on_street, junction) for junction in self.get_possible_junctions()]
    
    def get_samples(self, origin_lat: float, origin_lon: float, interval: float):
        line = [p.to_point(origin_lat, origin_lon) for p in self.get_line()]
        # line = self.get_line()
        # interval = interval / R * 180
        acc = interval
        result = []

        # for p in line:
        #     result.append((p, p.to_point(origin_lat, origin_lon)))

        for i in range(1, len(line)):
            p = line[i]
            p_ = line[i-1]
            dx = p.x - p_.x
            dy = p.y - p_.y
            # dx = p.lon - p_.lon
            # dy = p.lat - p_.lat
            d = math.sqrt(dx * dx + dy * dy)
            dx /= d
            dy /= d
            while acc < d:
                sample = Point(p_.x + acc * dx, p_.y + acc * dy)
                result.append((sample, sample.to_geo_point(origin_lat, origin_lon)))
                # sample = GeoPoint(p_.lat + acc * dy, p_.lon + acc * dx)
                # result.append((sample, sample.to_point(origin_lat, origin_lon)))
                acc += interval
            acc -= d

        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.scatter([p.x for p, _ in result], [p.y for p, _ in result])
        # # plt.scatter(
        # #     [p.lon for p, _ in result], [p.lat for p, _ in result], 
        # #     s=np.linspace(10, 200, len(result)), 
        # #     alpha=0.5, 
        # #     edgecolors='none')
        # plt.show()

        # plt.scatter([p.lon for _, p in result], [p.lat for _, p in result])
        # plt.show()
        
        return result


def get_segment_key(on_street: str, streets: list[str]):
    streets.sort()
    return tuple([on_street] + streets)


class StreetSegment:
    def __init__(self, pci: float, on_street: str, between: list[str]):
        self.pci = pci
        self.on_street = on_street
        self.between = between
        self.key = get_segment_key(self.on_street, self.between)
        self.text = None 
    
    def get_text(self):
        return f'PCI: {self.pci}, {self.on_street} between {"and".join(self.between)}'
