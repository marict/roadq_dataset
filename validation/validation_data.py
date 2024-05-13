import glob
import os
import re
import sys
from datetime import datetime

import click
import folium
import osmnx as ox
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from osm_utils import Edge, GeoPoint, Node, StreetSegment, get_segment_key

import plot

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

OSM_DIR = f"{CURRENT_PATH}/osm_data"
MAPS_DIR = f"{CURRENT_PATH}/maps"

SEATTLE_STREET_CSV_ORIGINAL = f"{CURRENT_PATH}/seattle_streets.csv"
SEATTLE_STREET_CSV_PREPARED = f"{CURRENT_PATH}/seattle_streets_min.csv"

SAMPLE_MAP_FILE = f"{MAPS_DIR}/sample_map.html"
QUALITY_MAP_FILE = f"{MAPS_DIR}/quality_map.html"
EDGES_FILE = f"{CURRENT_PATH}/edges.pkl"
VALIDATION_SAMPLES = f"{CURRENT_PATH}/validation_samples.csv"

EAST = -122.276137
WEST = -122.374446
NORTH = 47.649695
SOUTH = 47.594825

NODES: dict[int, Node] = {}
EDGES: list[Edge] = []
SAMPLE_EDGES: dict[tuple[str, str, str], Edge] = {}
STREET_SEGMENTS: dict[str, StreetSegment] = {}
STREET_NAMES = set()

MAPPED_NAMES = {
    "East Terrace Street": "E TERRACE ST",
    "Sound View Terrace West": "SOUND VIEW TER W",
    "South Lane Street": "S LANE ST",
    "West Montlake Place East": "WEST MONTLAKE PL E",
    "East Montlake Place East": "EAST MONTLAKE PL E",
    "East North Street": "E NORTH ST",
    "West Laurelhurst Drive Northeast": "WEST LAURELHURST DR NE",
    "East Laurelhurst Drive Northeast": "EAST LAURELHURST DR NE",
    "Terrace Street": "TERRACE ST",
}

SKIPPED_ROADS = [
    "PORTAGE BAY VIADUCT",
    "MARTIN LUTHER KING JUNIOR WAY",
    "VOLUNTEER PARK RD",
    "2ND AVE EXTENSION S",
    "FIR ST",
    "MARTIN LUTHER KING JUNIOR WAY E",
    "MARTIN LUTHER KING JUNIOR WAY S",
    "E BOSTON TR",
    "STADIUM PL S",
    "E JOHN CT",
    "COLMAN DOCK",
    "SR 99",
    "SR 99 TUNNEL",
    "EVERGREEN POINT FLOATING BRIDGE",
    "SR 99 OFFRAMP",
]


@click.group()
def main():
    pass


@main.command("download-data")
@click.option("--east", default=EAST, help="East boundary in decimal degrees.")
@click.option("--west", default=WEST, help="West boundary in decimal degrees.")
@click.option("--north", default=NORTH, help="North boundary in decimal degrees.")
@click.option("--south", default=SOUTH, help="South boundary in decimal degrees.")
def download_data(east: str, west: str, north: str, south: str):
    ox.config(use_cache=True, log_console=True, all_oneway=True)

    # Download the street network
    G = ox.graph_from_bbox(
        bbox=(north, south, east, west),
        network_type="drive",
        retain_all=True,
        truncate_by_edge=True,
        simplify=True,
    )

    # Optionally, save to a file
    now = datetime.now()
    create_dir(OSM_DIR)
    file_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"{OSM_DIR}/{file_name}.graphml"
    ox.save_graphml(G, filepath=file_path)
    with open(file_path, "r") as f:
        contents = f.read()
        contents = (
            contents.replace("'yes'", "'True'")
            .replace("'no'", "'False'")
            .replace(">yes<", ">True<")
            .replace(">no<", ">False<")
            .replace(">reversible<", ">True<")
        )
    with open(file_path, "w") as f:
        f.write(contents)


@main.command("prepare-seattle-street")
def prepare_seattle_street():
    df = pd.read_csv(
        SEATTLE_STREET_CSV_ORIGINAL,
        usecols=[
            "XSTRLO",
            "XSTRHI",
            "ONSTREET",
            "PVMTCONDINDX1",
            "PVMTCONDINDX2",
            "UNITDESC",
        ],
    )
    df = df.rename(
        columns={
            "UNITDESC": "DESCRIPTION",
            "XSTRLO": "ROAD1",
            "XSTRHI": "ROAD2",
            "ONSTREET": "NAME",
            "PVMTCONDINDX1": "PCI1",
            "PVMTCONDINDX2": "PCI2",
        }
    )
    df.to_csv(SEATTLE_STREET_CSV_PREPARED, index=False)


@main.command("generate-validation-data")
@click.option(
    "--file", help="File to be used. If not provided, the last generated will be used."
)
def generate_validation_data(file: str):
    create_dir(OSM_DIR)

    read_segments_from_seattle_data()

    load_nodes_and_edges_from_osm(file)

    assign_segments_to_edges()

    origin_lat, origin_lon = generate_quality_map()

    generate_samples(origin_lat, origin_lon, 10)


def generate_samples(origin_lat: float, origin_lon: float, interval: float):
    print("Generating samples")
    sample_map = folium.Map(location=[origin_lat, origin_lon], zoom_start=17)
    rows = []
    for i, edge in enumerate(SAMPLE_EDGES.values()):
        segment = edge.get_segment()
        if not segment:
            print(f"No segment found for edge: {edge.on_street}")
            continue
        samples = edge.get_samples(origin_lat, origin_lon, interval)
        for sample in samples:
            _, geopoint = sample
            rows.append(
                {
                    "LATITUDE": geopoint.lat,
                    "LONGITUDE": geopoint.lon,
                    "SEGID": i,
                    "PCI": segment.pci,
                }
            )

            color = get_pci_color(segment.pci)
            folium.CircleMarker(
                location=[geopoint.lat, geopoint.lon],
                radius=2,
                # color=color,
                stroke=False,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
            ).add_to(sample_map)

    print(f"Generating sample map...")
    sample_map.save(SAMPLE_MAP_FILE)
    print(f"Sample map complete: {SAMPLE_MAP_FILE}")

    print(f"Generating samples file...")
    df = pd.DataFrame(rows)
    df.to_csv(VALIDATION_SAMPLES, index=False)
    print(f"Samples file complete: {VALIDATION_SAMPLES}")


def assign_segments_to_edges():
    print("Assigning segments to edges")
    for edge in EDGES:
        segments = []
        possible_segment_keys = edge.get_possible_keys()
        if not possible_segment_keys:
            print("No possible segment keys found in edge: " + edge.on_street)
        else:
            for key in possible_segment_keys:
                segment = STREET_SEGMENTS.get(key, None)
                if segment:
                    segments.append(segment)
            if not segments:
                print("No segment found for keys:")
                for key in possible_segment_keys:
                    print(f"  {key}")
        edge.set_segments(segments)
        if not segments:
            print(f"No segments found for edge: {edge.on_street}")
            continue
        if len(segments) > 1:
            print(f"Multiple segments found for edge: {edge.on_street}")
            continue

        segment_key = edge.get_segment().key
        if segment_key in SAMPLE_EDGES:
            # raise Exception(f'Segment already assigned to another edge: {segment_key}')
            print(f"Segment already assigned to another edge: {segment_key}")
            continue

        SAMPLE_EDGES[segment_key] = edge


def load_nodes_and_edges_from_osm(file: str):
    if not file:
        files = glob.glob(f"{OSM_DIR}/*.graphml")
        file = max(files, key=os.path.getctime)

    print(f"Loading nodes and edges from {file}")

    G = ox.load_graphml(filepath=file)
    for osm_node in G.nodes(data=True):
        node_id = osm_node[0]
        NODES[node_id] = Node(GeoPoint(lat=osm_node[1]["y"], lon=osm_node[1]["x"]))

    for osm_edge in G.edges(keys=True, data=True):
        data = osm_edge[3]
        name = data.get("name", None)

        if not name:
            print(f"Edge without name: {osm_edge}")
            continue
        if type(name) == list:
            street_name = "|".join(normalize_street_name(n) for n in name)
        elif type(name) == str:
            street_name = normalize_street_name(name)
            if street_name in SKIPPED_ROADS:
                print(f"Skipping {street_name}")
                continue
            if street_name not in STREET_NAMES:
                print(f"Street not found in seattle database: {street_name}")
                # raise Exception(f'Street not found in seattle database: {street_name}')

        node1 = NODES[osm_edge[0]]
        node2 = NODES[osm_edge[1]]
        edge = Edge(node1, node2, data.get("geometry", None), street_name)
        node1.edges.append(edge)
        node2.edges.append(edge)
        EDGES.append(edge)


def read_segments_from_seattle_data():
    print(f"Reading segments from Seattle data {SEATTLE_STREET_CSV_PREPARED}")
    df = pd.read_csv(SEATTLE_STREET_CSV_PREPARED)
    for _, entry in df.iterrows():
        street_name = entry["NAME"]
        segment = StreetSegment(
            entry["PCI1"], street_name, [entry["ROAD1"], entry["ROAD2"]]
        )
        STREET_NAMES.add(street_name)
        STREET_SEGMENTS[segment.key] = segment


def generate_quality_map():
    print("Generating quality map")

    lat_avg = sum([node.p.lat for node in NODES.values()]) / len(NODES)
    lon_avg = sum([node.p.lon for node in NODES.values()]) / len(NODES)
    map_center = [lat_avg, lon_avg]

    mymap = folium.Map(location=map_center, zoom_start=17)

    for edge in EDGES:
        folium.PolyLine(
            locations=[(p.lat, p.lon) for p in edge.get_line()],
            color=get_line_color([segment.pci for segment in edge.segments]),
            weight=3,
            opacity=0.6,
            popup=[segment.get_text() for segment in edge.segments],
        ).add_to(mymap)

    create_dir(MAPS_DIR)
    mymap.save(QUALITY_MAP_FILE)
    return lat_avg, lon_avg


def get_pci_color(pci: float):
    if pci == 0:
        return "black"
    if pci < 20:
        return "red"
    if pci < 40:
        return "orange"
    if pci < 60:
        return "yellow"
    if pci < 80:
        return "green"
    return "blue"


def get_line_color(pcis: list[float]):
    if not pcis:
        return "grey"
    if len(pcis) == 1:
        return get_pci_color(pcis[0])
    return "pink"


def normalize_street_name(name: str):
    if not name:
        return name
    if name in MAPPED_NAMES:
        return MAPPED_NAMES[name]
    name = re.sub(r"\bAvenue\b", "AVE", name)
    name = re.sub(r"\bStreet\b", "ST", name)
    name = re.sub(r"\bRoad\b", "RD", name)
    name = re.sub(r"\bDrive\b", "DR", name)
    name = re.sub(r"\bDriveway\b", "DRWY", name)
    name = re.sub(r"\bPlace\b", "PL", name)
    name = re.sub(r"\bLane\b", "LN", name)
    name = re.sub(r"\bCourt\b", "CT", name)
    name = re.sub(r"\bWay\b", "WAY", name)
    name = re.sub(r"\bTerrace\b", "TR", name)
    name = re.sub(r"\bEast\b", "E", name)
    name = re.sub(r"\bWest\b", "W", name)
    name = re.sub(r"\bSouth\b", "S", name)
    name = re.sub(r"\bNorth\b", "N", name)
    name = re.sub(r"\bSoutheast\b", "SE", name)
    name = re.sub(r"\bSouthwest\b", "SW", name)
    name = re.sub(r"\bNortheast\b", "NE", name)
    name = re.sub(r"\bNorthwest\b", "NW", name)
    name = re.sub(r"\bBoulevard\b", "BLVD", name)
    name = re.sub(r"\bShip Canal Bridge\b", "SHIP CANAL TRL", name)
    return name.upper()


def create_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    main()
