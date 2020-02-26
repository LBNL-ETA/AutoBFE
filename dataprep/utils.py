import csv
import mercantile
import io
from rasterio.features import rasterize
from PIL import Image
from shapely.geometry import Point
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import transform
from supermercado import burntiles
from enum import Enum, unique


def tiles_from_csv(tiles_cover_path):
    """Read tiles from a line-delimited csv file.
    Args:
      file: the path to read the csv file from.
    Yields:
        The mercantile tiles from the csv file.
    """
    with open(tiles_cover_path) as fp:
        reader = csv.reader(fp)

        for row in reader:
            if not row:
                continue

            yield mercantile.Tile(*map(int, row))



def fetch_image(session, url, timeout=10):
    """download the satellite image for a tile.

    Args:
        session: the HTTP session to download the image from
        url: the tile imagery's url to download the image from
        timeout: the HTTP timeout in seconds.

    Returns:
        The satellite imagery as bytes or None in case of error.
    """

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return io.BytesIO(resp.content)
    except Exception:
        return None

    
def burn(tile, features, size):
    """Burn tile with features.

    Args:
      tile: the mercantile tile to burn.
      features: the geojson features to burn.
      size: the size of burned image.

    Returns:
      image: rasterized file of size with features burned.
    """

    # the value you want in the output raster where a shape exists
    burnval = 1
    shapes = ((geometry, burnval) for feature in features for geometry in feature_to_mercator(feature))

    bounds = mercantile.xy_bounds(tile)
    transform = from_bounds(*bounds, size, size)

    return rasterize(shapes, out_shape=(size, size), transform=transform, all_touched = True)


def feature_to_mercator(feature):
    """Normalize feature and converts coords to 3857.

    Args:
      feature: geojson feature to convert to mercator geometry.
    """
    # Ref: https://gist.github.com/dnomadb/5cbc116aacc352c7126e779c29ab7abe

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)

    geometry = feature["geometry"]
    if geometry["type"] == "Polygon":
        xys = (zip(*part) for part in geometry["coordinates"])
        xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)

        yield {"coordinates": list(xys), "type": "Polygon"}

    elif geometry["type"] == "MultiPolygon":
        for component in geometry["coordinates"]:
            xys = (zip(*part) for part in component)
            xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)

            yield {"coordinates": list(xys), "type": "Polygon"}


