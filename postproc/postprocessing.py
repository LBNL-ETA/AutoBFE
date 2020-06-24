import csv
import io
import os
import sys
import mercantile
import geojson
import concurrent.futures as futures
import numpy as np
import shapely.geometry
from tqdm import tqdm
from PIL import Image
from common.colors import make_palette
from modeling.utils.metrics import Evaluator
from postproc.utils import *
from postproc.building_footprint import BuildingExtract


def get_masks_from_probs(probs_path, pred_masks_path, colors,weights=None, 
                         probs_threshold= None, num_workers = 4):
   
    """generate masks from probabilities files

    Args:
      probs_path: list of the directory with probabilities files that have a name with aslippy map structure "z_x_y.png"
      pred_masks_path: directory where the masks will be saved
      weights: array-like for weighting probabilities

    """
    if weights and len(probs_path) != len(weights):
        sys.exit("Error: number of probs directories and weights must be the same")

    os.makedirs(pred_masks_path, exist_ok=True)

    tilesets = map(tiles_from_directory, probs_path)
    list_tilesets = list(zip(*tilesets))

    with futures.ThreadPoolExecutor(num_workers) as executor:

        def worker(tileset):
            anchors = np.linspace(0, 1, 256)

            try:
                tiles = [tile for tile, _ in tileset]
                paths = [path for _, path in tileset]
                assert len(set(tiles)), "tilesets ont sync"

            except OSError:
                    return tiles, False

            x, y, z = tiles[0]
            probs = [probs_load(path,anchors,probs_threshold) for path in paths]
            # Predictions Ensemblling using weighted average probabilities
            # See http://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
            mask = np.argmax(np.average(probs, axis=0, weights=weights), axis=0)
            mask = mask.astype(np.uint8)

            bg = colors[0]
            fg = colors[1]
            palette = make_palette(bg, fg)
            out = Image.fromarray(mask, mode="P")
            out.putpalette(palette)
        
            path = os.path.join(pred_masks_path,"{}_{}_{}.{}".format(z,x,y,"png"))
            out.save(path, optimize=True)

            return tiles, True 

        for tiles, ok in executor.map(worker, list_tilesets):
            if not ok:
                print("Warning: {} tilesets ont sync".format(tiles), file=sys.stderr)


def get_polygons(pred_masks_path, polygons_path,
                 kernel_opening = 20, simplify_threshold = 0.01):
    
    """Generate GeoJSON polygons from predicted masks

    Args:
      pred_masks_path: directory where the predicted mask are saved
      polygons_path: path to GeoJSON file to store features in
      kernel_opening: the opening morphological operation's kernel size in pixel
      simplify_threshold: the simplification accuracy as max. percentage of the arc length, in [0, 1]

    """
    bldg_extract = BuildingExtract()
    tiles = list(tiles_from_directory(pred_masks_path))

    for tile, path in tqdm(tiles, ascii=True, unit="mask"):
        mask = np.array(Image.open(path).convert("P"), dtype=np.uint8)
        bldg_extract.extract(tile, mask)

    bldg_extract.save(polygons_path)


def merge_polygons(polygons_path, new_polygons_path, distance_threshold):
    """
    Adapted from: https://github.com/mapbox/robosat 

    Args:
      polygons_path: "GeoJSON file to read polygons from"
      new_polygons_path: path to GeoJSON file where the merged polygons will be saved
      distance_threshold: minimum distance to define adjacent polygons, in meters

    """
    with open(polygons_path) as fp:
        collection = geojson.load(fp)

    shapes = [shapely.geometry.shape(feature["geometry"]) for feature in collection["features"]]
    del collection

    graph = UndirectedGraph()
    idx = make_index(shapes)
    def buffered(shape):
        projected = project(shape, "epsg:4326", "epsg:3395")
        buffered = projected.buffer(distance_threshold)
        unprojected = project(buffered, "epsg:3395", "epsg:4326")
        return unprojected

    def unbuffered(shape):
        projected = project(shape, "epsg:4326", "epsg:3395")
        unbuffered = projected.buffer(-1 * distance_threshold)
        unprojected = project(unbuffered, "epsg:3395", "epsg:4326")
        return unprojected

    for i, shape in enumerate(tqdm(shapes, desc="Building graph", unit="shapes", ascii=True)):
        embiggened = buffered(shape)

        graph.add_edge(i, i)

        nearest = [j for j in idx.intersection(embiggened.bounds, objects=False) if i != j]

        for t in nearest:
            if embiggened.intersects(shapes[t]):
                graph.add_edge(i, t)

    components = list(graph.components())
    assert sum([len(v) for v in components]) == len(shapes), "components capture all shape indices"

    features = []

    for component in tqdm(components, desc="Merging components", unit="component", ascii=True):
        embiggened = [buffered(shapes[v]) for v in component]
        merged = unbuffered(union(embiggened))

        if merged.is_valid:
            # Orient exterior ring of the polygon in counter-clockwise direction.
            if isinstance(merged, shapely.geometry.polygon.Polygon):
                merged = shapely.geometry.polygon.orient(merged, sign=1.0)
            elif isinstance(merged, shapely.geometry.multipolygon.MultiPolygon):
                merged = [shapely.geometry.polygon.orient(geom, sign=1.0) for geom in merged.geoms]
                merged = shapely.geometry.MultiPolygon(merged)
            else:
                print("Warning: merged feature is neither Polygon nor MultiPoylgon, skipping", file=sys.stderr)
                continue

            # equal-area projection; round to full m^2, we're not that precise anyway
            area = int(round(project(merged, "epsg:4326", "esri:54009").area))

            feature = geojson.Feature(geometry=shapely.geometry.mapping(merged), properties={"area": area})
            features.append(feature)
        else:
            print("Warning: merged feature is not valid, skipping", file=sys.stderr)

    collection = geojson.FeatureCollection(features)

    with open(new_polygons_path, "w") as fp:
        geojson.dump(collection, fp)


def postproc_evaluation(pred_masks_path, masks_path, nclass):
    # Define Evaluator
    evaluator = Evaluator(nclass)

    mask_names = list(os.listdir(masks_path))

    for i, mask_name in enumerate(mask_names):
        mask_path = os.path.join(masks_path,mask_name)
        pred_mask_path = os.path.join(pred_masks_path,mask_name)

        mask = Image.open(mask_path)#.convert('L')
        # convert RGB into B & W
        thresh = 100
        fn = lambda x : 255 if x < thresh else 0
        mask = mask.convert('L').point(fn, mode='1')
        mask = np.array(mask)

        pred_mask = Image.open(pred_mask_path)#.convert('L')
        # convert RGB into B & W
        fn = lambda x : 255 if x < thresh else 0
        pred_mask = pred_mask.convert('L').point(fn, mode='1')
        pred_mask = np.array(pred_mask)

        evaluator.add_batch(mask, pred_mask)
    
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    F1Score = evaluator.F1_Score()
    Recall = evaluator.Recall()
    Precision = evaluator.Precision()
    metrics_results = "Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, F1Score: {}, Recall: {}, Precision: {}".format(Acc, Acc_class, mIoU, FWIoU, F1Score, Recall, Precision)
    print('Test:')
    print(metrics_results)


    
    