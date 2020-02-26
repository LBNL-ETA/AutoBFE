import sys
import collections

import geojson

import shapely.geometry

from postproc.utils import opening, extract_contours, simplify, parents_in_hierarchy, featurize

class BuildingExtract(object):
    def __init__(self, kernel_opening = 20, simplify_threshold = 0.01):
        """
        Adapted from: https://github.com/mapbox/robosat 
        """
        self.kernel_opening = kernel_opening
        self.simplify_threshold = simplify_threshold
        self.features = []

    def extract(self, tile, mask):

        mask = opening(mask, self.kernel_opening)
        multipolygons, hierarchy = extract_contours(mask)

        if hierarchy is None:
            return

        assert len(hierarchy) == 1, "always single hierarchy for all polygons in multipolygon"
        hierarchy = hierarchy[0]

        assert len(multipolygons) == len(hierarchy), "polygons and hierarchy in sync"

        polygons = [simplify(polygon, self.simplify_threshold) for polygon in multipolygons]

        # All child ids in hierarchy tree, keyed by root id.
        features = collections.defaultdict(set)

        for i, (polygon, node) in enumerate(zip(polygons, hierarchy)):
            if len(polygon) < 3:
                print("Warning: simplified feature no longer valid polygon, skipping", file=sys.stderr)
                continue

            _, _, _, parent_idx = node

            ancestors = list(parents_in_hierarchy(i, hierarchy))

            # Only handles polygons with a nesting of two levels for now => no multipolygons.
            if len(ancestors) > 1:
                print("Warning: polygon ring nesting level too deep, skipping", file=sys.stderr)
                continue

            # A single mapping: i => {i} implies single free-standing polygon, no inner rings.
            # Otherwise: i => {i, j, k, l} implies: outer ring i, inner rings j, k, l.
            root = ancestors[-1] if ancestors else i

            features[root].add(i)

        for outer, inner in features.items():
            rings = [featurize(tile, polygons[outer], mask.shape[:2])]

            # In mapping i => {i, ..} i is not a child.
            children = inner.difference(set([outer]))

            for child in children:
                rings.append(featurize(tile, polygons[child], mask.shape[:2]))

            assert 0 < len(rings), "at least one outer ring in a polygon"

            geometry = geojson.Polygon(rings)
            shape = shapely.geometry.shape(geometry)

            if shape.is_valid:
                self.features.append(geojson.Feature(geometry=geometry))
            else:
                print("Warning: extracted feature is not valid, skipping", file=sys.stderr)

    def save(self, out):
        collection = geojson.FeatureCollection(self.features)

        with open(out, "w") as fp:
            geojson.dump(collection, fp)
