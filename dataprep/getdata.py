import numpy as np
import pandas as pd
import collections
import json
import csv
import os
import mercantile
import sys
import shutil
import time
import requests
import concurrent.futures as futures
from PIL import Image
from supermercado import burntiles
from sklearn.model_selection import train_test_split
from dataprep.utils import tiles_from_csv, fetch_image, burn
from common.colors import make_palette

class SatMask(object):
    def __init__(self, url_mapbox=None, image_format=None, mask_format=None, 
                 tiles_cover_path=None, sat_path=None, gis_path=None, masks_path=None, 
                 zoom=19, size=512, num_workers=1):
        """
        Args:
          url_mapbox: the mapboxurl to download satellite images from
          image_format: the downloaded image fomat 
          tiles_cover_path: path to csv file to store tiles information (slippy map format)
          sat_path: path to  directory to store the downloaded satellite 
          images tiles
          gis_path: the path to read the GeoJSON features
          masks_path: path to directory to store masks images
          num_workers: rate limit in max. requests per second
        """
        self.url_mapbox = url_mapbox
        self.image_format = image_format
        self.mask_format = mask_format
        self.tiles_cover_path = tiles_cover_path
        self.sat_path = sat_path
        self.gis_path = gis_path
        self.masks_path = masks_path
        self.zoom = zoom
        self.size = size
        self.num_workers = num_workers


    def tiles_cover(self):
        """generates tiles covering GeoJSON file.
        """
        features = json.load(open(self.gis_path))
        tiles = []

        for feature in features["features"]:
                tiles.extend(map(tuple, burntiles.burn([feature], self.zoom).tolist()))

        tiles = list(set(tiles))

        with open(self.tiles_cover_path, "w") as fp:
            writer = csv.writer(fp)
            writer.writerows(tiles)


    def mapbox_download(self):
        """downloads images from Mapbox Maps API.
        """
        tiles = list(tiles_from_csv(self.tiles_cover_path))
        os.makedirs(self.sat_path, exist_ok=True)

        with requests.Session() as session:

            with futures.ThreadPoolExecutor(self.num_workers) as executor:

                def worker(tile):
                    tick = time.monotonic()
                    x, y, z = map(str, [tile.x, tile.y, tile.z])
                    path = os.path.join(self.sat_path,"{}_{}_{}.{}".format(z,x,y, self.image_format))

                    if os.path.isfile(path):
                        return tile, True

                    url = self.url_mapbox.format(x=tile.x, y=tile.y, z=tile.z)
                    res = fetch_image(session,url)

                    if not res:
                        return tile, False

                    try:
                        image = Image.open(res)
                        image.save(path, optimize=True)
                    except OSError:
                        return tile, False

                    tock = time.monotonic()
                    time_for_req = tock - tick
                    time_per_worker = self.num_workers / self.num_workers

                    if time_for_req < time_per_worker:
                        time.sleep(time_per_worker - time_for_req)

                    return tile, True

                for tile, ok in executor.map(worker, tiles):
                    if not ok:
                        print("Warning: {} failed, skipping".format(tile), file=sys.stderr)
    

    def map_masks(self,colors):

        os.makedirs(self.masks_path, exist_ok=True)
        assert all(tile.z == self.zoom for tile in tiles_from_csv(self.tiles_cover_path))
    
        with open(self.gis_path) as f:
            fc = json.load(f)
    
        # Find all tiles the features cover and make a map object for quick lookup.
        feature_map = collections.defaultdict(list)
        for i, feature in enumerate(fc["features"]):

            if feature["geometry"]["type"] != "Polygon":
                continue

            try:
                for tile in burntiles.burn([feature], zoom=self.zoom):
                    feature_map[mercantile.Tile(*tile)].append(feature)
            except ValueError: # as e:
                print("Warning: invalid feature {}, skipping".format(i), file=sys.stderr)
                continue
        # Burn features to tiles and write to a png image.
        for tile in list(tiles_from_csv(self.tiles_cover_path)):
            if tile in feature_map:
                out = burn(tile, feature_map[tile], self.size)
            else:
                out = np.zeros(shape=(self.size, self.size), dtype=np.uint8)

            x, y, z = map(str, [tile.x, tile.y, tile.z])
            out_path = os.path.join(self.masks_path,"{}_{}_{}.{}".format(z,x,y,self.mask_format))
            bg = colors[0]
            fg = colors[1]
            out = Image.fromarray(out, mode="P")
            palette = make_palette(bg, fg)
            out.putpalette(palette)

            out.save(out_path, optimize=True)
        
    def split_tiles_cover(self, rtrain = 0, rval = 0, rtest =0,
                          train_tiles_cover_path = None, 
                          val_tiles_cover_path = None,
                          test_tiles_cover_path = None):
        assert (rtrain + rval + rtest) == 1, "the sum of rtrain, rval amd rtest should be equal to 1"
        
        self.train_tiles_cover_path = train_tiles_cover_path
        self.val_tiles_cover_path = val_tiles_cover_path
        self.test_tiles_cover_path = test_tiles_cover_path
        self.rtrain = rtrain
        self.rval = rval
        self.rtest = rtest

        tiles_df = pd.read_csv(self.tiles_cover_path)
        
        tiles_train, tiles_remain = train_test_split(tiles_df, test_size = (rtest+rval))
        tiles_train.to_csv(train_tiles_cover_path,index = False, header = False)
        print("Size of training sample: {}".format(tiles_train.shape))

        if rval == 0:
            tiles_remain.to_csv(test_tiles_cover_path,index = False, header = False)
            print("Size of test sample: {}".format(tiles_remain.shape))

        elif rtest == 0:
            tiles_remain.to_csv(val_tiles_cover_path,index = False, header = False)
            print("Size of validation sample: {}".format(tiles_remain.shape))

        else:
            newrval = np.around(rval / (rtest + rval), 2)
            tiles_test, tiles_val = train_test_split(tiles_remain, test_size=newrval)
            tiles_val.to_csv(val_tiles_cover_path,index = False, header = False)
            tiles_test.to_csv(test_tiles_cover_path,index = False, header = False)
            print("Size of validation sample: {}".format(tiles_val.shape))
            print("Size of test sample: {}".format(tiles_test.shape))


    def split_data(self, train_val_test_path):
        train_path = os.path.join(train_val_test_path, "train")
        os.makedirs(train_path, exist_ok=True)
        self._split_images_masks(train_path,self.train_tiles_cover_path)

        if self.rval == 0:
            test_path = os.path.join(train_val_test_path, "test")
            os.makedirs(test_path, exist_ok=True)
            self._split_images_masks(test_path,self.test_tiles_cover_path)
            
        elif self.rtest == 0:
            val_path = os.path.join(train_val_test_path, "val")
            os.makedirs(val_path, exist_ok=True)
            self._split_images_masks(val_path,self.val_tiles_cover_path)

        else:
            val_path = os.path.join(train_val_test_path, "val")
            os.makedirs(val_path, exist_ok=True)
            test_path =os.path.join(train_val_test_path, "test")
            os.makedirs(test_path, exist_ok=True)
            self._split_images_masks(val_path,self.val_tiles_cover_path)
            self._split_images_masks(test_path,self.test_tiles_cover_path)


    def _split_images_masks(self, out_path, tiles_cover_path):
        out_sat_path = os.path.join(out_path, "images")
        out_masks_path =  os.path.join(out_path, "masks")
        os.makedirs(out_sat_path, exist_ok=True)
        os.makedirs(out_masks_path, exist_ok=True)
        tiles = list(tiles_from_csv(tiles_cover_path))

        with futures.ThreadPoolExecutor(self.num_workers) as executor:

            def worker(tile):

                x, y, z = map(str, [tile.x, tile.y, tile.z])
                image_path = os.path.join(self.sat_path,"{}_{}_{}.{}".format(z,x,y, self.image_format))
                feature_path = os.path.join(self.masks_path, "{}_{}_{}.{}".format(z,x,y, self.mask_format))

                try:
                    image = Image.open(image_path)
                    feature = Image.open(feature_path)

                except OSError:
                    return tile, False
        
                image_path_new = os.path.join(out_sat_path, "{}_{}_{}.{}".format(z, x, y, self.image_format))
                feature_path_new = os.path.join(out_masks_path, "{}_{}_{}.{}".format(z, x, y, self.mask_format))
                image.save(image_path_new)
                feature.save(feature_path_new)
                
                return tile, True

            for tile, ok in executor.map(worker, tiles):
                if not ok:
                    print("Warning: {} does not exist".format(tile), file=sys.stderr)
    
    def delete_no_split(self):
        """code adapted from: https://stackoverflow.com/questions/6996603/delete-a-file-or-folder 
        """
        try:
            shutil.rmtree(self.sat_path)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
        try:
            shutil.rmtree(self.masks_path)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))








    
    




