import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import limap
import limap.base as _base
import limap.estimators as _estimators
import limap.util.io as limapio
import matplotlib.pyplot


class LineObservation:
    def __init__(self, line_3d_id, line_2d):
        self.line_3d_id = line_3d_id
        self.line_2d = line_2d


class LineMap:
    def __init__(self, lines_dir):
        self.TAG = "[LineMap]"
        self.weight_path = "/home/viki/.limap/models"
        print(self.TAG, "load lines from :", lines_dir)
        self.lines, linetracks = limapio.read_lines_from_input(os.path.join(lines_dir, "finaltracks"))
        print(self.TAG, "   # lines:", len(self.lines))
        print(self.TAG, "   # linetracks:", len(linetracks))

        self.image_folder = os.path.join(lines_dir, "colmap_outputs", "images")

        # read image collection
        self.imagecols = _base.ImageCollection(limapio.read_npy(os.path.join(lines_dir, "imagecols.npy")).item())
        print(self.TAG, "   # imagecols:", self.imagecols.NumImages())

        # load the map with the lines
        self.images_name_to_id = {}
        self.images = {}
        for image_id in self.imagecols.get_img_ids():
            self.images[image_id] = []
            image_name_raw = self.imagecols.image_name(image_id);
            self.images_name_to_id[self.imagecols.image_name(image_id)] = image_id

        # load all the images' connection with lines
        for line_track in linetracks:
            for i in range(line_track.count_lines()):
                image_id = line_track.image_id_list[i]
                line_id = line_track.line_id_list[i]
                if image_id not in self.images:
                    print(self.TAG, "image", image_id, "not found!")
                    continue
                self.images[image_id].append(LineObservation(line_id, line_track.line2d_list[i]))
        print(self.TAG, "map load done.")

        # create extractor and matchor
        detector_cfg = {}
        detector_cfg['method'] = "deeplsd"
        self.detector = limap.line2d.get_detector(detector_cfg, max_num_2d_segs=3000, do_merge_lines=False, visualize=False, weight_path=self.weight_path)
        extractor_cfg = {}
        extractor_cfg['method'] = "wireframe"
        self.extractor = limap.line2d.get_extractor(extractor_cfg, weight_path=self.weight_path)
        matcher_cfg = {}
        matcher_cfg['method'] = "gluestick"
        matcher_cfg['n_jobs'] = 1
        matcher_cfg['topk'] = 0  # process one-to-one match at localization stage
        self.matcher = limap.line2d.get_matcher(matcher_cfg, self.extractor, n_neighbors=20, weight_path=self.weight_path)
        print(self.TAG, "models load done.")


    def get_map_image(self, camera_name, timestamp):
        image_name = os.path.join(self.image_folder, camera_name, str(timestamp) + ".jpg")
        image = cv2.imread(image_name)
        if image_name not in self.images_name_to_id:
            print(self.TAG, image_name, "not found")
            return None, -1
        image_id = self.images_name_to_id[image_name]
        camera = self.imagecols.cam(self.imagecols.camimage(image_id).cam_id)
        image = cv2.resize(image, (camera.w(), camera.h()))
        return image, image_id

    def match_image_with_query(self, image_query, ref_camera_name, ref_timestamp, visualize=False):
        # get the 2d lines in reference image
        image_ref, image_id_ref = self.get_map_image(ref_camera_name, ref_timestamp)
        if image_ref is None:
            return None, None
        # and get descriptors
        segs_ref = []
        image_lines_ref = self.images[image_id_ref]
        for line in image_lines_ref:
            p1 = line.line_2d.start
            p2 = line.line_2d.end
            segs_ref.append([p1[0], p1[1], p2[0], p2[1], line.line_2d.length()])
        segs_ref = np.array(segs_ref)
        image_ref_gray = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
        descinfo_ref = self.extractor.compute_descinfo(image_ref_gray, segs_ref)

        # detect lines in query image
        image_query_gray = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
        segs_query = self.detector.detect_with_image(image_query_gray)
        segs_query, _ = self.detector.take_longest_k(segs_query, max_num_2d_segs=self.detector.max_num_2d_segs)
        descinfo_query = self.extractor.compute_descinfo(image_query_gray, segs_query)

        # match the lines & draw
        matches = self.matcher.match_pair(descinfo_ref, descinfo_query)
        if not visualize:
            return matches, None

        # draw the visualization and return the render image
        # draw matched lines to each image
        colors = matplotlib.cm.hsv(np.random.rand(len(matches))) * 255
        colors = colors.tolist()
        for i in range(len(matches)):
            match = matches[i]
            line_ref = segs_ref[match[0]]
            line_qry = segs_query[match[1]]

            p1 = (int(line_ref[0]), int(line_ref[1]))
            p2 = (int(line_ref[2]), int(line_ref[3]))
            cv2.line(image_ref, p1, p2, colors[i], 2)

            p1 = (int(line_qry[0]), int(line_qry[1]))
            p2 = (int(line_qry[2]), int(line_qry[3]))
            cv2.line(image_query, p1, p2, colors[i], 2)

        # resize image query height to image ref
        new_width = image_query.shape[1] * image_ref.shape[0] / image_query.shape[0]
        image_query_resized = cv2.resize(image_query, (int(new_width), image_ref.shape[0]))
        image_show = cv2.hconcat([image_query_resized, image_ref])
        return matches, image_show


    def draw_image_with_lines(self, camera_name, timestamp, thickness=2, endpoints=True):
        # draw the lines
        image, image_id = self.get_map_image(camera_name, timestamp)
        if image is None:
            return None
        image_lines = self.images[image_id]

        for line in image_lines:
            color = (0, 0, 255)
            # get color by depth
            p1 = (int(line.line_2d.start[0]), int(line.line_2d.start[1]))
            p2 = (int(line.line_2d.end[0]), int(line.line_2d.end[1]))
            cv2.line(image, p1, p2, color, thickness)
            if endpoints:
                cv2.circle(image, p1, thickness * 2, color, -1)
                cv2.circle(image, p2, thickness * 2, color, -1)
        return image
