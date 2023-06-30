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


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid,
                                                delimiter="&",
                                                max_rows=1,
                                                usecols=(0, 1, 2),
                                                dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def read_rgbd(depth_path, image_path, pose_path):
    from decimal import Decimal
    image = cv2.imread(image_path)
    depth = read_array(depth_path)

    pose_file = open(pose_path, "r")
    line = pose_file.readline()
    tmp = line.split(" ")
    K = np.array([[float(tmp[0]), 0, float(tmp[2])], [0, float(tmp[1]), float(tmp[3])], [0, 0, 1]]).astype(np.float64)

    world_to_ref_trans = np.array([0, 0, 0]).astype(np.float64)
    world_to_ref_rot = np.eye(3).astype(np.float64)
    for i in range(3):
        line = pose_file.readline()
        tmp = line.split(" ")
        world_to_ref_rot[i][0] = float(tmp[0])
        world_to_ref_rot[i][1] = float(tmp[1])
        world_to_ref_rot[i][2] = float(tmp[2])

    line = pose_file.readline()
    tmp = line.split(" ")
    world_to_ref_trans[0] = Decimal(float(tmp[0]))
    world_to_ref_trans[1] = Decimal(float(tmp[1]))
    world_to_ref_trans[2] = Decimal(float(tmp[2]))
    return depth, image, world_to_ref_rot, world_to_ref_trans, K


class LineObservation:
    def __init__(self, line_3d_id, line_2d):
        self.line_3d_id = line_3d_id
        self.line_2d = line_2d


class LineMap:
    def __init__(self):
        self.TAG = "[LineMap]"
        self.weight_path = "/home/viki/.limap/models"
        self.loaded = False
        self.model_loaded = False
        self.model_label = ""

    def load_model(self):
        # create extractor and matchor
        # deeplsd will have strange lines, LSD result is more stable
        detector_cfg = {}
        # ["lsd", "sold2", "hawpv3", "tp_lsd", "deeplsd"]
        detector_cfg['method'] = "deeplsd"
        self.detector = limap.line2d.get_detector(detector_cfg, max_num_2d_segs=1000, do_merge_lines=False, visualize=False, weight_path=self.weight_path)
        extractor_cfg = {}
        # ["sold2", "lbd", "l2d2", "linetr", "superpoint_endpoints", "wireframe"]
        extractor_cfg['method'] = "wireframe"
        self.extractor = limap.line2d.get_extractor(extractor_cfg, weight_path=self.weight_path)
        matcher_cfg = {}
        # ["sold2", "lbd", "l2d2", "linetr", "nn_endpoints", "superglue_endpoints", "gluestick"]
        matcher_cfg['method'] = "gluestick"
        matcher_cfg['n_jobs'] = 1
        matcher_cfg['topk'] = 0  # process one-to-one match at localization stage
        matcher_cfg['superglue'] = {'weights' : "outdoor"}
        self.matcher = limap.line2d.get_matcher(matcher_cfg, self.extractor, n_neighbors=20, weight_path=self.weight_path)
        print(self.TAG, "models load done.")
        self.model_loaded = True
        self.model_label = detector_cfg['method'] + "_" + extractor_cfg['method'] + "_" + matcher_cfg['method']

    def load(self, lines_dir):
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
        self.load_model()
        self.loaded = True


    def get_map_image(self, camera_name, timestamp):
        if not self.loaded:
            print(self.TAG, "map not loaded.")
            return None, -1
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

        # get line3ds and line 2ds
        line3ds = []
        line3d_ids = []
        line2ds = []
        # add all the line3d seen by reference image
        for line in image_lines_ref:
            line3ds.append(self.lines[line.line_3d_id])
        for i in range(len(matches)):
            line3d_ids.append(matches[i][0])
            line_qry = segs_query[matches[i][1]]
            p1 = np.array([line_qry[0], line_qry[1]], dtype=np.float64)
            p2 = np.array([line_qry[2], line_qry[3]], dtype=np.float64)
            line2ds.append(_base.Line2d(p1, p2))


        if not visualize:
            return line3ds, line3d_ids, line2ds, None

        # draw the visualization and return the render image
        # draw all the map lines to ref
        for line in image_lines_ref:
            color = (0, 0, 255)
            p1 = (int(line.line_2d.start[0]), int(line.line_2d.start[1]))
            p2 = (int(line.line_2d.end[0]), int(line.line_2d.end[1]))
            cv2.line(image_ref, p1, p2, color, 2)

        cv2.putText(image_ref, "#map lines :" + str(len(image_lines_ref)),
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 100), 1, cv2.LINE_AA)

        # draw matched lines to each image
        colors = matplotlib.cm.hsv(np.random.rand(len(matches))) * 255
        colors = colors.tolist()
        for i in range(len(matches)):
            match = matches[i]
            line_ref = segs_ref[match[0]]
            line_qry = segs_query[match[1]]

            p1 = (int(line_ref[0]), int(line_ref[1]))
            p2 = (int(line_ref[2]), int(line_ref[3]))
            cv2.line(image_ref, p1, p2, colors[i], 3)

            p1 = (int(line_qry[0]), int(line_qry[1]))
            p2 = (int(line_qry[2]), int(line_qry[3]))
            cv2.line(image_query, p1, p2, colors[i], 3)

        # resize image query height to image ref
        new_width = image_query.shape[1] * image_ref.shape[0] / image_query.shape[0]
        image_query_resized = cv2.resize(image_query, (int(new_width), image_ref.shape[0]))
        image_show = cv2.hconcat([image_query_resized, image_ref])
        cv2.putText(image_show, "#matches :" + str(len(matches)),
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 100), 1, cv2.LINE_AA)

        return line3ds, line3d_ids, line2ds, image_show


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


    def detect_image(self, image_rgb, resize_ratio=0.5):
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        if resize_ratio != 1.0:
            dsize = (int(image_gray.shape[1] * resize_ratio), int(image_gray.shape[0] * resize_ratio))
            image_gray = cv2.resize(image_gray, dsize)
        segs = self.detector.detect_with_image(image_gray)
        if self.detector.do_merge_lines:
            segs = self.detector.merge_lines(segs)
        segs, _ = self.detector.take_longest_k(segs, max_num_2d_segs=self.detector.max_num_2d_segs)
        if resize_ratio != 1.0:
            resize_ratio_inv = 1.0 / resize_ratio
            for i in range(len(segs)):
                seg_new = segs[i]
                for j in range(5):
                    seg_new[j] = resize_ratio_inv * seg_new[j]
                segs[i] = seg_new

            dsize = (int(image_rgb.shape[1]), int(image_rgb.shape[0]))
            image_gray = cv2.resize(image_gray, dsize)
        return segs, image_gray


    def locate_rgbd(self, image_query, depth_ref, image_ref, world_to_ref_rot, world_to_ref_trans, K_ref, visualize):
        line3ds = []
        line3d_ids = []
        line2ds = []
        line2ds_ref = []
        if not self.model_loaded:
            print(self.TAG, "model not loaded.")
            return line3ds, line3d_ids, line2ds, line2ds_ref, None

        ref_to_world_rot = np.transpose(world_to_ref_rot)
        ref_to_world_trans = -(ref_to_world_rot.dot(world_to_ref_trans))
        K_ref_inv = np.linalg.inv(K_ref)

        def ref_pixel_to_world_point(pixel):
            if int(pixel[1]) >= depth_ref.shape[0] or int(pixel[0]) >= depth_ref.shape[1]:
                return False, None
            depth = depth_ref[int(pixel[1])][int(pixel[0])]
            if depth < 0.1:
                return False, None
            pixel_homo = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
            pt_ref = depth * (K_ref_inv.dot(pixel_homo))
            pt_world = ref_to_world_rot.dot(pt_ref) + ref_to_world_trans
            return True, pt_world

        # run detection & extraction
        segs_qry, image_query_gray = self.detect_image(image_query)
        descinfo_qry = self.extractor.compute_descinfo(image_query_gray, segs_qry)
        segs_ref_raw, image_ref_gray = self.detect_image(image_ref)

        # filter ref segs and make lines
        segs_ref = []
        shrinkage = 0.1
        for seg in segs_ref_raw:
            dx = seg[2] - seg[0]
            dy = seg[3] - seg[1]
            if dx * dx + dy * dy < 100:
                continue

            # shrink the line to get 3d points
            ret_1, begin_world = ref_pixel_to_world_point([seg[0] + dx * shrinkage, seg[1] + dy * shrinkage])
            ret_2, end_world = ref_pixel_to_world_point([seg[2] - dx * shrinkage, seg[3] - dy * shrinkage])
            if ret_1 and ret_2:
                segs_ref.append(seg.tolist())
                line3ds.append(_base.Line3d(begin_world, end_world))

        segs_ref = np.array(segs_ref)
        descinfo_ref = self.extractor.compute_descinfo(image_ref_gray, segs_ref)

        # run match
        matches = self.matcher.match_pair(descinfo_ref, descinfo_qry)
        for i in range(len(matches)):
            line3d_ids.append(matches[i][0])
            line_qry = segs_qry[matches[i][1]]
            p1 = np.array([line_qry[0], line_qry[1]], dtype=np.float64)
            p2 = np.array([line_qry[2], line_qry[3]], dtype=np.float64)
            line2ds.append(_base.Line2d(p1, p2))
            line_ref = segs_ref[matches[i][0]]
            p1 = np.array([line_ref[0], line_ref[1]], dtype=np.float64)
            p2 = np.array([line_ref[2], line_ref[3]], dtype=np.float64)
            line2ds_ref.append(_base.Line2d(p1, p2))

        if not visualize:
            return line3ds, line3d_ids, line2ds, line2ds_ref, image_ref

        # render the depth
        depth_int = 2 * np.expand_dims(depth_ref, axis=2).astype(np.uint8)
        depth_int[depth_int == 0] = 255
        depth_color = cv2.applyColorMap(depth_int, cv2.COLORMAP_JET)
        image_ref = (0.5 * depth_color + 0.5 * image_ref).astype(np.uint8)

        # show the matches
        colors = matplotlib.cm.hsv(np.random.rand(len(matches))) * 255
        colors = colors.tolist()
        for i in range(len(matches)):
            match = matches[i]
            line_ref = segs_ref[match[0]]
            line_qry = segs_qry[match[1]]

            p1 = (int(line_ref[0]), int(line_ref[1]))
            p2 = (int(line_ref[2]), int(line_ref[3]))
            cv2.line(image_ref, p1, p2, colors[i], 3)

            p1 = (int(line_qry[0]), int(line_qry[1]))
            p2 = (int(line_qry[2]), int(line_qry[3]))
            cv2.line(image_query, p1, p2, colors[i], 3)

        # resize image query height to image ref
        new_width = image_query.shape[1] * image_ref.shape[0] / image_query.shape[0]
        image_query_resized = cv2.resize(image_query, (int(new_width), image_ref.shape[0]))
        image_show = cv2.hconcat([image_query_resized, image_ref])
        cv2.putText(image_show, "#matches :" + str(len(matches)),
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 100), 1, cv2.LINE_AA)

        return line3ds, line3d_ids, line2ds, line2ds_ref, image_ref
