import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import limap
from decimal import Decimal
import limap.base as _base
import limap.estimators as _estimators
import limap.util.io as limapio
import matplotlib.pyplot as plt
import matplotlib
import logging
import argparse
from pathlib import Path
from hloc.utils.read_write_model import *
from read_write_model import qvec2rotmat, get_intrinsics_matrix

from websocket_server import WebsocketServer
from line_map import LineMap, read_rgbd

ransac_cfg = {}
request_cnt = 0
line_map = LineMap()

base_folder = ""

# Called for every client connecting (after handshake)
def new_client(client, server):
	print("[WEBSOCKET] New client connected and was given id %d" % client['id'])
	server.send_message_to_all("Hey all, a new client has joined us")


# Called for every client disconnecting
def client_left(client, server):
	print("[WEBSOCKET] Client(%d) disconnected" % client['id'])


def process_localization():
    global request_cnt
    request_cnt = request_cnt + 1

    # read the request from file and run pnp
    points_file_path = os.path.join(base_folder, "request.txt")
    image_path = os.path.join(base_folder, "request.jpg")
    line_image_path = os.path.join(base_folder, "render_line_" + str(request_cnt) + ".jpg")
    output_file_path = os.path.join(base_folder, "result.txt")

    point_3ds = []
    point_2ds = []
    points_file = open(points_file_path, "r")
    points_size = int(points_file.readline()[:-1])
    has_offset = False
    offset = np.array([0, 0, 0]).astype(np.float64)
    for i in range(points_size):
        line = points_file.readline()[:-1]
        tmp = line.split(" ")
        # transform the point to local coordinate
        pt_world = np.array([0, 0, 0]).astype(np.float64)
        pt_world[0] = Decimal(float(tmp[0]))
        pt_world[1] = Decimal(float(tmp[1]))
        pt_world[2] = Decimal(float(tmp[2]))

        if not has_offset:
            has_offset = True
            offset = pt_world
        # transform by offset
        pt_local = pt_world - offset
        point_3ds.append([pt_local[0], pt_local[1], pt_local[2]])
        point_2ds.append([float(tmp[3]), float(tmp[4])])

    # read camera
    line_intrin = points_file.readline()[:-1]
    tmp = line_intrin.split(" ")
    K = np.array([[float(tmp[0]), 0, float(tmp[2])], [0, float(tmp[1]), float(tmp[3])], [0, 0, 1]]).astype(np.float32)
    camera = _base.Camera("PINHOLE", K, cam_id=request_cnt, hw=[int(tmp[5]), int(tmp[4])])

    # read reference image id
    camera_name_ref = points_file.readline()[:-1]  # skip the last \n char
    timestamp_ref = int(points_file.readline())
    print("Request Information:")
    print(" - Reference :", camera_name_ref, timestamp_ref)
    print(" - intrinsics :", line_intrin)
    print(" - # points :", len(point_3ds))

    # read recover matrix of 2d pixels
    recover_matrix = np.eye(3, dtype=np.float64)
    for i in range(3):
        rcm_tmp = points_file.readline()
        tmp = rcm_tmp.split(" ")
        recover_matrix[i][0] = float(tmp[0])
        recover_matrix[i][1] = float(tmp[1])
        recover_matrix[i][2] = float(tmp[2])

    # find line matches
    line3ds = []
    line2ds = []
    line3d_ids = []
    line2ds_ref = []
    depth_ref_path = os.path.join(base_folder, "ref_depth.bin")
    image_ref_path = os.path.join(base_folder, "ref_image.jpg")
    pose_ref_path = os.path.join(base_folder, "ref_pose.txt")
    if not line_map.model_loaded:
        print("Model Not Loaded")
        return
    image_query = cv2.imread(image_path)
    depth_ref, image_ref, world_to_ref_rot, world_to_ref_trans_raw, K_ref = read_rgbd(depth_ref_path, image_ref_path, pose_ref_path)
    world_to_ref_trans = world_to_ref_trans_raw + world_to_ref_rot.dot(offset)
    line3ds, line3d_ids, line2ds_raw, line2ds_ref, image_ref = line_map.locate_rgbd(image_query, depth_ref, image_ref, world_to_ref_rot, world_to_ref_trans, K_ref, False)
    print(" - # lines :", len(line2ds_raw))

    def world_pt_to_ref_pixel(pt):
        pt_cam = world_to_ref_rot.dot(pt) + world_to_ref_trans
        pixel = K_ref.dot((1.0 / pt_cam[2]) * pt_cam)
        # print(pt_cam, pixel, pt)
        return (int(pixel[0]), int(pixel[1]))

    # use recover matrix to fix the line2ds
    for line2d in line2ds_raw:
        p1 = line2d.start
        p2 = line2d.end
        p1_corr = recover_matrix.dot(np.array([p1[0], p1[1], 1.0], dtype=np.float64))
        p2_corr = recover_matrix.dot(np.array([p2[0], p2[1], 1.0], dtype=np.float64))
        start = np.array([p1_corr[0], p1_corr[1]], dtype=np.float64)
        end = np.array([p2_corr[0], p2_corr[1]], dtype=np.float64)
        line2ds.append(_base.Line2d(start, end))

    final_pose, ransac_stats = _estimators.pl_estimate_absolute_pose(
            ransac_cfg, line3ds, line3d_ids, line2ds, point_3ds, point_2ds, camera, silent=True)

    log = "RANSAC stats: \n"
    log += f"  - num_iterations_total: {ransac_stats.num_iterations_total}\n"
    log += f"  - best_num_inliers: {ransac_stats.best_num_inliers}\n"
    log += f"  - best_model_score: {ransac_stats.best_model_score}\n"
    log += f"  - inlier_ratios (Points, Lines): {ransac_stats.inlier_ratios}\n"
    print(log)

    log = "Results Pose: \n"
    log += f"  - Result(P+L) Pose (qvec, tvec): {final_pose.qvec}, {final_pose.tvec}\n"
    print(log)

    recover_matrix_inv = np.linalg.inv(recover_matrix)
    def transform_pixel(pt):
        res = recover_matrix_inv.dot(np.array([pt[0], pt[1], 1.0], dtype=np.float64))
        return (int(res[0]), int(res[1]))

    def world_pt_to_qry_pixel(pt):
        pt_cam = final_pose.R().dot(pt) + final_pose.tvec
        pixel = transform_pixel(K.dot((1.0 / pt_cam[2]) * pt_cam))
        return pixel

    save_image = True
    if save_image:
        # render the depth
        depth_int = 2 * np.expand_dims(depth_ref, axis=2).astype(np.uint8)
        depth_int[depth_int == 0] = 255
        depth_color = cv2.applyColorMap(depth_int, cv2.COLORMAP_JET)
        image_ref = (0.5 * depth_color + 0.5 * image_ref).astype(np.uint8)

        # project the map points to the image
        for i in range(len(point_2ds)):
            pixel_1 = transform_pixel(point_2ds[i])
            cv2.circle(image_query, pixel_1, 4, (255, 0, 0), -1)
            pixel_2 = world_pt_to_qry_pixel(np.array(point_3ds[i]))
            cv2.circle(image_query, pixel_2, 4, (0, 0, 255), -1)
            cv2.line(image_query, pixel_1, pixel_2, (255, 0, 0), 2)
            pixel_ref = world_pt_to_ref_pixel(np.array(point_3ds[i]))
            cv2.circle(image_ref, pixel_ref, 6, (0, 255, 255), -1)

        colors = matplotlib.cm.hsv(np.random.rand(len(line2ds))) * 125
        colors_light = (2 * colors).tolist()
        colors = colors.tolist()
        for i in range(len(line2ds)):
            line_2d = line2ds[i]
            cv2.line(image_query, transform_pixel(line_2d.start), transform_pixel(line_2d.end), colors[i], 6)
            line_2d_ref = line2ds_ref[i]
            cv2.line(image_ref, (int(line_2d_ref.start[0]), int(line_2d_ref.start[1])), (int(line_2d_ref.end[0]), int(line_2d_ref.end[1])), colors_light[i], 3)
            line_3d = line3ds[line3d_ids[i]]
            pixel_a = world_pt_to_qry_pixel(line_3d.start)
            pixel_b = world_pt_to_qry_pixel(line_3d.end)
            cv2.line(image_query, pixel_a, pixel_b, colors_light[i], 2)

        new_width = image_query.shape[1] * image_ref.shape[0] / image_query.shape[0]
        image_query_resized = cv2.resize(image_query, (int(new_width), image_ref.shape[0]))
        image_show = cv2.hconcat([image_query_resized, image_ref])

        print_offset = 0
        print_interval = 30
        def print_log(text):
            cv2.putText(image_show, text, (10, print_offset), cv2.FONT_HERSHEY_DUPLEX, 1, (50, 50, 50), 2, cv2.LINE_AA)
            cv2.putText(image_show, text, (10, print_offset), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 100), 1, cv2.LINE_AA)
        print_offset = print_offset + print_interval
        print_log("#line matches :" + str(len(line2ds_raw)))
        print_offset = print_offset + print_interval
        print_log("#point matches :" + str(len(line2ds)))
        print_offset = print_offset + print_interval
        print_log(f"  - num_iterations_total: {ransac_stats.num_iterations_total}")
        print_offset = print_offset + print_interval
        print_log(f"  - best_num_inliers: {ransac_stats.best_num_inliers}")
        print_offset = print_offset + print_interval
        print_log(f"  - best_model_score: {ransac_stats.best_model_score}")
        print_offset = print_offset + print_interval
        print_log(f"  - inlier_ratios (Points, Lines): {ransac_stats.inlier_ratios}")
        cv2.imwrite(line_image_path, image_show)

    # save the result to file
    with open(output_file_path, 'w') as output_file:
        # transform pose back to world coordinate
        rot_mat = final_pose.R().astype(np.float64)
        tvec = final_pose.tvec.astype(np.float64) - rot_mat.dot(offset)

        output_file.write(str(rot_mat[0][0]) + " " + str(rot_mat[0][1]) + " " + str(rot_mat[0][2]) + '\n')
        output_file.write(str(rot_mat[1][0]) + " " + str(rot_mat[1][1]) + " " + str(rot_mat[1][2]) + '\n')
        output_file.write(str(rot_mat[2][0]) + " " + str(rot_mat[2][1]) + " " + str(rot_mat[2][2]) + '\n')
        output_file.write(str(tvec[0]) + " " + str(tvec[1]) + " " + str(tvec[2]) + '\n')

        # write the line observations
        output_file.write(str(len(line2ds)) + '\n')
        for i in range(len(line2ds)):
            p1 = line2ds[i].start
            p2 = line2ds[i].end
            output_file.write(str(p1[0]) + " " + str(p1[1]) + " " + str(p2[0]) + " " + str(p2[1]) + '\n')
            p3 = line3ds[line3d_ids[i]].start + offset
            p4 = line3ds[line3d_ids[i]].end + offset
            output_file.write(str(p3[0]) + " " + str(p3[1]) + " " + str(p3[2]) + " ")
            output_file.write(str(p4[0]) + " " + str(p4[1]) + " " + str(p4[2]) + '\n')


# Called when a client sends a message
def message_received(client, server, message):
    if len(message) > 200:
        message = message[:200]+'..'
    print("Client(%d) said: %s" % (client['id'], message))
    process_localization()
    return_message = "received your message : " + message
    server.send_message(client, return_message)
    print("Send message :", return_message)


def main():
    PORT=56789
    print("Start websocket server at", PORT)
    server = WebsocketServer(port = PORT)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    server.run_forever()


def parse_args():
    arg_parser = argparse.ArgumentParser(description='Line Localization Server')
    arg_parser.add_argument('--use_line', type=bool, default=True,
                            help='Whether to add line measurements, default: %(default)s')
    arg_parser.add_argument('--test', type=bool, default=False,
                            help='Whether to test, default: %(default)s')
    arg_parser.add_argument('--ransac_threshold', type=float, default=18.0,
                            help='Threshold for RANSAC/Solver first RANSAC, default: %(default)s')
    arg_parser.add_argument('--base_folder', type=Path, default="/home/viki/Development/Alpha/Data/request",
                            help='base folder to read request, default: %(default)s')
    args, unknown = arg_parser.parse_known_args()
    global base_folder
    base_folder = args.base_folder
    return args


def read_depth_test():
	depth_path = "/home/viki/Development/Alpha/Data/request/ref_depth.bin"
	image_path = "/home/viki/Development/Alpha/Data/request/ref_image.jpg"
	pose_path = "/home/viki/Development/Alpha/Data/request/ref_pose.txt"
	depth, image, world_to_ref_rot, world_to_ref_trans, K = read_rgbd(depth_path, image_path, pose_path)
	print("world_to_ref_rot:")
	print(world_to_ref_rot)
	print("world_to_ref_trans:")
	print(world_to_ref_trans)
	print("K:")
	print(K)

	# show the images
	depth_int = 2 * np.expand_dims(depth, axis=2).astype(np.uint8)
	depth_int[depth_int == 0] = 255
	depth_color = cv2.applyColorMap(depth_int, cv2.COLORMAP_JET)
	image_show = (0.5 * depth_color + 0.5 * image).astype(np.uint8)

	plt.figure()
	plt.imshow(image_show)
	plt.title('depth image')
	plt.show()


if __name__ == '__main__':
    args = parse_args()

    # read_depth_test()
    if args.use_line:
        line_map.load_model()

    ransac_cfg['line_cost_func'] = "PerpendicularDist"
    ransac_cfg['epipolar_filter'] = False
    ransac_cfg['IoU_threshold'] = 0.2
    ransac_cfg['optimize'] = {'loss_func': 'HuberLoss', 'loss_func_args': [2.0], 'normalize_weight': False}
    ransac_cfg['ransac'] = {}
    ransac_cfg['ransac']['solver_flags'] = [True, True, True, True]
    ransac_cfg['ransac']['min_num_iterations'] = 100
    ransac_cfg['ransac']['final_least_squares'] = True
    ransac_cfg['ransac']['method'] = "hybrid"
    ransac_cfg['ransac']['thres'] = args.ransac_threshold
    ransac_cfg['ransac']['thres_point'] = args.ransac_threshold
    ransac_cfg['ransac']['thres_line'] = args.ransac_threshold
    ransac_cfg['ransac']['weight_point'] = 1.0
    ransac_cfg['ransac']['weight_line'] = 1.0

    print("ransac_cfg: ", ransac_cfg)
    if args.test:
        process_localization()
    else:
        main()
