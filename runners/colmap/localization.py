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
import logging
import argparse
from pathlib import Path
from hloc.utils.read_write_model import *
from read_write_model import qvec2rotmat, get_intrinsics_matrix

from websocket_server import WebsocketServer
from line_map import LineMap

ransac_cfg = {}
request_cnt = 0
world_to_local_trans = np.array([0, 0, 0]).astype(np.float64)
world_to_local_rot = np.eye(3).astype(np.float64)


# Called for every client connecting (after handshake)
def new_client(client, server):
	print("[WEBSOCKET] New client connected and was given id %d" % client['id'])
	server.send_message_to_all("Hey all, a new client has joined us")


# Called for every client disconnecting
def client_left(client, server):
	print("[WEBSOCKET] Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):
    if len(message) > 200:
        message = message[:200]+'..'
    print("Client(%d) said: %s" % (client['id'], message))
    global request_cnt
    request_cnt = request_cnt + 1

    # read the request from file and run pnp
    points_file_path = "/home/viki/Development/Alpha/Data/request/tmp.txt"
    image_path = "/home/viki/Development/Alpha/Data/request/tmp.jpg"
    output_file_path = "/home/viki/Development/Alpha/Data/request/result.txt"
    image = cv2.imread(image_path)

    point_3ds = []
    point_2ds = []
    points_file = open(points_file_path, "r")
    points_size = int(points_file.readline())
    for i in range(points_size):
        line = points_file.readline()
        tmp = line.split(" ")
        # transform the point to local coordinate
        pt_world = np.array([0, 0, 0]).astype(np.float64)
        pt_world[0] = Decimal(float(tmp[0]))
        pt_world[1] = Decimal(float(tmp[1]))
        pt_world[2] = Decimal(float(tmp[2]))
        # transform by offset
        pt_local = world_to_local_rot.dot(pt_world) + world_to_local_trans
        point_3ds.append([pt_local[0], pt_local[1], pt_local[2]])
        point_2ds.append([float(tmp[3]), float(tmp[4])])

    line_intrin = points_file.readline()
    tmp = line_intrin.split(" ")
    K = np.array([[float(tmp[0]), 0, float(tmp[2])], [0, float(tmp[1]), float(tmp[3])], [0, 0, 1]]).astype(np.float32)
    camera = _base.Camera("PINHOLE", K, cam_id=request_cnt, hw=[image.shape[0], image.shape[1]])

    # find line matches
    # l3ds = data['l3ds']
    # l2ds = data['l2ds']
    # l3d_ids = data['l3d_ids']



    final_pose, ransac_stats = _estimators.pl_estimate_absolute_pose(
            ransac_cfg, [], [], [], point_3ds, point_2ds, camera, silent=True)

    log = "RANSAC stats: \n"
    log += f"  - num_iterations_total: {ransac_stats.num_iterations_total}\n"
    log += f"  - best_num_inliers: {ransac_stats.best_num_inliers}\n"
    log += f"  - best_model_score: {ransac_stats.best_model_score}\n"
    log += f"  - inlier_ratios (Points, Lines): {ransac_stats.inlier_ratios}\n"
    print(log)

    log = "Results Pose: \n"
    log += f"  - Result(P+L) Pose (qvec, tvec): {final_pose.qvec}, {final_pose.tvec}\n"
    print(log)

    # save the result to file
    with open(output_file_path, 'w') as output_file:
        # transform pose back to world coordinate
        rot_mat_pnp = final_pose.R().astype(np.float64)
        tvec_pnp = final_pose.tvec.astype(np.float64)
        # world_to_camera = local_to_camera * world_to_local
        rot_mat = rot_mat_pnp.dot(world_to_local_rot)
        tvec = rot_mat_pnp.dot(world_to_local_trans) + tvec_pnp

        output_file.write(str(rot_mat[0][0]) + " " + str(rot_mat[0][1]) + " " + str(rot_mat[0][2]) + '\n')
        output_file.write(str(rot_mat[1][0]) + " " + str(rot_mat[1][1]) + " " + str(rot_mat[1][2]) + '\n')
        output_file.write(str(rot_mat[2][0]) + " " + str(rot_mat[2][1]) + " " + str(rot_mat[2][2]) + '\n')
        output_file.write(str(tvec[0]) + " " + str(tvec[1]) + " " + str(tvec[2]))

    return_message = "received your message : " + message
    server.send_message(client, return_message)
    print("Send message :", return_message)


def test_image_line_match(args):
    image_query = cv2.imread("./outputs/0.jpg")
    # image_query = cv2.rotate(image_query, cv2.ROTATE_90_CLOCKWISE)
    line_map = LineMap(args.lines_dir)
    matches, image_show = line_map.match_image_with_query(image_query, "camera_back", "1669268767414037205", True)

    print("find", len(matches), "matches.")
    if image_show is not None:
        cv2.destroyAllWindows()
        cv2.imshow("lines in image", image_show)
        cv2.waitKey(0)


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
    arg_parser.add_argument('--lines_dir', type=Path, help='Path to lines file')
    arg_parser.add_argument('--ransac_threshold', type=float, default=12.0,
                            help='Threshold for RANSAC/Solver first RANSAC, default: %(default)s')
    args, unknown = arg_parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    ransac_cfg['line_cost_func'] = "PerpendicularDist"
    ransac_cfg['epipolar_filter'] = False
    ransac_cfg['optimize'] = {'loss_func': 'TrivialLoss', 'loss_func_args': [], 'normalize_weight': True}
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


    # read the offset
    offset_file_path = os.path.join(args.lines_dir, "colmap_outputs", "world_to_local.proto.txt")
    offset_file = open(offset_file_path, "r")

    offset_file.readline()
    world_to_local_trans[0] = Decimal(offset_file.readline()[5:])
    world_to_local_trans[1] = Decimal(offset_file.readline()[5:])
    world_to_local_trans[2] = Decimal(offset_file.readline()[5:])
    offset_file.readline()
    offset_file.readline()
    quad = np.zeros(4)
    quad[1] = float(offset_file.readline()[5:])
    quad[2] = float(offset_file.readline()[5:])
    quad[3] = float(offset_file.readline()[5:])
    quad[0] = float(offset_file.readline()[5:])
    world_to_local_rot = qvec2rotmat(quad).astype(np.float64)

    print("world_to_local_trans:\n", world_to_local_trans)
    print("world_to_local_rot:\n", world_to_local_rot)

    main()
