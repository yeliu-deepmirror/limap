import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from read_write_model import read_model, qvec2rotmat, get_intrinsics_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.base as _base


def read_scene_colmap(cfg, session_name):
    colmap_path = os.path.join(cfg["data_dir"], session_name, cfg["sfm"]["colmap_output_path"], "sparse")
    print("[COLMAP LOADER] load mode from", colmap_path)
    cameras_raw, images_raw, points3D_raw = read_model(colmap_path)

    cameras = {}
    for camera_id in cameras_raw.keys():
        camera = cameras_raw[camera_id]
        img_hw = [camera.height, camera.width]
        K = get_intrinsics_matrix(camera)
        cameras[camera_id] = _base.Camera("PINHOLE", K, cam_id=camera_id, hw=img_hw)

    camimages = {}
    # add images
    stride = cfg["input_stride"]
    cnt = 1
    for image_id in images_raw.keys():
        cnt = cnt + 1
        if cnt%stride != 0:
            continue
        image = images_raw[image_id]
        pose = _base.CameraPose(qvec2rotmat(image.qvec), image.tvec)
        imname = os.path.join(cfg["data_dir"], session_name, cfg["sfm"]["colmap_output_path"], "images", image.name)
        camimage = _base.CameraImage(image.camera_id, pose, image_name=imname)
        camimages[image_id] = camimage

    imagecols = _base.ImageCollection(cameras, camimages)

    return imagecols
