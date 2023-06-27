import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loader import read_scene_colmap

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.util.config as cfgutils
import limap.runners

def run_scene_colmap(cfg, session_name):
    imagecols = read_scene_colmap(cfg, session_name)
    linetracks = limap.runners.line_triangulation(cfg, imagecols)
    return linetracks

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='triangulate 3d lines')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/triangulation/colmap.yaml', help='config file')
    arg_parser.add_argument('-s', '--session_name', type=str, default='', help='session_name')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/triangulation/default_colmap.yaml', help='default config file')

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    shortcuts['-sid'] = '--scene_id'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    return cfg, args.session_name

def main():
    cfg, session_name = parse_config()
    run_scene_colmap(cfg, session_name)

if __name__ == '__main__':
    main()
