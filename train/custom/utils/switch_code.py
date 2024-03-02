import tarfile
import argparse
import glob
import os
import pathlib
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar_version', type=str, default='MWResUnet-PM-LSW')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    tar_version = pathlib.Path(args.tar_version)
    tar_path = "./Logs" / tar_version / "project.tar"
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path='.')
    