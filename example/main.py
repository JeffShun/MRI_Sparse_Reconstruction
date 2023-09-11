import argparse
import glob
import os
import sys
import tarfile
import traceback
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor import ReconstructionModel, ReconstructionPredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Test MRI Reconstruction')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='../example/data/input/test_mini', type=str)
    parser.add_argument('--output_path', default='../example/data/output/Dunet-41', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/Dunet/41.pth'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./reconstruction.yaml'
    )
    args = parser.parse_args()
    return args


def inference(predictor: ReconstructionPredictor, img: np.ndarray):
    pred_array = predictor.predict(img)
    return pred_array

def save_img(img, save_path):
    input_sos, label, pred_img = img
    input_sos = (input_sos-input_sos.min())/(input_sos.max()-input_sos.min())*255
    label = (label-label.min())/(label.max()-label.min())*255
    pred_img = (pred_img-pred_img.min())/(pred_img.max()-pred_img.min())*255
    save_img = np.concatenate((input_sos, label, pred_img),1)
    save_img = Image.fromarray(save_img.astype(np.uint8))
    save_img.save(save_path)


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    model_reconstruction = ReconstructionModel(
        model_f=args.model_file,
        config_f=args.config_file,
    )
    predictor_reconstruction = ReconstructionPredictor(
        device=device,
        model=model_reconstruction,
    )

    os.makedirs(output_path, exist_ok=True)
    for f_name in tqdm(os.listdir(input_path)):
        f_path = os.path.join(input_path, f_name)
        with h5py.File(f_path, 'r') as f:
            full_sampling_img = f['full_sampling_img'][:]                # 320*320 -complex64
            full_sampling_kspace = f['full_sampling_kspace'][:]          # 15*320*320 -complex64
            random_sample_img = f['random_sample_img'][:]                # 320*320 -complex64
            random_sample_mask = f['random_sample_mask'][:]              # 320*320 -int
            sensemap = f['sensemap'][:]                                  # 15*320*320 -complex64

        input_img = np.abs(random_sample_img)
        label = np.abs(full_sampling_img)
        pid = f_name.replace(".npz", "")
        inputs = [random_sample_img, sensemap, random_sample_mask, full_sampling_kspace]
        pred_array = inference(predictor_reconstruction, inputs)
        save_img([input_img, label, pred_array], os.path.join(output_path, f'{pid}.png'))

        meta_data_dir = os.path.join(output_path, "meta_datas", pid)
        os.makedirs(meta_data_dir, exist_ok=True)
        np.save(meta_data_dir + "/pred.npy", pred_array)
        np.save(meta_data_dir + "/label.npy", label)

if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )