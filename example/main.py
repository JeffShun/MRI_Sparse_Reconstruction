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
    parser.add_argument('--input_path', default='../example/data/input_test', type=str)
    parser.add_argument('--output_path', default='../example/data/output_test', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/v1/26.pth'
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
            acc_img = f['acc_img'][:]
            sos_img = f['sos_img'][:]
            random_sample_img_4 = f['random_sample_img_4'][:]
            random_sample_img_8 = f['random_sample_img_8'][:]
            eqs_sample_img_4 = f['eqs_sample_img_4'][:]
            eqs_sample_img_8 = f['eqs_sample_img_8'][:]

        input_img = random_sample_img_4
        input_img_sos = np.sqrt(np.sum(np.abs(input_img)**2, axis=0))
        pid = f_name.replace(".npz", "")
        pred_array = inference(predictor_reconstruction, input_img)
        save_img([input_img_sos, sos_img, pred_array], os.path.join(output_path, f'{pid}.png'))

        meta_data_dir = os.path.join(output_path, "meta_datas", pid)
        os.makedirs(meta_data_dir, exist_ok=True)
        np.save(meta_data_dir + "/pred.npy", pred_array)
        np.save(meta_data_dir + "/label.npy", sos_img)

if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )