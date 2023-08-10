import argparse
import glob
import os
import sys
import tarfile
import traceback

import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor import ReconstructionModel, ReconstructionPredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Test MRI Reconstruction')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_dicom_path', default='../example/data/input_test', type=str)
    parser.add_argument('--output_path', default='../example/data/output_test', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/v1/200.pth'
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
    img = (img-img.min())/(img.max()-img.min())*255
    image = Image.fromarray(img.astype(np.uint8))
    image.save(save_path)


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
        test_data= np.load(f_path, allow_pickle=True)

        acc_img = test_data['acc_img']
        sos_img = test_data['sos_img']
        random_sample_img_4 = test_data['random_sample_img_4']
        random_sample_img_8 = test_data['random_sample_img_8']
        eqs_sample_img_4 = test_data['eqs_sample_img_4']
        eqs_sample_img_8 = test_data['eqs_sample_img_8']
        pid = f_name.replace(".h5", "").replace("file","")
        img = np.concatenate((random_sample_img_4.real, random_sample_img_4.imag), axis=0)
        pred_array = inference(predictor_reconstruction, img)
        save_array = np.concatenate((sos_img, pred_array),1)
        save_img(save_array, os.path.join(output_path, f'{pid}.png'))


if __name__ == '__main__':
    args = parse_args()
    main(
        input_dicom_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )