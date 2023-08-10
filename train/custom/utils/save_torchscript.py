"""生成torchscript很难直接从Config构建的模型进行转换，需要剥离出组件."""

import argparse
import os
import sys
import torch

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)
from config.reconstruction_config import network_cfg

def load_model(model_path):
    model = network_cfg.network
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/v1/30.pth')
    parser.add_argument('--output_path', type=str, default='./checkpoints')
    args = parser.parse_args()
    return args

# branch
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['IS_SCRIPTING'] = '1'
    args = parse_args()
    model = load_model(args.model_path)
    model_jit = torch.jit.script(model)
    model_jit.save(os.path.join(args.output_path, 'model.pt'))

