from calendar import c
from dis import dis
from os.path import abspath, dirname
from typing import IO, Dict

import numpy as np
import torch
import yaml

from train.config.reconstruction_config import network_cfg
import tensorrt as trt
import pycuda.driver as pdd
import pycuda.autoinit

import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from train.custom.utils.common_tools import normlize, complex_to_multichannel

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class ReconstructionConfig:

    def __init__(self, test_cfg):
        # 配置文件
        self.patch_size = test_cfg.get('patch_size')

    def __repr__(self) -> str:
        return str(self.__dict__)


class ReconstructionModel:

    def __init__(self, model_f: IO, config_f):
        # TODO: 模型文件定制
        self.model_f = model_f 
        self.config_f = config_f
        self.network_cfg = network_cfg


class ReconstructionPredictor:

    def __init__(self, device: str, model: ReconstructionModel):
        self.device = torch.device(device)
        self.model = model
        self.tensorrt_flag = False 

        with open(self.model.config_f, 'r') as config_f:
            self.test_cfg = ReconstructionConfig(yaml.safe_load(config_f))
        self.network_cfg = model.network_cfg
        self.load_model()

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            # 根据后缀判断类型
            if self.model.model_f.endswith('.pth'):
                self.load_model_pth()
            elif self.model.model_f.endswith('.pt'):
                self.load_model_jit()
            elif self.model.model_f.endswith('.engine'):
                self.tensorrt_flag = True
                self.load_model_engine()

    def load_model_jit(self) -> None:
        # 加载静态图
        from torch import jit
        self.net = jit.load(self.model.model_f, map_location=self.device)
        self.net.eval()
        self.net.to(self.device)
    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.network
        checkpoint = torch.load(self.model.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device)

    def load_model_engine(self) -> None:
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.model.model_f, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = pdd.Stream()
        for i, binding in enumerate(engine):
            size = trt.volume(context.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = pdd.pagelocked_empty(size, dtype)
            device_mem = pdd.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def trt_inference(self, context, bindings, inputs, outputs, stream, batch_size):
        # Transfer input data to the GPU.
        [pdd.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [pdd.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def predict(self, inputs):
        img, sample_mask = inputs
        reconstruction = self._forward([img, sample_mask])
        return reconstruction

    def _forward(self, inputs):
        img, sample_mask = inputs
        img = torch.from_numpy(img)
        img = img[None]
        sample_mask = torch.from_numpy(sample_mask)
        sample_mask = sample_mask[None]
        # tensorrt预测
        if self.tensorrt_flag:
            cuda_ctx = pycuda.autoinit.context
            cuda_ctx.push()
            # 动态输入
            img = np.ascontiguousarray(img.numpy())
            self.context.active_optimization_profile = 0
            origin_inputshape = self.context.get_binding_shape(0)
            origin_inputshape[0], origin_inputshape[1], origin_inputshape[2], origin_inputshape[3], origin_inputshape[4] = img.shape
            # 若每个输入的size不一样，可根据inputs的size更改对应的context中的size
            self.context.set_binding_shape(0, (origin_inputshape))  
            inputs, outputs, bindings, stream = self.allocate_buffers(self.engine, self.context)
            inputs[0].host = img
            trt_outputs = self.trt_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs,stream=stream, batch_size=1)
            if cuda_ctx:
                cuda_ctx.pop()
            shape_of_output = [1, 1, 320, 320]
            reconstruction = trt_outputs[0].reshape(shape_of_output)
            reconstruction = torch.from_numpy(reconstruction)
        else:
            # pytorch预测
            with torch.no_grad():
                img_gpu = img.to(self.device)
                sample_mask_gpu = sample_mask.to(self.device)
                reconstruction = self.net([img_gpu, sample_mask_gpu])
                output = reconstruction[-2]

        output = output.squeeze().cpu().detach().numpy()
        return output


    
