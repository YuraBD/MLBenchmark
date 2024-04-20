import importlib
import numpy as np
import os
import sys


fr_runners_map = {
        'pytorch': {
            'module': 'pytorch_runner',
            'class': 'PytorchRunner'
        },
        'onnx': {
            'module': 'onnx_runner',
            'class': 'ONNXRunner'
        },
        'tflite': {
            'module': 'tflite_runner',
            'class': 'TFLiteRunner'
        },
        'tensorrt': {
            'module': 'trt_runner',
            'class': 'TRTRunner'
        },
    }


def get_runner_class(framework):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    runners_dir = os.path.abspath(os.path.join(script_dir, '..', 'runners'))
    sys.path.append(runners_dir)
    runner_module = importlib.import_module(fr_runners_map[framework]['module'])
    runner_class = getattr(runner_module, fr_runners_map[framework]['class'])
    return runner_class


def load_data(data_path):
    data = np.load(data_path)
    x_key, y_key = data.keys()
    return data[x_key], data[y_key]