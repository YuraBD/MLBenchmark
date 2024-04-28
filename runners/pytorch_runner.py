import torch
import numpy as np
import sys
import os
import importlib
import time


class PytorchRunner():
    def __init__(self, model_path, model_class_path, model_class_name):
        self.device = torch.device('cpu')
        self.model = self.load_model(model_path, model_class_path, model_class_name)

    def load_model(self, model_path, model_class_path, model_class_name):
        absolute_path = os.path.abspath(model_class_path)
        module_dir, module_file = os.path.split(absolute_path)
        module_name, _ = os.path.splitext(module_file)
        if module_dir not in sys.path:
            sys.path.append(module_dir)

        module = importlib.import_module(module_name)
        model_class = getattr(module, model_class_name)

        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()
        return model

    def do_inference(self, single_input):
        input_tensor = torch.from_numpy(single_input).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model(input_tensor)
        end_time = time.perf_counter()

        return [np.array(output), end_time - start_time]

    def to_onnx(self, onnx_path, input_shape):
        sample_input = torch.rand(input_shape)
        torch.onnx.export(
            self.model,
            sample_input,
            onnx_path,
            opset_version=17,
            input_names=['input'],
            output_names=['output']
        )
