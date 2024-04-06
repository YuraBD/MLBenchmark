import numpy as np
import torch
import time
import os
import sys
import importlib
import tensorflow as tf
import onnxruntime as ort


class InferenceRunner:
    def __init__(self, model_path: str, data_path: str, framework: str, model_type: str = 'classification',
                 model_class_path: str = None, model_class_name: str = None) -> None:
        self.model_path = model_path
        self.model_type = model_type
        self.framework = framework
        self.imgs, self.labels = self._load_data(data_path)
        if framework == 'pytorch':
            if model_class_name == None:
                raise Exception("Model class name is not defined")
            if model_class_path == None:
                raise Exception("Model class path is not defined")
            self.model_class_name = model_class_name
            self.model_class_path = model_class_path


    def _load_data(self, data_path):
        data = np.load(data_path)
        return data['imgs'], data['labels']

    def get_data_len(self):
        return len(self.imgs)

    def run_inference(self):
        if self.framework == 'pytorch':
            return self._run_pytorch_inference()
        elif self.framework == 'tflite':
            return self._run_tflite_inference()
        elif self.framework == 'onnx':
            return self._run_onnx_inference()
        elif self.framework == 'tensorrt':
            return self._run_tensorrt_inference()
        elif self.framework == 'executorch':
            return self._run_executorch_inference()


    def _run_pytorch_inference(self):
        device = torch.device("cpu")

        absolute_path = os.path.abspath(self.model_class_path)
        module_dir, module_file = os.path.split(absolute_path)
        module_name, _ = os.path.splitext(module_file)
        if module_dir not in sys.path:
            sys.path.append(module_dir)

        module = importlib.import_module(module_name)
        model_class = getattr(module, self.model_class_name)

        model = model_class()
        model.load_state_dict(torch.load(self.model_path))
        model.to(device)
        model.eval()

        inference_times = []
        correct_predictions = 0
        for i, img in enumerate(self.imgs):
            tensor_img = torch.from_numpy(img).unsqueeze(0).to(device)

            start_time = time.time()
            with torch.no_grad():
                output = model(tensor_img)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            if (isinstance(list(model.modules())[-1], torch.nn.Softmax)):
                probabilities = output
            else:
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            _, top1_catid = torch.topk(probabilities, 1)

            if self.labels[i] == top1_catid:
                correct_predictions += 1

        return inference_times, correct_predictions


    def _run_tflite_inference(self):
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        inference_times = []
        correct_predictions = 0
        for i, img in enumerate(self.imgs):
            if input_details[0]['dtype'] == np.float32:
                input_data = np.array(img, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], [input_data])

            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()
            inference_times.append(end_time - start_time)

            output_data = interpreter.get_tensor(output_details[0]['index'])

            # probabilities = tf.nn.softmax(output_data[0])
            probabilities = output_data
            predicted_class = np.argmax(probabilities)

            if self.labels[i] == predicted_class:
                correct_predictions += 1

        return inference_times, correct_predictions


    def _run_onnx_inference(self):
        session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

        inference_times = []
        correct_predictions = 0

        for i, img in enumerate(self.imgs):
            input_data = np.expand_dims(img, axis=0).astype(np.float32)

            start_time = time.time()
            output_data = session.run(None, {session.get_inputs()[0].name: input_data})
            end_time = time.time()
            inference_times.append(end_time - start_time)

            probabilities = tf.nn.softmax(output_data[0][0])
            predicted_class = np.argmax(probabilities)

            if self.labels[i] == predicted_class:
                correct_predictions += 1

        return inference_times, correct_predictions


    def _run_tensorrt_inference(self):
        pass


    def _run_executorch_inference(self):
        pass

