import onnxruntime as ort
import numpy as np
import time
import subprocess


class ONNXRunner():
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

    def do_inference(self, single_input):
        single_input = np.expand_dims(single_input, axis=0)

        start_time = time.perf_counter()
        output = self.session.run(None, {self.session.get_inputs()[0].name: single_input})
        end_time = time.perf_counter()

        return [np.array(output), end_time - start_time]

    def to_tf(self, tf_path, do_not_keep_shape):
        if do_not_keep_shape:
            command = ['onnx2tf', '-i', self.model_path, '-osd', '-b', '1', '-o', tf_path]
        else:
            command = ['onnx2tf', '-i', self.model_path, '-osd', '-kat', self.session.get_inputs()[0].name, '-b', '1', '-o', tf_path]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'Error in onn2tf conversion: {result.stderr} {result.stdout}')
