import onnxruntime as ort
import numpy as np
import time
import subprocess


class ONNXRunner():
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

    def do_inference(self, input):
        input = np.expand_dims(input, axis=0)

        start_time = time.time()
        output = self.session.run(None, {self.session.get_inputs()[0].name: input})
        end_time = time.time()

        return [np.array(output), end_time - start_time]

    def to_tf(self, tf_path):
        command = ['onnx2tf', '-i', self.model_path, '-osd', '-o', tf_path]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'Error in onn2tf conversion: {result.stderr}')
