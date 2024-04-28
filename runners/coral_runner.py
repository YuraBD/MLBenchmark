import numpy as np
import time
from pycoral.utils import edgetpu
from pycoral.adapters import common


class CoralRunner():
    def __init__(self, model_path):
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]


    def do_inference(self, single_input):
        if self.input_details['dtype'] != single_input.dtype:
            input_scale, input_zero_point = self.input_details["quantization"]
            single_input = single_input / input_scale + input_zero_point
        single_input = np.expand_dims(single_input, axis=0).astype(self.input_details["dtype"])
        common.set_input(self.interpreter, single_input)
        start_time = time.perf_counter()
        self.interpreter.invoke()
        end_time = time.perf_counter()
        output = self.interpreter.get_tensor(self.output_details['index'])
        return [np.array(output), end_time - start_time]
