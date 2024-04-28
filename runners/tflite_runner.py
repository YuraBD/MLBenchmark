import tensorflow as tf
import numpy as np
import time


class TFLiteRunner():
    def __init__(self, model_path, delegate_path = None, delegate_options = None, convert = False, quantization = False):
        if not convert:
            self.interpreter = self.initialize_interpreter(model_path, delegate_path, delegate_options)
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
        self.quantization = quantization

    def initialize_interpreter(self, model_path, delegate_path, delegate_options):
        print(delegate_path)
        if delegate_path:
            delegate = [tf.lite.experimental.load_delegate(library=delegate_path,options=delegate_options)]
        else:
            delegate = []

        interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegate)
        interpreter.allocate_tensors()
        return interpreter

    def do_inference(self, single_input):
        if self.quantization and self.input_details['dtype'] != single_input.dtype:
            input_scale, input_zero_point = self.input_details["quantization"]
            single_input = single_input / input_scale + input_zero_point
        single_input = np.expand_dims(single_input, axis=0).astype(self.input_details["dtype"])
        self.interpreter.set_tensor(self.input_details['index'], single_input)

        start_time = time.perf_counter()
        self.interpreter.invoke()
        end_time = time.perf_counter()

        output = self.interpreter.get_tensor(self.output_details['index'])

        return [np.array(output), end_time - start_time]

    def from_saved_model(self, tf_path, tflite_path, model_type='classification', representative_data_gen=None):
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        if self.quantization:
            print('Using full int quantization')
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = [tf.int8]
            if model_type == 'classification':
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            tflite_model = converter.convert()

        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)