import tensorflow as tf
import numpy as np
import time


class TFLiteRunner():
    def __init__(self, model_path, delegate_path = None, delegate_options = None, convert = False):
        if not convert:
            self.interpreter = self.initialize_interpreter(model_path, delegate_path, delegate_options)
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def initialize_interpreter(self, model_path, delegate_path, delegate_options):
        if delegate_path:
            delegate = tf.lite.experimental.load_delegate(library=delegate_path,options=delegate_options)
        else:
            delegate = []
        interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegate)
        interpreter.allocate_tensors()
        return interpreter

    def do_inference(self, input):
        self.interpreter.set_tensor(self.input_details[0]['index'], [input])

        start_time = time.time()
        self.interpreter.invoke()
        end_time = time.time()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        return [np.array(output), end_time - start_time]

    def from_saved_model(self, tf_path, tflite_path, quantization = None):
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        if quantization == 'full_int':
            print('Using full int quantization')
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # converter.representative_dataset = self.representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            tflite_model = converter.convert()

        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)