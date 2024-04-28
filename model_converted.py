import os
import numpy as np
import common.common as common


class ModelConverter():
    def __init__(self, model_path, do_not_keep_shape, quantization = False, dir = 'converted_models',
                 model_class_path = None, model_class_name = None, input_shape = None,
                 model_type = 'classification', representative_data_path = None):
        if model_path[-1] == '/':
            model_path = model_path[:-1]
        model_name = (model_path.split('/')[-1]).split('.')[0]
        self.conv_model_path = dir + '/' + model_name
        self.model_path = model_path
        self.model_class_path = model_class_path
        self.model_class_name = model_class_name
        self.input_shape = input_shape
        self.do_not_keep_shape = do_not_keep_shape
        self.quantization = quantization
        self.model_type = model_type
        self.representative_data_path = representative_data_path
        os.makedirs(dir, exist_ok=True)

    def convert(self, origin_fr, target_fr):
        conv_map = {
            'pytorch': {
                'tensorflow': self.pytorch_to_tf,
                'tflite': self.pytorch_to_tflite,
                'onnx': self.pytorch_to_onnx,
                'tensorrt': self.pytorch_to_trt
            },
            'onnx': {
                'tensorflow': self.onnx_to_tf,
                'tflite': self.onnx_to_tflite,
                'tensorrt': self.onnx_to_trt
            },
            'tensorflow': {
                'onnx': self.tf_to_onnx,
                'tflite': self.tf_to_tflite,
                'tensorrt': self.tf_to_trt
            }
        }

        try:
            return conv_map[origin_fr][target_fr](True)
        except KeyError:
            if origin_fr not in conv_map:
                raise Exception(f'Conversion from {origin_fr} is not supported')
            else:
                raise Exception(f'Conversion to {target_fr} is not supported')

    def pytorch_to_tflite(self, *args):
        self.pytorch_to_onnx()
        return self.onnx_to_tflite()

    def pytorch_to_tf(self, *args):
        self.pytorch_to_onnx()
        return self.onnx_to_tf()

    def pytorch_to_onnx(self, *args):
        if not self.model_class_name or not self.model_class_path:
            raise Exception('model_class_name must be defined for this conversion')
        if not self.model_class_path:
            raise Exception('model_class_path must be defined for this conversion')
        if not self.input_shape:
            raise Exception('input_shape must be defined for this conversion')

        PytorchRunner = common.get_runner_class('pytorch')
        pytorch_runner = PytorchRunner(self.model_path, self.model_class_path, self.model_class_name)
        pytorch_runner.to_onnx(self.conv_model_path + '.onnx', self.input_shape)
        return self.conv_model_path + '.onnx'

    def pytorch_to_trt(self, *args):
        self.pytorch_to_onnx()
        return self.onnx_to_trt()

    def onnx_to_tflite(self, origin = False):
        self.onnx_to_tf(origin)
        return self.tf_to_tflite()

    def onnx_to_trt(self, origin = False):
        if origin:
            model_path = self.model_path
        else:
            model_path = self.conv_model_path + '.onnx'

        TRTRunner = common.get_runner_class('tensorrt')
        trt_runner = TRTRunner(None, convert = True)
        trt_runner.build_engine(model_path, self.conv_model_path + '.trt')
        return self.conv_model_path + '.trt'

    def onnx_to_tf(self, origin = False):
        if origin:
            model_path = self.model_path
        else:
            model_path = self.conv_model_path + '.onnx'

        ONNXRunner = common.get_runner_class('onnx')
        onnx_runner = ONNXRunner(model_path)
        onnx_runner.to_tf(self.conv_model_path, self.do_not_keep_shape)

        return self.conv_model_path

    def tf_to_onnx(self, origin = False):
        pass


    def representative_data_gen(self):
        x_data, _ = common.load_data(self.representative_data_path)
        for input_data in x_data:
            input_data = np.array(input_data)
            input_data = np.expand_dims(input_data, axis=0)
            yield [input_data]


    def tf_to_tflite(self, origin = False):
        if origin:
            model_path = self.model_path
        else:
            model_path = self.conv_model_path

        if self.quantization:
            conv_model_name = self.conv_model_path + '_quantized.tflite'
        else:
            conv_model_name = self.conv_model_path + '.tflite'
        TFLiteRunner = common.get_runner_class('tflite')
        tflite_runner = TFLiteRunner(None, convert = True, quantization = self.quantization)
        tflite_runner.from_saved_model(
            model_path, conv_model_name,
            model_type = self.model_type, representative_data_gen = self.representative_data_gen)

        return conv_model_name


    def tf_to_trt(self, origin = False):
        self.tf_to_onnx(origin)
        return self.onnx_to_trt()
