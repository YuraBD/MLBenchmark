import os
import numpy as np
import common.common as common


class ModelConverter():
    def __init__(self, model_path, dir = 'converted_models', data_path = None,
                 model_class_path = None, model_class_name = None, input_shape = None):
        model_name = (model_path.split('/')[-1]).split('.')[0]
        self.conv_model_path = dir + '/' + model_name
        self.model_path = model_path
        self.model_class_path = model_class_path
        self.model_class_name = model_class_name
        self.input_shape = input_shape
        self.data_path = data_path
        os.makedirs(dir, exist_ok=True)

    def convert(self, origin_fr, target_fr):
        conv_map = {
            'pytorch': {
                'tflite': self.pytorch_to_tflite,
                'onnx': self.pytorch_to_onnx,
                'tensorrt': self.pytorch_to_trt
            },
            'onnx': {
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
            return conv_map[origin_fr][target_fr](True), self.data_path
        except KeyError:
            if origin_fr not in conv_map:
                raise Exception(f'Conversion from {origin_fr} is not supported')
            else:
                raise Exception(f'Conversion to {target_fr} is not supported')

    def pytorch_to_tflite(self, *args):
        self.pytorch_to_onnx()
        return self.onnx_to_tflite()

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
        onnx_runner.to_tf(self.conv_model_path)

        if not self.data_path:
            raise Exception('data_path must be defined for this conversion')
        data = np.load(self.data_path)
        x_key, y_key = data.keys()
        x_data = data[x_key].transpose(0,2,3,1)
        y_data = data[y_key]
        conv_data_path = self.conv_model_path + '_data.npz'
        np.savez_compressed(conv_data_path, x = x_data, y = y_data)
        self.data_path = conv_data_path
        return self.conv_model_path

    def tf_to_onnx(self, origin = False):
        pass

    def tf_to_tflite(self, origin = False):
        if origin:
            model_path = self.model_path
        else:
            model_path = self.conv_model_path

        TFLiteRunner = common.get_runner_class('tflite')
        tflite_runner = TFLiteRunner(None, convert = True)
        tflite_runner.from_saved_model(model_path, self.conv_model_path + '.tflite')
        return self.conv_model_path + '.tflite'


    def tf_to_trt(self, origin = False):
        self.tf_to_onnx(origin)
        return self.onnx_to_trt()
