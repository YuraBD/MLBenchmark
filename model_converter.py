class ModelConverter:
    def __init__(self) -> None:
        pass

    def tf_to_tflite(self, model_path: str) -> int:
        pass

    def tf_to_onnx(self, model_path: str) -> int:
        pass

    def _onnx_to_tf(self, model_path: str) -> int:
        pass

    def onnx_to_tflite(self, model_path: str) -> int:
        pass

    def onnx_to_tensorrt(self, model_path: str) -> int:
        pass

    def pytorch_to_onnx(self, model_path: str) -> int:
        pass

    def pytorch_to_tflite(self, model_path: str) -> int:
        pass

    def pytorch_to_tensorrt(self, model_path: str) -> int:
        pass
