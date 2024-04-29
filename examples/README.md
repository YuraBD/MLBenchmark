# Examples
Here are included several examples to demonstrate usage of benchmarking tool

Before using them run in the root directory:
```bash
$ python3 setenv.py
```

## PyTorch example
In this example benchmarking tool is evaluating performance of PyTorch model
```bash
$ ./pytorch.sh
```

## PyTorch -> ONNX example
In this example benchmarking tool is converting PyTorch model to ONNX and then measuring performance using it
```bash
$ ./pytorch-onnx.sh
```

## PyTorch -> TensorFlow Lite example
In this example benchmarking tool is converting PyTorch model to TensorFlow Lite and then measuring performance using it
```bash
$ ./pytorch-tflite.sh
```

## TensorFlow -> TensorFlow Lite example
In this example TensorFlow model is converted to TensorFlow Lite and then performance is measured
```bash
$ ./tf-tflite.sh
```

## TensorFlow -> TensorFlow Lite ARM NN example
In this example TensorFlow model is converted to TensorFlow Lite model and inference is done with using ARM NN TFLite Delegate.
For this example it is needed to download pre-built binaries from here https://github.com/ARM-software/armnn/releases
The last tested and working version wheree 24.02 for Raspberry Pi 4 and 5, and 23.05 for Jetson Nan
```bash
$ ./tf-tflite_armnn.sh
```

## TensorFlow -> TensorFlow Lite quantized example
In this example TensorFlow model is converted to TensorFlow Lite applying full-integer quantizaton
```bash
$ ./tf-tflite_quantization.sh
```

## Coral example
In this example compiled for Edge TPU model is used to measure its performance on Coral USB Accelerator
```bash
$ ./coral.sh
```
