import numpy as np
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import os


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTRunner():
    def __init__(self, engine_path, convert = False):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if not convert:
            self.engine = self.load_engine(engine_path)
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
            self.batch_size = 1

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, single_input):
        self.inputs[0].host = single_input
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        start_time = time.perf_counter()
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        end_time = time.perf_counter()
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        return [np.array([out.host for out in self.outputs]), end_time - start_time]

    def build_engine(self, onnx_path, engine_path):
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)

        config = builder.create_builder_config()
        builder.max_batch_size = 1

        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build the engine.")
            return None

        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
