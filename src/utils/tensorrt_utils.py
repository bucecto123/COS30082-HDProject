
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTInference:
    def __init__(self, trt_engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(trt_engine_path)
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.get_binding_dtype(binding).itemsize
            host_mem = cuda.pagelocked_empty(size // self.engine.get_binding_dtype(binding).itemsize, dtype=trt.nptype(self.engine.get_binding_dtype(binding)))
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def _load_engine(self, trt_engine_path):
        with open(trt_engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data):
        # Transfer input data to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])

        # Execute inference
        self.context.execute_v2(bindings=self.bindings)

        # Transfer output data to host
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])

        return self.outputs[0]['host'].reshape(self.outputs[0]['host'].shape)
