try:
    from tensorrt import parsers
    import tensorrt as trt
except:
    raise ImportError("Make sure you installed Tensorrt") 
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import time
from random import randint
from PIL import Image
import matplotlib.pyplot as plt #to show test case
import cv2
from IPython  import  get_ipython
import os 

#########################################################################
# Global pooling layer works by the method in this document theoretically,
# but TensorRT fix the size of input and every layer, with a problem you
# still can input different size image but output wrong results, so if you
# want to test images in various sizes with correct results, you need to
# change the image size in prototxt every time before you import your model 
#########################################################################


INPUT_LAYER = 'data'
BREAK_LAYER = 'res5b'
OUTPUT_SIZE = 2
MODEL_NAME = ''
MODEL_PROTOTXT = './data/model_v1/model/porn_globalpool.prototxt'
CAFFE_MODEL = './data/model_v1/model/porn.caffemodel'


def preprocess(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224,224))
    
    img_array = []
    img_array.append(image.transpose(2,0,1).astype(float) - 128.0)
    #image_flip = cv2.flip(image, 1)
    #img_array.append(image_flip.transpose(2,0,1).astype(float) - 128.0)
    return np.asarray(img_array).astype(np.float32)


fc_weights = np.load('fc_w.npy').astype(np.float32)
fc_bias = np.zeros((2,), dtype = np.float32)

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
builder = trt.infer.create_infer_builder(G_LOGGER)
network = builder.create_network()
# TODO Parse model until break layer
parser = trt.parsers.caffeparser.create_caffe_parser()
model_datatype = trt.infer.DataType_kFLOAT
blob_name_to_tensor = parser.parse(MODEL_PROTOTXT, CAFFE_MODEL, network, model_datatype)
assert(blob_name_to_tensor)


# TODO dimension of break layer output / global pooling layer input
res5b = blob_name_to_tensor.find(BREAK_LAYER)
dims = res5b.get_dimensions().to_DimsCHW()
print dims.C(), dims.H(),dims.W()

# TODO set pooling window with corresponding size 
pool5 = network.add_pooling(res5b, trt.infer.PoolingType.AVERAGE, trt.infer.DimsHW(dims.H(),dims.W()))
assert(pool5)
dims_globalpool_blob = pool5.get_output(0).get_dimensions().to_DimsCHW()
print dims_globalpool_blob.C(), dims_globalpool_blob.H(), dims_globalpool_blob.W()


fc = network.add_fully_connected(pool5.get_output(0), OUTPUT_SIZE, trt.infer.Weights(fc_weights),trt.infer.Weights(fc_bias))
assert(fc)
final = network.add_softmax(fc.get_output(0))
final.get_output(0).set_name("prob")
network.mark_output(final.get_output(0))

#build the engine
builder.set_max_batch_size(1)
builder.set_max_workspace_size(1<<20)
engine = builder.build_cuda_engine(network)
assert(engine)

network.destroy()
parser.destroy()
builder.destroy()
trt.parsers.caffeparser.shutdown_protobuf_library()

# Test
context = engine.create_execution_context()
image_path = '49873b57e28061741175fd4ae2945127.jpg'
img=preprocess(image_path)
#input_mat = np.random.rand(3,280,280).astype(np.float32)
output = np.empty(OUTPUT_SIZE, dtype = np.float32)

d_input = cuda.mem_alloc(img.size*img.dtype.itemsize)
d_output = cuda.mem_alloc(output.size*output.dtype.itemsize)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()
cuda.memcpy_htod_async(d_input,img,stream)
context.enqueue(1,bindings,stream.handle,None)
cuda.memcpy_dtoh_async(output,d_output,stream)
stream.synchronize()

print output

context.destroy()
engine.destroy()




