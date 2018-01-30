try:
    from tensorrt import parsers
    import tensorrt as trt
except:
    raise ImportError("Make sure you installed Tensorrt") 
import pycuda.driver as cuda
#import pycuda.autoinit

import numpy as np
import time
from random import randint
from PIL import Image
import matplotlib.pyplot as plt #to show test case

from IPython  import  get_ipython

#G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)


INPUT_LAYERS = ['data']
OUTPUT_LAYERS = ['prob']
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000


MODEL_NAME='ResNet50'
MODEL_PROTOTXT = '/home/zhaohao/TensorRT-2.9.0/data/ResNet50/ResNet-50-deploy.prototxt'
CAFFE_MODEL = '/home/zhaohao/TensorRT-2.9.0/data/ResNet50/ResNet-50-model.caffemodel'
DATA = '/home/zhaohao/TensorRT-2.9.0/data/ResNet50/'
IMAGE_MEAN = '/home/zhaohao/TensorRT-2.9.0/data/ResNet50/ResNet_mean.binaryproto'



class Tensorrt_model(object):
    
    def __init__(self,Prototxt,Caffemodel):
        self.prototxt = Prototxt
        self.caffemodel = Caffemodel
        self.__glogger = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)       

    def set_batchsize(self,Batchsize):
        self.batchsize=Batchsize

    def build_engine(self,output_layers,floatmode):

        print 'begin building engine'
        self.__engine = trt.utils.caffe_to_TRT_engine(self.__glogger,
                                           self.prototxt,
                                           self.caffemodel,self.batchsize,
                                           1 <<20,output_layers,
                                           floatmode)
        print 'finish building engine \n'
        

    def set_gpu_id(self,gpu_id=0):

        # set cuda context on specific gpu 
        import pycuda.driver
        pycuda.driver.init()
        #global cuda_context
        ndevices = pycuda.driver.Device.count()
        if ndevices == 0:
            raise RuntimeError("No CUDA enabled device found")
                                
        if gpu_id in range(ndevices):
            dev = pycuda.driver.Device(gpu_id)
            self.cuda_context=dev.make_context()
         
           
    def create_runtime(self):
        runtime=trt.infer.create_infer_runtime(self.__glogger) # used to deserialize engine, no need for context
        return runtime
    def create_context(self):
        self.context=self.__engine.create_execution_context()
           
    def infer(self, input_img,output_size,num_binding):
        #self.runtime=self.create_runtime()
        #self.context=self.create_context()
       
        assert(self.__engine.get_nb_bindings() == num_binding)
        output = np.empty(output_size, dtype = np.float32)
       
        d_input = cuda.mem_alloc(self.batchsize *input_img.size* input_img.dtype.itemsize)
        d_output = cuda.mem_alloc(self.batchsize * output.size * output.dtype.itemsize)
        
        
        # pointers to gpu memory
        bindings = [int(d_input), int(d_output)]
        
        stream = cuda.Stream()
  
        #transfer input data to device
        cuda.memcpy_htod_async(d_input, input_img, stream)
        
        #execute model
        self.context.enqueue(self.batchsize, bindings, stream.handle, None)
        
        #transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
       
        #syncronize threads
        stream.synchronize()
       
        print 'all of activities in stream is done: {}'.format(stream.is_done())
        #destroy cuda context
        d_input.free()
        d_output.free()
        
        print 1999-cuda.mem_get_info()[0]/1048576,cuda.mem_get_info()[1]/1048576
       
        
        #self.context.destroy
        
        #self.runtime.destroy()


        return output

    def save_engine(self,path):
        trt.utils.write_engine_to_file(path, self.__engine.serialize())


    def load_engine(self,path):
        self.__engine = trt.utils.load_engine(self.__glogger, path)



    def destroy_all(self):
        self.context.destroy
        self.__engine.destroy()
        #global cuda_context
        self.cuda_context.pop()
        self.cuda_context=None
        
        print 'finish release memory'

def get_testcase(path):
    im=Image.open(path)
    #plt.imshow(im)
    #plt.show()
    assert(im)
    arr=np.array(im)
    arr=np.transpose(arr,(2,0,1))
    
    #arr = np.stack([np.asarray(i.resize((224, 224), Image.ANTIALIAS)) for i in im])
    #convert input data to Float32
    img = arr.astype(np.float32)
    return np.ascontiguousarray(img)


def get_meanimage(path):
    parser = parsers.caffeparser.create_caffe_parser()
    mean_blob = parser.parse_binary_proto(path)
    parser.destroy()
    mean = mean_blob.get_data(3*INPUT_W ** 2)
    mean_blob.destroy()    
    return mean




def main():

    Resnet101=Tensorrt_model(MODEL_PROTOTXT,CAFFE_MODEL)
    Resnet101.set_gpu_id(0)
    Resnet101.set_batchsize(1)
    
    #Resnet101.build_engine(OUTPUT_LAYERS,'FP32')
    Resnet101.load_engine('/home/zhaohao/TensorRT-2.9.0/new.engine')
    
    #Resnet101.create_runtime()
   
    #context is used to infer, one engine can use many contexts for different batches to infer at same time by sharing same weights 
    Resnet101.create_context()
       
    Test_Object='cat'
    path = DATA + Test_Object + '.jpeg'
    img=get_testcase(path)
    
    starttime=time.time()
    
    output=Resnet101.infer(img,1000,2)
    
    print("Test Case: " + Test_Object)
    print ("Prediction: " + str(np.argmax(output)))
    print("running time is {}".format(time.time()-starttime))

    #Resnet101.save_engine('/home/zhaohao/TensorRT-2.9.0/new.engine')
    Resnet101.destroy_all()

    return 0

if __name__=="__main__":
    main()
