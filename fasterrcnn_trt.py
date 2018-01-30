import os
import sys
from random import randint
import numpy as np
import cv2
import time
import traceback
import matplotlib.pyplot as plt
try:
    from PIL import Image
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import argparse
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have pycuda and the example dependencies installed. 
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))
    exit(1)

try:
    import tensorrt as trt
    from tensorrt.parsers import caffeparser
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have the TensorRT Library installed 
and accessible in your LD_LIBRARY_PATH
""".format(err))
    exit(1) 

try:
    import tensorrtplugins
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import tensorrtplugins ({})
Please build and install the example custom layer wrapper
Follow the instructions in the README on how to do so""".format(err))
    exit(1)



CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

G_LOGGER = trt.infer.ColorLogger(trt.infer.LogSeverity.INFO)
INPUT_LAYERS = ["data","im_info"]
OUTPUT_LAYERS = ["cls_prob","bbox_pred","rois"]
INPUT_H = 375
INPUT_W =  500
OUTPUT_CLS_SIZE = 21
OUTPUT_BBOX_SIZE=OUTPUT_CLS_SIZE * 4


MODEL_PROTOTXT =  'TensorRT-3.0.0/data/faster-rcnn/faster_rcnn_test_iplugin.prototxt'
CAFFE_MODEL =   'TensorRT-3.0.0/data/faster-rcnn/VGG16_faster_rcnn_final.caffemodel'
DATA =   ''
IMAGE =  "TensorRT-3.0.0/data/faster-rcnn/000456.ppm"

NMS_MaxOut= 300


def caffe_to_trt_fasterrcnn(logger, deploy_file, model_file, max_batch_size, max_workspace_size, output_layers, datatype=trt.infer.DataType.FLOAT, plugin_factory=None, calibrator=None):
    
    #create the builder
    builder = trt.infer.create_infer_builder(logger)

    #parse the caffe model to populate the network
    network = builder.create_network()
    parser = caffeparser.create_caffe_parser()

    if plugin_factory:
        parser.set_plugin_factory(plugin_factory)

    model_datatype = trt.infer.DataType_kFLOAT
    
    blob_name_to_tensor = parser.parse(deploy_file, model_file, network, model_datatype)
    logger.log(trt.infer.LogSeverity.INFO, "Parsing caffe model {}, {}".format(deploy_file, model_file))
    try:
        assert(blob_name_to_tensor)
    except AssertionError:
        logger.log(trt.infer.LogSeverity.ERROR, "Failed to parse caffe model")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
    
        raise AssertionError('Caffe parsing failed on line {} in statement {}'.format(line, text))

    input_dimensions = {}

    for i in range(network.get_nb_inputs()):
        dims = network.get_input(i).get_dimensions().to_DimsCHW()
        logger.log(trt.infer.LogSeverity.INFO, "Input \"{}\":{}x{}x{}".format(network.get_input(i).get_name(), dims.C(), dims.H(), dims.W()))
        input_dimensions[network.get_input(i).get_name()] = network.get_input(i).get_dimensions().to_DimsCHW()

    if type(output_layers) is str:
        output_layers = [output_layers]
    #mark the outputs
    for l in output_layers:
        logger.log(trt.infer.LogSeverity.INFO, "Marking " + l + " as output layer")
        t = blob_name_to_tensor.find(l)
        try:
            assert(l)
        except AssertionError:
            logger.log(trt.infer.LogSeverity.ERROR, "Failed to find output layer {}".format(l))
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
    
            raise AssertionError('Caffe parsing failed on line {} in statement {}'.format(line, text))

        layer = network.mark_output(t)

   # for i in range(network.get_nb_outputs()):
   #     dims = network.get_output(i).get_dimensions().to_DimsNCHW()
        #print (dims.N(), dims.C(), dims.H(), dims.W())
   #     logger.log(tensorrt.infer.LogSeverity.INFO, "Output \"{}\":{}x{}x{}".format(network.get_output(i).get_name(), dims.C(), dims.H(), dims.W()))

    #build the engine
    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(max_workspace_size)

    engine = builder.build_cuda_engine(network)
    
    try:
        assert(engine)
    except AssertionError:
        logger.log(trt.infer.LogSeverity.ERROR, "Failed to create engine")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
    
        raise AssertionError('Caffe parsing failed on line {} in statement {}'.format(line, text))

    network.destroy()
    parser.destroy()
    builder.destroy()
    caffeparser.shutdown_protobuf_library()

    return engine


def infer(context, batch_size, input_img, im_info, outputCls_size, outputBbox_size):
    #load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 5)
    
    output_cls = np.empty([batch_size,  1,  NMS_MaxOut, outputCls_size], dtype = np.float32)
    output_bbox = np.empty([batch_size, 1,  NMS_MaxOut, outputBbox_size], dtype = np.float32)
    output_rois = np.empty([batch_size, 1, NMS_MaxOut, 4], dtype = np.float32)
    #output_reshape = np.empty([batch_size, 18, 25, 32], dtype = np.float32)    


    #alocate device memory
    d_input_img = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_input_imfo = cuda.mem_alloc(batch_size * 3 * im_info.dtype.itemsize)
    d_output_cls = cuda.mem_alloc(batch_size * NMS_MaxOut * outputCls_size * output_cls.dtype.itemsize)
    d_output_bbox = cuda.mem_alloc(batch_size * NMS_MaxOut *outputBbox_size * output_bbox.dtype.itemsize)
    d_output_rois = cuda.mem_alloc(batch_size * NMS_MaxOut * 4 * output_rois.dtype.itemsize)
  
    #d_output_reshape = cuda.mem_alloc(batch_size * 18*25* 32 * output_rois.dtype.itemsize)
    
    bindings = [int(d_input_img), int(d_input_imfo), int(d_output_cls), int(d_output_bbox), int(d_output_rois)]

    stream = cuda.Stream()

    #transfer input data to device
    cuda.memcpy_htod_async(d_input_img, input_img, stream)
    cuda.memcpy_htod_async(d_input_imfo, im_info, stream)

    #execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(output_cls, d_output_cls, stream)
    cuda.memcpy_dtoh_async(output_bbox, d_output_bbox, stream)
    cuda.memcpy_dtoh_async(output_rois, d_output_rois, stream)
    
    return output_cls, output_bbox, output_rois

def get_testcase(path):
    #im = cv2.resize(cv2.imread(path),(224,224),interpolation=cv2.INTER_CUBIC)
    #assert(im)
    im = cv2.imread(path)
    arr = np.array(im)
    arr = np.transpose(arr, (2,0,1))
    #assert(arr)
    #make array 1D
    #img = arr.ravel()
    return np.ascontiguousarray(arr, dtype=np.float32)
    

def apply_mean(img, mean):

    for i in range(len(img[:,0,0])):
        img[i,:] = img[i,:] - mean[i]

    return img
def bbox_inv_clip(rois, bbox_delta, batchsize, iminfo):
    
    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    dx = bbox_delta[:, 0::4]
    dy = bbox_delta[:, 1::4]
    dw = bbox_delta[:, 2::4]
    dh = bbox_delta[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(bbox_delta.shape, dtype=bbox_delta.dtype)
    # x1 >= 0
    pred_boxes[:, 0::4] = np.maximum(np.minimum(pred_ctr_x - 0.5 * pred_w, iminfo[1] - 1), 0)
    # y1 >= 0
    pred_boxes[:, 1::4] = np.maximum(np.minimum(pred_ctr_y - 0.5 * pred_h, iminfo[0] - 1), 0)
    # x2 < im_shape[1]
    pred_boxes[:, 2::4] = np.maximum(np.minimum(pred_ctr_x + 0.5 * pred_w, iminfo[1] - 1), 0)
    # y2 < im_shape[0]
    pred_boxes[:, 3::4] = np.maximum(np.minimum(pred_ctr_y + 0.5 * pred_h, iminfo[0] - 1), 0)
    return pred_boxes


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def vis_detections(im, class_name, dets, thresh=0.5):
    
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def main():
    plugin_factory = tensorrtplugins.FasterRCNNPluginFactory()
    engine = caffe_to_trt_fasterrcnn(G_LOGGER,
        MODEL_PROTOTXT,
        CAFFE_MODEL,
        1,
        15 << 20,
        OUTPUT_LAYERS,
        trt.infer.DataType.FLOAT,
        plugin_factory)
    
    Batch_Size = 1
    input_img = get_testcase(IMAGE)  #should be bgr and reduce mean
    im_info = np.array([375, 500, 1],dtype=np.float32) #all input must be float
    
    img_mean =np.array([102.9801, 115.9465, 122.7717], dtype=np.float32) 
    input_img = apply_mean(input_img, img_mean)

    context = engine.create_execution_context()
    starttime=time.time()
    output_cls, output_bbox, output_rois = infer(context, 1, input_img.ravel(), im_info, OUTPUT_CLS_SIZE, OUTPUT_BBOX_SIZE)
    print 'inference time is {}'.format(time.time()-starttime)

    context.destroy()
    engine.destroy()
    print np.shape(output_cls), np.shape(output_bbox), np.shape(output_rois)
    print output_rois
    im_scale = np.float32(1)
    for i in range(Batch_Size):
        output_rois[i,:]=output_rois[i,:] / im_scale

    pred_boxes = bbox_inv_clip(output_rois[0,0,:,:], output_bbox[0,0,:,:], Batch_Size, im_info)
    
    SCORE_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_index, cls in enumerate(CLASSES[1:]):
        cls_index +=1
        cls_boxes = pred_boxes[:, 4*cls_index:4*(cls_index + 1)]
        cls_scores = output_cls[0,0,:, cls_index]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        
        keep = nms(dets, NMS_THRESH)        
        dets = dets[keep, :]
        
        vis_detections(np.array(cv2.imread(IMAGE)), cls, dets, thresh=SCORE_THRESH)
        plt.show()
if __name__ == "__main__":
    main()
    
    #plugin_factory = tensorrtplugins.FasterRCNNPluginFactory()
    #engine = caffe_to_trt_fasterrcnn(G_LOGGER,
    #    MODEL_PROTOTXT,
    #    CAFFE_MODEL,
    #    1,
    #    16 << 20,
    #    OUTPUT_LAYERS,
    #    trt.infer.DataType.FLOAT,
    #    plugin_factory)
    
