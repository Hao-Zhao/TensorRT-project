# -*- coding:utf-8 -*-
from flask import Flask, request, json
from Tensorrt_Class import Tensorrt_model,get_testcase
import tensorrt as trt
from tensorrt import parsers
import pycuda.driver as cuda
#import pycuda.autoinit
import os
import sqlite3

import numpy as np
import os
import time
from random import randint
from PIL import Image
import matplotlib.pyplot as plt #to show test case
from IPython  import  get_ipython

INPUT_LAYERS = ['data']
OUTPUT_LAYERS = ['prob']
OUTPUT_SIZE = 1000

MODEL_PROTOTXT = '/home/zhaohao/TensorRT-2.9.0/data/ResNet50/ResNet-50-deploy.prototxt'
CAFFE_MODEL = '/home/zhaohao/TensorRT-2.9.0/data/ResNet50/ResNet-50-model.caffemodel'
DATA = '/home/zhaohao/TensorRT-2.9.0/data/ResNet50/'

hao = Flask(__name__)

Resnet101=Tensorrt_model(MODEL_PROTOTXT,CAFFE_MODEL)
Resnet101.set_batchsize(1)
    

#load caffe model into tensorrt engine  
Resnet101.build_engine(OUTPUT_LAYERS,'FP32')



#Resnet101.create_runtime()
  
Resnet101.create_context()
Resnet101.set_gpu_id(0)

db = sqlite3.connect('tensorrtdb')
cursor = db.cursor()

# create datebase 
#cursor.execute(''' CREATE TABLE tensorrt_models(model_id,num_mission,gpu_id)''')
#model_status=[(0,0,'gpu0'),
#              (1,0,'gpu1'),
#              (2,0,'gpu2')]
#cursor.executemany("INSERT INTO tensorrt_models  VALUES (?,?,?)",model_status)


def set_model(list_mission):
    if all(x is list_mission[0] for x in list_mission):
        return 0
    else:
        return list_mission.index(min(list_mission))

def mission_sta(num_id):
    cursor.execute('''SELECT num_mission FROM tensorrt_models  WHERE model_id = ?''',(num_id,))
    num_misn=cursor.fetchone()
    cursor.execute('''UPDATE tensorrt_models SET num_mission = ? WHERE model_id = ? ''',(num_misn[0]+1,num_id))
    db.commit()

def mission_fin(num_id):
    cursor.execute('''SELECT num_mission FROM tensorrt_models  WHERE model_id = ?''',(num_id,))
    num_misn=cursor.fetchone()
    cursor.execute('''UPDATE tensorrt_models SET num_mission =? WHERE model_id = ? ''',(num_misn[0]-1,num_id))
    db.commit()

def mission_init():
    cursor.executemany('''UPDATE tensorrt_models SET num_mission = ?''',([(0,),(0,),(0,)]))
    db.commit()



@hao.route('/', methods=['GET'])
def hello_Ibalace():

    #post_data=request.form
    Test_Object='cat'
    #os.popen('wget '+post_data['image_url']+' -O /home/zhaohao/TensorRT-2.9.0/data/ResNet50/'+Test_Object+'.jpeg')
    path = DATA + Test_Object + '.jpeg'
    img=get_testcase(path)
    

    starttime=time.time()
    output=Resnet101.infer(img,1000,2)
    
    print("\nTest Case: " + Test_Object)
    print ("Prediction: " + str(np.argmax(output)))
    print("running time is {} \n".format(time.time()-starttime))
    cursor.execute('BEGIN EXCLUSIVE')
    cursor.execute('''SELECT num_mission FROM tensorrt_models''')
    choice=set_model(cursor.fetchall())
    db.commit()
    print choice

    mission_sta(choice)
    if choice is 0:
        time.sleep(5)
    elif choice is 1:
        time.sleep(2)
    else:
        time.sleep(1)
    mission_fin(choice)
    
    return str(np.argmax(output))


if __name__ == "__main__":
    hao.run(host = '0.0.0.0', port = 5755)
    Resnet101.destroy_all()

