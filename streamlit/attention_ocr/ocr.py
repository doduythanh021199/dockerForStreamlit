import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import cv2
from PIL import Image
def preprocessing(image):
    size=150
    h, w = image.shape[:2]
    result = np.zeros(shape=(size, size, 3), dtype=np.uint8)
    if w > h:
        neww = 150
        newh = int(neww / w * h)
        image = cv2.resize(image, (neww, newh))
        result[int((size - newh) / 2):int((size - newh) / 2) + newh, :] = image

    if h > w:
        newh = 150
        neww = int(newh / h * w)
        image = cv2.resize(image, (neww, newh))
        result[:, int((size - neww) / 2):int((size - neww) / 2) + neww] = image

    background = np.zeros(shape=(150, 600, 3), dtype=np.uint8)
    background[0:150, 0:150, :] = result
    return background


def load_images(path):
    images=[]
    result=np.zeros(shape=(32,150,600,3),dtype=np.uint8)

    length=len(os.listdir(path))

    for i,image_name in enumerate(sorted(os.listdir(path))):
        image = Image.open(path + image_name)
        image=np.array(image)
        image=preprocessing(image)
        image=np.reshape(image,newshape=(1,150,600,3))
        result[i,:,:,:]=image

    return result

def getDict():
    dict={}
    with open('vnese_charset.txt',encoding='utf8') as dict_file:
        for line in dict_file:

            if len(line.strip().split('\t'))==2:
                (key,value) = line.strip().split('\t')
                dict[value]=int(key)
            else:
                key=line.strip().split('\t')[0]
                dict[' ']=int(key)
    return dict

def gety_out():
    images=load_images("../object_detection/object/")
    input_tensor = tf.convert_to_tensor(images)

    sess=tf.Session()
    signature_key=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_key="images"
    output_key="predictions"

    export_path="attention_ocr_export2/"
    meta_graph_def=tf.saved_model.loader.load(sess,
                                              [tf.saved_model.tag_constants.SERVING],
                                              export_path)
    signature=meta_graph_def.signature_def
    x_tensor_name = signature[signature_key].inputs[input_key].name
    y_tensor_name = signature[signature_key].outputs[output_key].name

    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)

    y_out=sess.run(y,{x:images})
    # export_path = "attention_ocr_export/"
    # model=tf.compat.v2.saved_model.load(str(export_path))
    # input_tensor = input_tensor[tf.newaxis, ...]
    #
    # model_fn = model.signatures['serving_default']
    # output_dict = model_fn(input_tensor)
    # print(output_dict)
    return y_out

def getPredictString(arr):
    dict=getDict()
    result=""
    for i in arr:
        char=list(dict.keys())[list(dict.values()).index(i)]
        if char != "<nul>": result+=char
    return result

def writeFile():
    y_out=gety_out()
    file=open("ocr_result.txt","w")
    for predict in y_out:
        file.write(getPredictString(predict)+"\n")

if __name__ == "__main__":
    writeFile()
