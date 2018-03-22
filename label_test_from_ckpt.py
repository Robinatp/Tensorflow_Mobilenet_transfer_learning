# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
import PIL.Image as Image
import  numpy as np
import PIL.Image as Image
from pylab import *
import time
from tensorflow.python.platform import gfile
# print all op names
def print_tensor_name(chkpt_fname):
    reader = pywrap_tensorflow.NewCheckpointReader(chkpt_fname)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("tensor_name")
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key)) # Remove this is you want to print only variable names

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('mobilenet/model.ckpt.meta')#直接加载持久化的图

    chkpt_fname = tf.train.latest_checkpoint("mobilenet")#获取checkpoint
    print("model_name: " +  chkpt_fname)
    if  chkpt_fname:
        saver.restore(sess,chkpt_fname)
        print_tensor_name(chkpt_fname)
    
    #打印图中的节点名
    print("operation")
    ops = sess.graph.get_operations()
    for op in ops:
        print(op.name)

    writer = tf.summary.FileWriter("logs_mobilenet_from_ckpt", graph=sess.graph)
    graph = tf.get_default_graph()#获取session中的默认图
    #恢复传入值
    xx = graph.get_tensor_by_name('input:0')
    print(xx)
    #计算利用训练好的模型参数计算预测值
    output = graph.get_tensor_by_name('final_result:0')
    print(output)

    # 读入待识别图片
    image_data = Image.open("tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg")
    # 该MobileNet模型需要128*128的图片输入
    image_data = array(image_data.resize((224, 224)),dtype=float32)
    # 图片预处理
    image_data = (image_data-128)*1.0/128
#     print(image_data)
#     img_out = sess.run(bottneck, feed_dict={xx:image_data})
    pre = sess.run(output, feed_dict={xx:np.reshape(image_data, [-1, 224, 224, 3])})
#     print(img_out.shape)
    print(pre)
    
    results = np.squeeze(pre)
    classes =load_labels("tf_files/retrained_labels.txt")
    top_k = results.argsort()[-5:][::-1]
    for i in top_k:
        print(classes[i], results[i])


   
    
    
