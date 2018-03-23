# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from pylab import *
import time

# load the graph and return graph
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
   
  ops = graph.get_operations()
  for op in ops:
        print(op.name) 
  writer =tf.summary.FileWriter("log_load_graph",graph)
  writer.close()
  
  return graph

# recognize
def recognize(jpg_path, pb_file_path, classes):
  with tf.Graph().as_default():
      graph = load_graph(pb_file_path)
      
      with tf.Session(graph=graph) as sess:
          # 获取输入张量
          input_x = graph.get_tensor_by_name("import/DecodeJpeg:0")
          # 获取输出张量
          output = graph.get_tensor_by_name("import/final_result:0")
          # 读入待识别图片
          img = Image.open(jpg_path)
          # 该MobileNet模型需要128*128的图片输入
          img = array(img.resize((224, 224)),dtype=float32)
          # 图片预处理
          img = (img-128)*1.0/128
          t1 = time.time()
          final_result = sess.run(output, feed_dict={input_x:img})
          t2 = time.time()

          prediction_labels = np.squeeze(np.argmax(final_result, axis=1))
          
          results = np.squeeze(final_result)

          top_k = results.argsort()[-5:][::-1]
          for i in top_k:
              print(classes[i], results[i])
          print('probability: %s: %.3g, running time: %.3g' % (classes[prediction_labels],results[prediction_labels], t2-t1))
          
          
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__=="__main__":
  
  jpg_path = "tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg"
  pb_file_path="tf_files/retrained_inception_graph.pb"
  classes = load_labels("tf_files/retrained_inception_labels.txt")
  recognize(jpg_path, pb_file_path, classes)
  
  