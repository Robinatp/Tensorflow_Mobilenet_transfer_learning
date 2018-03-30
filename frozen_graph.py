from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import app
import os
import PIL.Image as Image
import  numpy as np


FLAGS = None

def freeze_graph():
    """
    freeze the saved checkpoints/graph to *.pb
    """
    checkpoint = tf.train.get_checkpoint_state(FLAGS.input_checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    output_graph = os.path.join(FLAGS.input_checkpoint, FLAGS.output_graph)
    
    saver = tf.train.import_meta_graph(input_checkpoint + ".meta", 
                                       clear_devices=True)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                     input_graph_def,
                                                                     FLAGS.output_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph" % (len(output_graph_def.node)))
        
        
def load_graph(frozen_graph_filename):
    """
    Loads Frozen graph
    """
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def main(unused_args):

    freeze_graph()
       
    frozen_graph_path = os.path.join(FLAGS.input_checkpoint, FLAGS.output_graph)   
    graph = load_graph(frozen_graph_path)
    
    for op in graph.get_operations():
        print(op.name)
    
    input_operation = graph.get_operation_by_name('import/'+FLAGS.input_names)
    print(input_operation.outputs[0])
    output_operation = graph.get_operation_by_name('import/'+FLAGS.output_names)
    print(output_operation.outputs[0])
    
    return 0

def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
      "--input_checkpoint",
      type=str,
      default="tf_files/inception/",
      help="TensorFlow variables file to load.")
  
  parser.add_argument(
      "--output_graph",
      type=str,
      default="frozen_graph.pb",
      help="Output \'GraphDef\' file name.")
  

  parser.add_argument(
      "--input_names",
      type=str,
      default="DecodeJpeg",
      help="Input node names, comma separated.")
  
  parser.add_argument(
      "--output_names",
      type=str,
      default="final_result",
      help="Output node names, comma separated.")

  return parser.parse_known_args()


if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)





  