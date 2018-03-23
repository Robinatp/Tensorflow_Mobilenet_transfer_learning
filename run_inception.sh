#! /bin/bash
#git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
#curl http://download.tensorflow.org/example_images/flower_photos.tgz \
#    | tar xz -C tf_files

ARCHITECTURE="inception_v3"
echo "input command:train or test or optimize or quantize or tflite:"
read a
echo "input is $a"

if [ $a = train ] ; then
echo $ARCHITECTURE
python -m scripts.retrain \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models/inception-2015-12-05/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_inception_graph.pb \
  --output_labels=tf_files/retrained_inception_labels.txt \
  --checkpoint_path=tf_files/inception/ \
  --learning_rate=0.001 \
  --how_many_training_steps=50000 

fi

if [ $a = test ] ; then
python -m scripts.label_image \
	--image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg \
    --graph=tf_files/retrained_inception_graph.pb  \
    --labels=tf_files/retrained_inception_labels.txt \
    --input_layer="DecodeJpeg" \
    --output_layer="final_result" 
    
python -m scripts.label_image \
    --image=tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg \
    --graph=tf_files/retrained_inception_graph.pb  \
    --labels=tf_files/retrained_inception_labels.txt \
    --input_layer="DecodeJpeg" \
    --output_layer="final_result" 
 
fi

if [ $a = optimize ] ; then
python -m tensorflow.python.tools.optimize_for_inference \
  --input=tf_files/retrained_inception_graph.pb \
  --output=tf_files/optimized_inception_graph.pb \
  --input_names="Cast" \
  --output_names="final_result"

python -m scripts.label_image \
  --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg \
  --graph=tf_files/optimized_inception_graph.pb  \
  --labels=tf_files/retrained_inception_labels.txt \
  --input_layer="Cast" \
  --output_layer="final_result" 

python -m scripts.graph_pb2tb tf_files/training_summaries/retrained \
  tf_files/retrained_inception_graph.pb 

python -m scripts.graph_pb2tb tf_files/training_summaries/optimized \
  tf_files/optimized_inception_graph.pb 

pkill -f tensorboard
tensorboard --logdir tf_files/training_summaries 
fi


if [ $a = quantize ] ; then
python -m scripts.quantize_graph \
  --input=tf_files/optimized_inception_graph.pb \
  --output=tf_files/rounded_inception_graph.pb \
  --output_node_names=final_result \
  --mode=weights_rounded

python -m scripts.label_image \
  --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg \
  --graph=tf_files/rounded_inception_graph.pb  \
   --labels=tf_files/retrained_inception_labels.txt \
   --input_layer="Cast" \
   --output_layer="final_result" 

python -m scripts.label_image \
  --image=tf_files/flower_photos/daisy/3475870145_685a19116d.jpg \
  --graph=tf_files/rounded_inception_graph.pb  \
  --labels=tf_files/retrained_inception_labels.txt \
   --input_layer="Cast" \
   --output_layer="final_result" 
  
  
python -m scripts.evaluate  tf_files/retrained_inception_graph.pb

python -m scripts.evaluate  tf_files/optimized_inception_graph.pb

python -m scripts.evaluate  tf_files/rounded_inception_graph.pb
fi

if [ $a = tflite ] ; then
IMAGE_SIZE=224
toco \
  --input_file=tf_files/retrained_inception_graph.pb \
  --output_file=tf_files/optimized_inception_graph.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \
  --input_array=input \
  --output_array=final_result \
  --inference_type=FLOAT \
  --input_type=FLOAT
fi
