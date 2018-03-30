#inception
python frozen_graph.py \
	--input_checkpoint=tf_files/inception/ \
	--output_graph=frozen_graph.pb \
	--input_names=DecodeJpeg \
	--output_names=final_result
	
#mobilenet
python frozen_graph.py \
	--input_checkpoint=tf_files/mobilenet/ \
	--output_graph=frozen_graph.pb \
	--input_names=input \
	--output_names=final_result