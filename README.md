# Tensorflow_Mobilenet_transfer_learning
transfor learning of Mobilenet by tensorflow,include train , test, frozen graph

#five script as follows:
		'''
		run_mobilenet.sh
		run_inception.sh
		run_evaluate.sh
		run_frozen_graph.sh
		run_count_ops.sh
		'''
1,train or test or optimize or quantize or tflite for mobilenet/inception.When setting the checkpoint_path,it could load latest checkpoint and restore values,and train the model by this weight values.If you set output_graph,you will get a pb after retrain finishs.
		run_mobilenet.sh
		run_inception.sh

2,run_evaluate.sh——eval the model and print the final accuracy

3,run_frozen_graph.sh——Genetare the pb file through the ckpt, read this pb file and print operation.
That's important,you must set the correct input_names and output_names.If you do'nt know  the name of input and output layer,you coud run the follow function

	def print_tensor_name(chkpt_fname):
			reader = pywrap_tensorflow.NewCheckpointReader(chkpt_fname)
			var_to_shape_map = reader.get_variable_to_shape_map()
			print("tensor_name")
			for key in var_to_shape_map:
					print("tensor_name: ", key)
					print(reader.get_tensor(key)) # Remove this is you want to print only variable names

	python frozen_graph.py \
		--input_checkpoint=tf_files/inception/ \
		--output_graph=frozen_graph.pb \
		--input_names=DecodeJpeg \
		--output_names=final_result
	
	
4,run_count_ops.sh——print all the oprations of the model and summary,you could use tensboard command
