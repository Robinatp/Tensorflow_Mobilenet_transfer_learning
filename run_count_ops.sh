python -m scripts.count_ops  tf_files/retrained_graph.pb

tensorboard --logdir log_graph/

python -m scripts.count_ops  tf_files/optimized_graph.pb

tensorboard --logdir log_graph/

python -m scripts.count_ops  tf_files/rounded_graph.pb

tensorboard --logdir log_graph/
