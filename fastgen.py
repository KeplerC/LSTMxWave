import os
import numpy as np
from scipy.io import wavfile
import tensorflow as tf


def encode(wav_data, checkpoint_path, sample_length=64000):
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        wav_data, sample_length = utils.trim_for_encoding(wav_data, sample_length,Config().ae_hop_length)
        
        config = Config()
        with tf.device("/gpu:0"):
            x = tf.placeholder(tf.float32, shape=[batch_size, sample_length])
            graph = config.build({"wav": x}, is_training=False)
            graph.update({"X": x})
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        encodings = sess.run(graph["encoding"], feed_dict={net["X"]: wav_data})
    return encodings

