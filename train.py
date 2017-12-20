import tensorflow as tf
import utils
import config

FLAGS = tf.app.flags.FLAGS
batch_size = 100
tf.app.flags.DEFINE_integer("total_batch_size", 1,
                            "Batch size spread across all sync replicas."
                            "We use a size of 32.")
tf.app.flags.DEFINE_string("train_path", "", "The path to the train tfrecord.")

'''
because of the limitation of my hardware, 
I will train in on single device
'''
def main():
    config = config.Config()
    tf.logging.set_verbosity(".")
    with tf.Graph().as_default():
        total_batch_size = FLAGS.total_batch_size
        inputs_dict = config.get_batch(batch_size)
        learning_rate = config.learning_rate
        tf.summary.scalar("learning rate", learning_rate)
        outputs_dict = config.build(inputs_dict, is_training = True)
        loss = outputs_dict["loss"]
        train = tf.train.AdamOptimizer.minimize(loss)

        
if __name__=="__main__":
    main()
