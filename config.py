
import tensorflow as tf
import utils.py

class Config():
    def __init__(self, bidirectional=True, lsmt_units = 512):
        self.ae_hop_length = 512
        self.num_iter = 10000
        self.learning_rate = 0.00001
        self.batch_size = 100
        self.lstm_units = lstm_units
    def get_batch():
        batch_size = self.batch_size
        #TODO:get batch
        pass
    def build():
        x = inputs['wav']
        x_quantized = utils.mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)
        num_z = 16

        embedding = tf.placeholder(name="embedding", shape=[batch_size, num_z], dtype=tf.float32)
        en = tf.expand_dims(encoding, 1)
        
        #TODO: build an autoencoder
        global_step = tf.Variable(0, name="global step", trainable = False)    

        initializer = tf.contrib.layers.xavier_initializer()
        self.lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_units,
                                               initializer=initializer)
        self.lstm_bw = tf.nn.rnn_cell.LSTMCell(lstm_units,
                                               initializer=initializer)
        shape = (lstm_units, num_z)
        self.projection_w = tf.get_variable('projection_w', shape,
                                            initializer=initializer)
        self.projection_b = tf.get_variable('projection_b',shape =(self.vocab_size,), initializer=tf.zeros_initializer())
        #embedded = tf.nn.embedding_lookup(self.embeddings, self.sentence)
        #embedded = tf.nn.dropout(embedded, self.dropout_keep)
        #ret = tf.nn.dynamic_rnn(self.lstm_fw, embedded, dtype=tf.float32,sequence_length=self.sentence_zie)
                                               
    return {
        'predictions': probs,
        'loss': loss,
        'eval': {
            'nll': loss
        },
        'quantized_input': x_quantized,
        'encoding': encoding,
    }
