import tensorflow as tf

def save(save_path, sess):
    saver = tf.train.Saver()
    saver.save(sess, save_path)

def load(load_path, sess):
    saver = tf.train.Saver()
    print('Loading ' + load_path)
    saver.restore(sess, load_path)