# encoding: utf-8
"""
@author: yxluo
@contact: nihaoseeing@gmail.com
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

def predict_result(test_imgs, test_names, model_path, model_fn):
  img_id = list()
  prob_result = list()
  
  test_out = model_fn(test_imgs, 10, is_training=False)
    
  test_prob = tf.nn.softmax(test_out)
  test_pred = tf.argmax(test_prob, axis=-1)
  test_onehot = tf.one_hot(test_pred, 10)
    
  saver = tf.train.Saver()
  
  sess = tf.Session()
  saver.restore(sess, model_path)
  
  try:
    ind = 0
    while True:
      name, pred = sess.run([test_names, test_onehot])
      img_id.append(name)
      prob_result.append(pred)
      ind += pred.shape[0]
      if ind % 1000 == 0:
        print('%d done!' % ind)
      
  except tf.errors.OutOfRangeError:
    pass
    
  prob_result = np.concatenate(prob_result, axis=0)
  img_id = np.concatenate(img_id, axis=0)[:, None]
  all_data = np.concatenate((img_id, prob_result), axis=1)
  submission = pd.DataFrame(all_data)
  
  return submission
