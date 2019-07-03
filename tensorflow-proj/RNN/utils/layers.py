from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

# RNN
def build_rnn(num_units, num_layers, batch_size, keep_prob=1):
  def build_cell(num_units, keep_prob):
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        
    return cell
    
  cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(num_units, keep_prob) for _ in range(num_layers)])
  init_state = cell.zero_state(batch_size, tf.float32)
  
  return cell, init_state
  
def rnn(x, num_units, num_layers, batch_size, init_state=None, keep_prob=1, time_major=True, scope='rnn', reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    cell, zero_state = build_rnn(num_units, num_layers, batch_size, keep_prob)
    if init_state is not None:
      out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major=time_major)
    else:
      out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=zero_state, time_major=time_major)
      
    return out, final_state

# LSTM
def build_lstm(num_units, num_layers, batch_size, keep_prob=1):
  def build_cell(num_units, keep_prob):
    cell = tf.nn.rnn_cell.LSTMCell(num_units)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        
    return cell
    
  cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(num_units, keep_prob) for _ in range(num_layers)])
  init_state = cell.zero_state(batch_size, tf.float32)
  
  return cell, init_state

def lstm(x, num_units, num_layers, batch_size, init_state=None, keep_prob=1, time_major=True, scope='lstm', reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    cell, zero_state = build_lstm(num_units, num_layers, batch_size, keep_prob)
    if init_state is not None:
      out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major=time_major)
    else:
      out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=zero_state, time_major=time_major)
      
    return out, final_state

# GRU
def build_gru(num_units, num_layers, batch_size, keep_prob=1):
  def build_cell(num_units, keep_prob):
    cell = tf.nn.rnn_cell.GRUCell(num_units)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        
    return cell
    
  cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(num_units, keep_prob) for _ in range(num_layers)])
  init_state = cell.zero_state(batch_size, tf.float32)
  
  return cell, init_state
  
def gru(x, num_units, num_layers, batch_size, init_state=None, keep_prob=1, time_major=True, scope='gru', reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    cell, zero_state = build_gru(num_units, num_layers, batch_size, keep_prob)
    if init_state is not None:
      out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major=time_major)
    else:
      out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=zero_state, time_major=time_major)
      
    return out, final_state
