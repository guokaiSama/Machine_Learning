# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim

def subsample(x, factor, scope=None):
  if factor == 1:
    return x
  return slim.max_pool2d(x, [1, 1], factor, scope=scope)
    
def residual_block(x, bottleneck_depth, out_depth, stride=1, scope='residual_block'):
  in_depth = x.get_shape().as_list()[-1]
  with tf.variable_scope(scope):
    # 如果通道数没有改变,用下采样改变输入的大小
    if in_depth == out_depth:
        shortcut = subsample(x, stride, 'shortcut')
    # 如果有变化, 用卷积改变输入的通道以及大小
    else:
        shortcut = slim.conv2d(x, out_depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')

    residual = slim.conv2d(x, bottleneck_depth, [1, 1], stride=1, scope='conv1')
    residual = slim.conv2d(residual, bottleneck_depth, 3, stride, scope='conv2')
    residual = slim.conv2d(residual, out_depth, [1, 1], stride=1, activation_fn=None, scope='conv3')

    # 相加操作
    output = tf.nn.relu(shortcut + residual)

    return output
        
def resnet(inputs, num_classes, scope='resnet', reuse=None, is_training=None, verbose=False):
  with tf.variable_scope(scope, reuse=reuse):
    net = inputs
    
    if verbose:
        print('input: {}'.format(net.shape))
    
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], padding='SAME'):
          with tf.variable_scope('block1'):
            net = slim.conv2d(net, 32, [5, 5], stride=2, scope='conv_5x5')

            if verbose:
              print('block1: {}'.format(net.shape))
              
          with tf.variable_scope('block2'):
            net = slim.max_pool2d(net, [3, 3], 2, scope='max_pool')
            net = residual_block(net, 32, 128, scope='residual_block1')
            net = residual_block(net, 32, 128, scope='residual_block2')

            if verbose:
              print('block2: {}'.format(net.shape))
              
          with tf.variable_scope('block3'):
            net = residual_block(net, 64, 256, stride=2, scope='residual_block1')
            net = residual_block(net, 64, 256, scope='residual_block2')

            if verbose:
              print('block3: {}'.format(net.shape))
              
          with tf.variable_scope('block4'):
            net = residual_block(net, 128, 512, stride=2, scope='residual_block1')
            net = residual_block(net, 128, 512, scope='residual_block2')

            if verbose:
              print('block4: {}'.format(net.shape))
          
          with tf.variable_scope('classification'):
            net = tf.reduce_mean(net, [1, 2], name='global_pool', keep_dims=True)
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='logit')

            if verbose:
              print('classification: {}'.format(net.shape))
              
          return net

def resnet_arg_scope():                
  with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm) as sc:
    return sc
