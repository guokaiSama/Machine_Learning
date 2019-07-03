# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_images(images):  # 定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        img = img.reshape((28, 28, 3))
        #img = img.transpose(1, 2, 0)
        plt.imshow(img.astype(np.uint8))
    return

def preprocess(img):
  # crop face area
  img_shape = tf.shape(img)
  im_h = img_shape[0]
  im_w = img_shape[1]
  face_width = face_height = 108
  i = (im_h - face_height) // 2
  j = (im_w - face_width) // 2
  crop = img[i: i + face_height, j: j + face_width]
  
  # resize
  resize = tf.image.resize_images(crop, (28, 28))
  
  # normalize to [0, 1]
  normalize = resize / 255
  
  # normalize to [-1, 1]
  preprocessd = (normalize - 0.5) / 0.5
  
  return preprocessd
  
def deprocess(img):
  img = (img + 1.0) / 2.0 * 255
  return tf.clip_by_value(img, 0, 255)

def read(imgs_folder, batch_size=1, shuffle=True, epoch=None):
  def _read_name(name):
    img_content = tf.read_file(name)
    img_decoded = tf.image.decode_jpeg(img_content, channels=3)
    img_float = tf.cast(img_decoded, tf.float32)
    
    return img_float
      
  imgs_name = [os.path.join(imgs_folder, name) for name in os.listdir(imgs_folder)]
  
  imgs_name_dataset = tf.data.Dataset.from_tensor_slices(imgs_name)
  
  imgs_dataset = imgs_name_dataset.map(_read_name)
  
  imgs_dataset = imgs_dataset.map(preprocess)
  
  if epoch is not None:
    imgs_dataset = imgs_dataset.repeat(epoch)
  
  if shuffle:
    imgs_dataset = imgs_dataset.shuffle(100)
    
  imgs_dataset = imgs_dataset.batch(batch_size)
  
  iterator = imgs_dataset.make_one_shot_iterator()
  
  imgs = iterator.get_next()
  
  return imgs
