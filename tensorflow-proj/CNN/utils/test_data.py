# encoding: utf-8
"""
@author: yxluo
@contact: nihaoseeing@gmail.com
"""

from __future__ import absolute_import

import os
import tensorflow as tf

def read_test(data_dir, batch_size=50):
  def _read_img(name):
    content = tf.read_file(name)
    image = tf.image.decode_image(content, channels=3)
    image.set_shape((None, None, 3))
    image = tf.cast(image, dtype=tf.float32)
    
    return image
    
  def _normalize(img):
    img_resized = tf.image.resize_images(img, (256, 256))
    img_normed = tf.image.per_image_standardization(img_resized)
    
    return img_normed
  
  names = os.listdir(data_dir)
  full_names = [os.path.join(data_dir, name) for name in names]
  
  name_dataset = tf.data.Dataset.from_tensor_slices(names)
  img_name_dataset = tf.data.Dataset.from_tensor_slices(full_names)
  
  image_dataset = img_name_dataset.map(_read_img)
  image_dataset = image_dataset.map(_normalize)
  
  dataset = tf.data.Dataset.zip((name_dataset, image_dataset))
  
  dataset = dataset.repeat(1)
  
  if batch_size is not None:
    dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  
  name, image = iterator.get_next()
      
  return name, image
