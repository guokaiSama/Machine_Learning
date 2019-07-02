# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tensorflow as tf

def read_img(name):
  img_file = tf.read_file(name)
  img_decoded = tf.image.decode_image(img_file, channels=3)
  img_decoded.set_shape([None, None, 3])
  img_float = tf.cast(img_decoded, tf.float32)
  
  return img_float
    
imagenet_mean = tf.constant([0.485, 0.456, 0.406])
imagenet_var = tf.constant([0.229, 0.224, 0.225])
    
def train_preprocess_img(img):
  img_resized = tf.image.resize_images(img, [256, 256])
  img_cropped = tf.random_crop(img_resized, [224, 224, 3])
  img_flipped = tf.image.random_flip_left_right(img_cropped)
  img_normed = tf.image.per_image_standardization(img_flipped)
  
  return img_normed
    
def eval_preprocess_img(img):
  img_resized = tf.image.resize_images(img, [256, 256])
  img_cropped = tf.image.central_crop(img_resized, 224 / 256)
  img_normed = tf.image.per_image_standardization(img_cropped)
  
  return img_normed

def read(root_dir, 
         category_label_dict=None, 
         train=True, 
         epoch=None, 
         shuffle=False, 
         batch_size=None):
  if category_label_dict is None:
    categories = os.listdir(root_dir)
    category_label_dict = {}
    
    for i, category in enumerate(categories):
      category_label_dict[category] = i
    
  img_names = []
  img_labels = []
  
  for category in category_label_dict.keys():
    curr_dir = os.path.join(root_dir, category)
    if not os.path.exists(curr_dir):
      continue
    for img_name in os.listdir(curr_dir):
      if img_name.endswith('.gif'):
        continue
      img_names.append(os.path.join(curr_dir, img_name))
      img_labels.append(category_label_dict[category])
      
  num_examples = len(img_names)
    
  name_dataset = tf.data.Dataset.from_tensor_slices(img_names)
  label_dataset = tf.data.Dataset.from_tensor_slices(img_labels)
  
  image_dataset = name_dataset.map(lambda name: read_img(name))
  
  if train:
    image_dataset = image_dataset.map(lambda img: train_preprocess_img(img))
  else:
    image_dataset = image_dataset.map(lambda img: eval_preprocess_img(img))
  
  dataset = tf.data.Dataset.zip((name_dataset, image_dataset, label_dataset))
  
  if epoch is not None:
    dataset = dataset.repeat(epoch)
  else:
    dataset = dataset.repeat()
    
  if shuffle:
    dataset = dataset.shuffle(100)
  
  if batch_size is not None:
    dataset = dataset.batch(batch_size)
  
  iterator = dataset.make_one_shot_iterator()
  names, images, labels = iterator.get_next()
    
  return category_label_dict, names, images, labels, num_examples
