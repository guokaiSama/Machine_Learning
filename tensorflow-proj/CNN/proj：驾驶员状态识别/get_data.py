#-*-coding:utf-8-*-

import os
import shutil
import zipfile

def unzip(fname):
  zip_ref = zipfile.ZipFile(fname)
  zip_ref.extractall('.')
  zip_ref.close()

def copy(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def main():
  os.chdir('dataset')
  unzip('driver_imgs_list.csv.zip')
  unzip('imgs.zip')
  copy('train', 'train_valid')
  os.mkdir(valid)
  os.chdir('..')
  os.system('python reorg_data.py')
  
if __name__ == '__main__':
  main()
