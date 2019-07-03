# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import os
import zipfile


def un_zip(file_name):
    zip_file = zipfile.ZipFile(file_name, 'r')
    zip_file.extractall('./dataset')
    zip_file.close()


if __name__ == '__main__':
    if not os.path.exists('./dataset'):
        os.mkdir('./dataset')
    un_zip('./celeba.zip')
    print('data unzip finished.')
