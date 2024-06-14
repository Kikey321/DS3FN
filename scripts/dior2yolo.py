# coding:utf-8

import os
import random
import argparse

import xml.etree.ElementTree as ET
from os import getcwd
from shutil import copyfile


parser = argparse.ArgumentParser()
#xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
parser.add_argument('--xml_path', default='/root/autodl-tmp/DIOR/Annotations/Horizontal Bounding Boxes', type=str, help='input xml label path')
#数据集的划分，地址选择自己数据下的ImageSets/Main

opt = parser.parse_args()

sets = ['train', 'val', 'test']
classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
              'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
              'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']


if not os.path.exists('/root/autodl-fs/mydata/DIOR'):
    os.makedirs('/root/autodl-fs/mydata/DIOR')

if not os.path.exists('/root/autodl-fs/mydata/DIOR/labels/'):
    os.makedirs('/root/autodl-fs/mydata/DIOR/labels/')
if not os.path.exists('/root/autodl-fs/mydata/DIOR/labels/train'):
    os.makedirs('/root/autodl-fs/mydata/DIOR/labels/train')
if not os.path.exists('/root/autodl-fs/mydata/DIOR/labels/test'):
    os.makedirs('/root/autodl-fs/mydata/DIOR/labels/test')
if not os.path.exists('/root/autodl-fs/mydata/DIOR/labels/val'):
    os.makedirs('/root/autodl-fs/mydata/DIOR/labels/val')

if not os.path.exists('/root/autodl-fs/mydata/DIOR/images/'):
    os.makedirs('/root/autodl-fs/mydata/DIOR/images/')
if not os.path.exists('/root/autodl-fs/mydata/DIOR/images/train'):
    os.makedirs('/root/autodl-fs/mydata/DIOR/images/train')
if not os.path.exists('/root/autodl-fs/mydata/DIOR/images/test'):
    os.makedirs('/root/autodl-fs/mydata/DIOR/images/test')
if not os.path.exists('/root/autodl-fs/mydata/DIOR/images/val'):
    os.makedirs('/root/autodl-fs/mydata/DIOR/images/val')


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(image_id, path):
#输入输出文件夹，根据实际情况进行修改
    in_file = open('/root/autodl-tmp/DIOR/Annotations/Horizontal Bounding Boxes/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open('/root/autodl-fs/mydata/DIOR/labels/' + path + '/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        #difficult = obj.find('difficult').text
        #difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



train_percent = 0.6
test_percent = 0.2
val_percent = 0.2

xmlfilepath = opt.xml_path
# txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
# if not os.path.exists(txtsavepath):
#     os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
list_index = list(list_index)
random.shuffle(list_index)


train_nums = list_index[:int(num * train_percent)]
test_nums = list_index[int(num * train_percent): int(num * test_percent)+int(num * train_percent)]
val_nums = list_index[int(num * test_percent)+int(num * train_percent):]



for i in list_index:
    name = total_xml[i][:-4]
    if i in train_nums:
        convert_annotation(name, 'train')   # lables
        image_origin_path = '/root/autodl-tmp/DIOR/JPEGImages/' + name + '.jpg'
        image_target_path = '/root/autodl-fs/mydata/DIOR/images/train/' + name + '.jpg'
        copyfile(image_origin_path, image_target_path)

    if i in test_nums:
        convert_annotation(name, 'test')   # lables
        image_origin_path = '/root/autodl-tmp/DIOR/JPEGImages/' + name + '.jpg'
        image_target_path = '/root/autodl-fs/mydata/DIOR/images/test/' + name + '.jpg'
        copyfile(image_origin_path, image_target_path)

    if i in val_nums:
        convert_annotation(name, 'val')   # lables
        image_origin_path = '/root/autodl-tmp/DIOR/JPEGImages/' + name + '.jpg'
        image_target_path = '/root/autodl-fs/mydata/DIOR/images/val/' + name + '.jpg'
        copyfile(image_origin_path, image_target_path)


