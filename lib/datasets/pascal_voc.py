# coding=UTF-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
import PIL


class pascal_voc(imdb):
  def __init__(self, image_set, year, use_diff=False):
    name = 'voc_' + year + '_' + image_set
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)
    self._year = year
    self._image_set = image_set
    self._devkit_path = self._get_default_path()
    self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
    # self._classes = ('__background__',  # always index 0
    #                 'chair', 'table', 'sofa', 'bed', 'shelf', 'cabinet')liu 修改
    self._classes = ('__background__',  # always index 0
                    'ape', 'can', 'cat', 'driller', 'duck', 'eggbox','glue','holepuncher')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,#liu change use_diff 
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    # print('*******************************')
    # print(gt_roidb)
    # fang[-1]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def get_pic_size(self, filename):
    w = PIL.Image.open(filename).size[0]
    h = PIL.Image.open(filename).size[1]
    return w, h

  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    picname = os.path.join(self._data_path, 'JPEGImages', index + '.jpg') # pci with  height
    w , h = self.get_pic_size(picname)
    # print(filename)
    tree = ET.parse(filename)
    objs = tree.findall('object')
    # if not self.config['use_diff']:
    #   # Exclude the samples labeled as difficult
    #   non_diff_objs = [
    #     obj for obj in objs if int(obj.find('difficult').text) == 0]
    #   # if len(non_diff_objs) != len(objs):
    #   #     print 'Removed {} difficult objects'.format(
    #   #    scipy     len(objs) - len(non_diff_objs))
    #   objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    gt_points2d = np.zeros((num_objs, 18), dtype=np.float32) #8 points + 1 center
    gt_points2d_tmp = np.zeros((num_objs, 16), dtype=np.float32)
    gt_center = np.zeros((num_objs, 2), dtype=np.float32)
    gt_front_4points = np.zeros((num_objs, 8), dtype = np.float32)
    gt_back_4points = np.zeros((num_objs, 8), dtype = np.float32)

    gt_front_2_1_points = np.zeros((num_objs, 4)).astype(int)
    gt_front_2_2_points = np.zeros((num_objs, 4)).astype(int)
    gt_front_center = np.zeros((num_objs, 2)).astype(int)
    gt_back_2_1_points = np.zeros((num_objs, 4)).astype(int)
    gt_back_2_2_points = np.zeros((num_objs, 4)).astype(int)
    gt_back_center = np.zeros((num_objs, 2)).astype(int)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
      bbox = obj.find('bndbox')
      # Make pixel indexes 0-based
      # x1 = float(bbox.find('xmin').text) - 1
      # y1 = float(bbox.find('ymin').text) - 1
      # x2 = float(bbox.find('xmax').text) - 1
      # y2 = float(bbox.find('ymax').text) - 1
      x1 = float(bbox.find('xmin').text)
      y1 = float(bbox.find('ymin').text)
      x2 = float(bbox.find('xmax').text)
      y2 = float(bbox.find('ymax').text)

      if x2 > w:
        x2 = w
      if x1 > w:
        x1 = w
      
      if y2 > h:
        y2 = h
      if y1 > h:
        y1 = h

      if x1 < 1:
        x1 = 1
      if x2 < 1:
        x2 = 1
        
      if y2 < 1:
        y2 = 1
      if y1 < 1:
        y1 = 1
      
      
      
      # if x1 > x2 or y1 > y2:
      #   print(index)
      #   print(x1,x2,y1,y2)
      #   fang[-1]
      points2d = obj.find('points2d')  #fang
      px1 = int(float(points2d.find('x1').text))
      py1 = int(float(points2d.find('y1').text))
      px2 = int(float(points2d.find('x2').text))
      py2 = int(float(points2d.find('y2').text))
      px3 = int(float(points2d.find('x3').text))
      py3 = int(float(points2d.find('y3').text))
      px4 = int(float(points2d.find('x4').text))
      py4 = int(float(points2d.find('y4').text))
      px5 = int(float(points2d.find('x5').text))
      py5 = int(float(points2d.find('y5').text))
      px6 = int(float(points2d.find('x6').text))
      py6 = int(float(points2d.find('y6').text))
      px7 = int(float(points2d.find('x7').text))
      py7 = int(float(points2d.find('y7').text))
      px8 = int(float(points2d.find('x8').text))
      py8 = int(float(points2d.find('y8').text))

      #这里交换顺序来预测上4点，而不是下4点 1,3-->2,4 5,7-->6,8
      # px1,px2,px3,px4,px5,px6,px7,px8=px3,px7,px1,px5,px4,px8,px2,px6
      # py1,py2,py3,py4,py5,py6,py7,py8=py3,py7,py1,py5,py4,py8,py2,py6
      # print("p1:",px1,py1)
      # print("p2:",px2,py2)
      # print("p3:",px3,py3)
      # print("p4:",px4,py4)
      # print("p5:",px5,py5)
      # print("p6:",px6,py6)
      # print("p7:",px7,py7)
      # print("p1:",px8,py8)

      objbox = obj.find('objbox') #fang
      # length = float(objbox.find('length').text)
      # width = float(objbox.find('width').text)
      # height = float(objbox.find('height').text)

      gt_points2d_tmp[ix, :] = [px1, py1, px2, py2, px3, py3, px4, py4, px5, py5, px6, py6, px7, py7, px8, py8] #fang
      # px = np.mean(gt_points2d_tmp[ix, 0::2])
      # py = np.mean(gt_points2d_tmp[ix, 1::2])
      # gt_points2d[ix, :] = [px1, py1, px2, py2, px3, py3, px4, py4, px5, py5, px6, py6, px7, py7, px8, py8, px, py]
      # gt_points2d[ix, 0::2] = gt_points2d[ix, 0::2] - (px - 730.0/2.0)
      # gt_points2d[ix, 1::2] = gt_points2d[ix, 1::2] - (py - 530.0/2.0)
      
      gt_front_4points[ix, :] = [px1, py1, px2, py2, px3, py3, px4, py4]
      gt_back_4points[ix, :] = [px5, py5, px6, py6, px7, py7, px8, py8]
      
      gt_front_center[ix, :] = [np.mean(gt_front_4points[ix, 0::2]), np.mean(gt_front_4points[ix, 1::2])]
      gt_back_center[ix, :] = [np.mean(gt_back_4points[ix, 0::2]), np.mean(gt_back_4points[ix, 1::2])]
      
      # print(gt_front_center[ix, :])
      if px1 == px4:
        px4 += 1
      if py1 == py4:
        py4 += 1
      gt_front_2_1_points[ix, :] = [px1, py1, px4, py4]

      if px2 == px3:
        px2 += 1
      if py2 == py3:
        py2 += 1
      gt_front_2_2_points[ix, :] = [px3, py3, px2, py2]

      if px5 == px8:
        px8 += 1
      if py5 == py8:
        py8 += 1
      gt_back_2_1_points[ix, :] = [px5, py5, px8, py8]

      if px6 == px7:
        px6 += 1
      if py6 == py7:
        py6 += 1
      gt_back_2_2_points[ix, :] = [px7, py7, px6, py6]

      
      # print(gt_front_2points[ix, :])
      #   fang[-1]
      
      # print(gt_front_2points[ix, :])
      #   fang[-1]
      # print(gt_front_2points[ix, :])
      # fang[-1]



      # print(index, px, py)
      # fang[-1]
 
      claname = obj.find('name').text.lower().strip()
      cls = self._class_to_ind[claname]
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

      



    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            # 'gt_points2d': gt_points2d, #fang
            'gt_front_center': gt_front_center, #fang
            'gt_back_center': gt_back_center,
            # 'gt_front_4points': gt_front_4points,
            # 'gt_back_4points': gt_back_4points,
            'gt_front_2_1_points': gt_front_2_1_points,
            'gt_front_2_2_points': gt_front_2_2_points,
            'gt_back_2_1_points': gt_back_2_1_points,
            'gt_back_2_2_points': gt_back_2_2_points,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'VOC' + self._year,
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'Annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'ImageSets',
      'Main',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
