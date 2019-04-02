"""Generates the train, val, and test split for STVR.

The generated pkl files contain a tuple (boxes, categories). Details of them are
defined in load_subset(). The generated json files contain the following data:
[({frames: [query_frame_nos],
   boxes: [[query_bounding_box]],
   id: query_video_id,
   label: query_combined_label},
  {start: reference_start_frame_no,
   end: reference_end_frame_no,
   id: reference_video_id,
   gt: [[[reference_bounding_box]]]})].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pkl
import csv
import json

import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS
np.random.seed(0)


def load_subset(subset):
  """Loads one csv file.

  Args:
    subset: Either train or val.

  Returns:
    boxes: A dictionary: {(video_id,  frame_no, person_id): bounding_box}.
    categories: A dictionary:
      {combined_label: [(video_id, person_id, [frame_nos])]}.
    cat2: A dictionary: {combined_label: {video_id: {frame_no: [person_ids]}}}.
  """
  instances = {}
  boxes = {}
  categories = {}
  cat2 = {}
  with open('data/ava_%s_v2.1.csv' % subset, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      index = (row[0], row[7])
      if index in instances:
        instances[index].setdefault(int(row[1]), []).append(int(row[6]))
      else:
        instances[index] = {}
        instances[index].setdefault(int(row[1]), []).append(int(row[6]))
      index = (row[0], row[1], row[7])
      box = map(float, row[2:6])
      if index in boxes:
        assert boxes[index] == box, index
      else:
        boxes[index] = box

  for k in instances.keys():
    v = instances[k]
    val = []
    label_frame = {}
    for f, l in v.items():
      l.sort()
      label_frame.setdefault(tuple(l), []).append(f)
    for l, f in label_frame.items():
      if l not in cat2:
        cat2[l] = {}
      if k[0] not in cat2[l]:
        cat2[l][k[0]] = {}
      target = cat2[l][k[0]]
      for i in f:
        target.setdefault(i, []).append(k[1])
      f.sort()
      f.append(99999)
      prev = 0
      for i in range(len(f) - 1):
        if f[i] + 1 != f[i + 1]:
          val.append((f[prev:i + 1], l))
          categories.setdefault(l, []).append(k + (f[prev:i + 1],))
          prev = i + 1
    instances[k] = val
  return boxes, categories, cat2


def subset(boxes, cat, cat2, labels):
  labels = [tuple(i) for i in labels]
  cat = {i: cat[i] for i in labels}
  cat2 = {i: cat2[i] for i in labels}
  new_boxes = {}
  for k, v in cat.items():
    for vid, sub, frame in v:
      for f in frame:
        index = (vid, '%04d' % f, sub)
        box = boxes[index]
        box = [box[1], box[0], box[3], box[2]]
        new_boxes[index] = box
  return new_boxes, cat, cat2


def filter_subset(boxes, cat, labels):
  """Removes the val/test categories in the training set."""
  labels = [tuple(i) for i in labels]
  rm = set()
  for k, v in cat.items():
    if k not in labels:
      continue
    for vid, sub, frame in v:
      for f in frame:
        rm.add((vid, f))
  for k in cat.keys():
    if k in labels:
      del cat[k]
  new_boxes = {}
  new_cat = {}
  for k, v in cat.items():
    if k not in new_cat:
      new_cat[k] = []
    for vid, sub, frame in v:
      prev = 0
      for i, f in enumerate(frame):
        if (vid, f) in rm:
          if i > prev:
            new_cat[k].append((vid, sub, frame[prev:i]))
          prev = i + 1
        elif i == len(frame) - 1:
          new_cat[k].append((vid, sub, frame[prev:]))
  cat = new_cat
  for k, v in cat.items():
    for vid, sub, frame in v:
      for f in frame:
        index = (vid, '%04d' % f, sub)
        box = boxes[index]
        # The order axis in AVA dataset and Tensorflow Ojbect Detection API are
        # different.
        box = [box[1], box[0], box[3], box[2]]
        new_boxes[index] = box
  return new_boxes, cat


def write_json(fp, boxes, cat, cat2):
  obj = []
  for k, v in cat.items():
    for vid, sub, frame in v:
      ref_ind = np.random.choice(len(v))
      query = {}
      query['id'] = vid
      query['frames'] = frame
      query['label'] = k
      box_list = []
      for f in frame:
        box_list.append(boxes[(vid, '%04d' % f, sub)])
      query['boxes'] = box_list
      ref = {}
      ref_video = v[ref_ind]
      ref['id'] = ref_video[0]
      ref_len = len(ref_video[2])
      if ref_len < 10:
        # Add some backgrounds before and after the segment.
        ref['start'] = max(ref_video[2][0] - ref_len, 900)
        ref['end'] = min(ref_video[2][-1] + ref_len + 1, 1800)
      else:
        ref['start'] = ref_video[2][0]
        ref['end'] = ref_video[2][-1] + 1
      gt = [[] for _ in range(ref['end'] - ref['start'])]
      for i, l in enumerate(gt):
        f = '%04d' % (ref['start'] + i)
        if ref['start'] + i not in cat2[k][ref['id']]:
          continue
        subjects = cat2[k][ref['id']][ref['start'] + i]
        for s in subjects:
          l.append(boxes[(ref['id'], f, s)])
      ref['gt'] = gt
      obj.append((query, ref))
  json.dump(obj, fp)


def main(_):
  with open('data/val_test.json', 'r') as f:
    test, val = json.load(f)
  train_boxes, train_categories, train_cat2 = load_subset('train')
  train_subset = filter_subset(train_boxes, train_categories, val + test)
  with open('data/train.pkl', 'w') as f:
    pkl.dump(train_subset, f)

  val_boxes, val_categories, val_cat2 = load_subset('val')
  val_subset = subset(val_boxes, val_categories, val_cat2, val)
  with open('data/val.json', 'w') as f:
    write_json(f, *val_subset)
  test_subset = subset(val_boxes, val_categories, val_cat2, test)
  with open('data/test.json', 'w') as f:
    write_json(f, *test_subset)

  val_subset = filter_subset(val_boxes, val_categories, val + test)
  with open('data/val.pkl', 'w') as f:
    pkl.dump(val_subset, f)


if __name__ == '__main__':
  app.run(main)
