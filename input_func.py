"""Input functions."""

import cPickle
import functools
import json
import os

import cv2
import numpy as np
import tensorflow as tf
from object_detection.builders import image_resizer_builder
from object_detection.builders import model_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import preprocessor_cache
from object_detection.core import standard_fields as fields
from object_detection.core.preprocessor import _flip_boxes_left_right
from object_detection.core.preprocessor import \
  _get_or_create_preprocess_rand_vars
from object_detection.utils import config_util
from object_detection.utils import ops as util_ops
from object_detection.utils import shape_utils
from tensorflow.contrib.estimator.python.estimator.replicate_model_fn import \
  _get_local_devices

tf.flags.DEFINE_integer('ref_sec', 2, 'The length to pad reference video.')

FLAGS = tf.flags.FLAGS

HASH_KEY = 'hash'
HASH_BINS = 1 << 31


def train_generator(subset):
  """Randomly generates a training pair.

  Args:
    subset: 'train' or 'val'.
  Returns:
    qid: The video name of the query video.
    qframe: The chosen clips in query video.
    qlabel: The combined label of the query video.
    qbox: The labeled bounding boxes in the query clips.
    rid: The reference video name.
    rframe: [num_seconds] The chosen clips in reference video.
    same: [num_seconds, max_num_boxes] Whether a reference box is having the
      same label with the query.
    rbox: [num_seconds, max_num_boxes, 4] reference boxes.
    nb: [num_seconds] number of boxes in each clip.
  """
  with open('data/%s.pkl' % subset, 'r') as f:
    boxes, cat = cPickle.load(f)
  index = {}
  for k, v in cat.items():
    for vid, sub, frame in v:
      for f in frame:
        index.setdefault((vid, f), []).append((sub, k))
  for k, v in cat.items():
    label = ','.join([str(i) for i in k])
    for vid, sub, frame in v:
      qid = vid
      # Limits the length of query video.
      if len(frame) > 10:
        st = np.random.choice(len(frame) - 10)
        frame = frame[st:st + 10]
      qframe = ['%04d' % i for i in frame]
      qlabel = label
      qbox = [boxes[(vid, i, sub)] for i in qframe]

      ref = v[np.random.choice(len(v))]
      rid = ref[0]
      # Randomly clip the reference video if it is longer than ref_sec.
      if len(ref[2]) <= FLAGS.ref_sec:
        rframe = ref[2]
      else:
        st = np.random.choice(len(ref[2]) - FLAGS.ref_sec)
        rframe = ref[2][st:st + FLAGS.ref_sec]
      rbox = [[] for _ in rframe]
      same = [[] for _ in rframe]
      for i, f in enumerate(rframe):
        for s, l in index[(rid, f)]:
          rbox[i].append(boxes[(rid, '%04d' % f, s)])
          same[i].append(l == k)
      nb = [len(i) for i in same]
      # makes all the rows in rbox and same have the same length.
      pad_list(rbox, max(nb))
      pad_list(same, max(nb))
      rframe = ['%04d' % i for i in rframe]
      yield qid, qframe, qlabel, qbox, rid, rframe, same, rbox, nb


def pad_list(data, l):
  for i in data:
    if len(i) < l:
      i += [i[0]] * (l - len(i))
  return data


def create_train_input_fn(train_config, train_input_config,
                          model_config, subset):
  def _input_fn(params=None):
    # For debugging.
    # g = train_generator(subset)
    # next(g)

    is_training = subset == 'train'
    dataset = tf.data.Dataset.from_generator(
      functools.partial(train_generator, subset=subset),
      output_types=(
      tf.string, tf.string, tf.string, tf.float32, tf.string, tf.string,
      tf.int32, tf.float32, tf.int32),
      output_shapes=(
      [], [None], [], [None, 4], [], [None], [None, None], [None, None, 4],
      [None]))
    if is_training:
      dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(65536))
    else:
      dataset = dataset.shuffle(65536)
    dataset = dataset.map(functools.partial(read_video, subset=subset),
                          num_parallel_calls=-1).prefetch(-1)

    def transform_and_pad_input_data_fn(tensor_dict):
      """Combines transform and pad operation."""
      model = model_builder.build(model_config, is_training=True)
      image_resizer_config = config_util.get_image_resizer_config(model_config)
      image_resizer_fn = image_resizer_builder.build(image_resizer_config)
      if is_training:
        data_augmentation_options = [
          preprocessor_builder.build(step)
          for step in train_config.data_augmentation_options
        ]
        data_augmentation_fn = functools.partial(
          augment_input_data,
          data_augmentation_options=data_augmentation_options)
        transform_data_fn = functools.partial(
          transform_input_data, model_preprocess_fn=model.preprocess,
          image_resizer_fn=image_resizer_fn,
          num_classes=config_util.get_number_of_classes(model_config),
          data_augmentation_fn=data_augmentation_fn,
          merge_multiple_boxes=train_config.merge_multiple_label_boxes,
          retain_original_image=train_config.retain_original_images)
      else:
        transform_data_fn = functools.partial(
          transform_input_data, model_preprocess_fn=model.preprocess,
          image_resizer_fn=image_resizer_fn,
          num_classes=config_util.get_number_of_classes(model_config),
          data_augmentation_fn=None,
          retain_original_image=train_config.retain_original_images)

      tensor_dict = transform_data_fn(tensor_dict)
      return (_get_features_dict(tensor_dict), _get_labels_dict(tensor_dict))

    dataset = dataset.map(transform_and_pad_input_data_fn,
                          num_parallel_calls=-1).prefetch(-1)

    devices = _get_local_devices('GPU') or _get_local_devices('CPU')
    batch_size = FLAGS.batch_size
    if is_training:

      def key_func(features, labels):
        id2 = features['query_sec']
        return tf.to_int64(id2)

      def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data, batch_size)

      dataset = dataset.apply(
        tf.contrib.data.group_by_window(
          key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
    else:
      dataset = batching_func(dataset, len(devices))
    dataset = dataset.prefetch(-1)
    return dataset

  return _input_fn


def batching_func(x, batch_size):
  # TODO: check query pad.
  if 'subset' in FLAGS:
    num_frames = None
  else:
    num_frames = FLAGS.ref_sec * 24
  dict0 = {
    'query_shape': tf.TensorShape([3]),
    'ref': tf.TensorShape([num_frames, FLAGS.im_size, FLAGS.im_size, 3]),
    fields.InputDataFields.true_image_shape: tf.TensorShape([3]),
    HASH_KEY: tf.TensorShape([]),
    'query': tf.TensorShape([None, FLAGS.im_size, FLAGS.im_size, 3]),
    'query_box': tf.TensorShape([None, 4]),
    'query_sec': tf.TensorShape([]),
    'ref_sec': tf.TensorShape([]),
  }
  if 'original_image' in x.output_classes[0]:
    dict0.update(
      {'original_image': tf.TensorShape([FLAGS.im_size, FLAGS.im_size, 3])})
  return x.padded_batch(
    batch_size,
    padded_shapes=(dict0, {
      fields.InputDataFields.groundtruth_classes: tf.TensorShape(
        [FLAGS.ref_sec * 3, None, 2]),
      fields.InputDataFields.groundtruth_boxes: tf.TensorShape(
        [FLAGS.ref_sec * 3, None, 4]),
      fields.InputDataFields.num_groundtruth_boxes: tf.TensorShape(
        [FLAGS.ref_sec * 3]),
    }),
    drop_remainder=True)


def decode_video(filename):
  assert os.path.exists(filename), filename + ' missing'
  cap = cv2.VideoCapture(filename)
  video = []
  while True:
    ret, frame = cap.read()
    if ret:
      video.append(frame)
    elif len(video) > 0:
      break

  cap.release()
  assert len(video) > 8, filename + ' %d' % len(video)
  video = np.asarray(video, dtype=np.float32)
  video = (video / 255) * 2 - 1
  return video


def load_clip(args, vid):
  frame_no = args[0]
  name = tf.string_join(
    [FLAGS.data_dir, 'clips/', vid, '_', frame_no, '.mkv'])
  video = tf.py_func(decode_video, [name], tf.float32)
  video.set_shape([24, None, None, 3])
  return video


def read_video(query_id, query_frame, query_label, query_box, ref_id,
               ref_frame, ref_label, ref_box, ref_nb, subset):
  query_video = tf.map_fn(
    functools.partial(load_clip, vid=query_id),
    [query_frame], tf.float32, parallel_iterations=1)
  query_shape = tf.shape(query_video)
  query_shape = tf.unstack(query_shape, axis=0)
  query_shape[-1] = 3
  query_video = tf.reshape(query_video, [-1] + query_shape[2:])

  ref_video = tf.map_fn(
    functools.partial(load_clip, vid=ref_id),
    [ref_frame], tf.float32, parallel_iterations=1)
  ref_shape = tf.shape(ref_video)
  ref_shape = tf.unstack(ref_shape, axis=0)
  ref_shape[-1] = 3
  ref_video = tf.reshape(ref_video, [-1] + ref_shape[2:])

  key = tf.string_join([ref_id, ref_frame[0], query_label])
  features = {'query': query_video, 'ref': ref_video,
              fields.InputDataFields.source_id: key}

  query_sec = tf.shape(query_box)[0]
  query_box = tf.expand_dims(query_box, axis=1)
  query_box = tf.tile(query_box, [1, 3, 1])
  query_box = tf.reshape(query_box, [-1, 4])

  ref_sec = tf.shape(ref_nb)[0]
  ref_box = tf.expand_dims(ref_box, axis=1)
  ref_box = tf.tile(ref_box, [1, 3, 1, 1])
  ref_box = tf.reshape(ref_box, [ref_sec * 3, -1, 4])
  ref_label = tf.expand_dims(ref_label, axis=1)
  ref_label = tf.tile(ref_label, [1, 3, 1])
  ref_label = tf.reshape(ref_label, [ref_sec * 3, -1])
  ref_nb = tf.expand_dims(ref_nb, axis=1)
  ref_nb = tf.tile(ref_nb, [1, 3])
  ref_nb = tf.reshape(ref_nb, [-1])
  labels = {'query_box': query_box,
            'query_sec': query_sec,
            fields.InputDataFields.num_groundtruth_boxes: ref_nb,
            fields.InputDataFields.groundtruth_classes: ref_label,
            fields.InputDataFields.groundtruth_boxes: ref_box,
            'ref_sec': ref_sec}
  features.update(labels)
  return features


def augment_input_data(tensor_dict, data_augmentation_options):
  """Applies data augmentation ops to input tensors."""

  video, boxes = tensor_dict['query'], tensor_dict['query_box']
  # boxes = tf.expand_dims(boxes, axis=0)
  video, boxes = random_horizontal_flip(video, boxes)
  tensor_dict['query'] = video
  tensor_dict['query_box'] = boxes

  video = tensor_dict['ref']
  boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]
  box_shape = tf.shape(boxes)
  boxes = tf.reshape(boxes, [-1, 4])
  video, boxes = random_horizontal_flip(video, boxes)
  boxes = tf.reshape(boxes, box_shape)
  tensor_dict['ref'] = video
  tensor_dict[fields.InputDataFields.groundtruth_boxes] = boxes
  return tensor_dict


def random_horizontal_flip(image,
                           boxes=None,
                           seed=None,
                           preprocess_vars_cache=None):
  """Randomly flips the image and detections horizontally."""

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
    result = []
    # random variable defining whether to do flip or not
    generator_func = functools.partial(tf.random_uniform, [], seed=seed)
    do_a_flip_random = _get_or_create_preprocess_rand_vars(
      generator_func,
      preprocessor_cache.PreprocessorCache.HORIZONTAL_FLIP,
      preprocess_vars_cache)
    do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
    result.append(image)

    # flip boxes
    if boxes is not None:
      boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_left_right(boxes),
                      lambda: boxes)
      result.append(boxes)

    return tuple(result)


def transform_input_data(tensor_dict,
                         model_preprocess_fn,
                         image_resizer_fn,
                         num_classes,
                         data_augmentation_fn=None,
                         merge_multiple_boxes=False,
                         retain_original_image=False):
  # if fields.InputDataFields.groundtruth_boxes in tensor_dict:
  #   tensor_dict = util_ops.filter_groundtruth_with_nan_box_coordinates(
  #     tensor_dict)
  if fields.InputDataFields.image_additional_channels in tensor_dict:
    channels = tensor_dict[fields.InputDataFields.image_additional_channels]
    tensor_dict[fields.InputDataFields.image] = tf.concat(
      [tensor_dict[fields.InputDataFields.image], channels], axis=2)

  # Apply data augmentation ops.
  if data_augmentation_fn is not None:
    tensor_dict = data_augmentation_fn(tensor_dict)

  # Apply model preprocessing ops and resize instance masks.
  query = tensor_dict['query']
  preprocessed_resized_image, true_image_shape = resize_image(
    query, new_height=FLAGS.im_size, new_width=FLAGS.im_size)
  tensor_dict['query'] = preprocessed_resized_image
  tensor_dict['query_shape'] = true_image_shape

  ref = tensor_dict['ref']
  preprocessed_resized_image, true_image_shape = resize_image(
    ref, new_height=FLAGS.im_size, new_width=FLAGS.im_size)
  tensor_dict['ref'] = preprocessed_resized_image
  tensor_dict[fields.InputDataFields.true_image_shape] = true_image_shape

  if retain_original_image:
    tensor_dict[
      fields.InputDataFields.original_image] = tf.image.convert_image_dtype(
      tensor_dict['ref'][0] / 2 + 0.5, tf.uint8)

  # Transform groundtruth classes to one hot encodings.
  zero_indexed_groundtruth_classes = tensor_dict[
    fields.InputDataFields.groundtruth_classes]
  tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.one_hot(
    zero_indexed_groundtruth_classes, num_classes)

  if merge_multiple_boxes:
    merged_boxes, merged_classes, _ = util_ops.merge_boxes_with_multiple_labels(
      tensor_dict[fields.InputDataFields.groundtruth_boxes],
      zero_indexed_groundtruth_classes, num_classes)
    merged_classes = tf.cast(merged_classes, tf.float32)
    tensor_dict[fields.InputDataFields.groundtruth_boxes] = merged_boxes
    tensor_dict[fields.InputDataFields.groundtruth_classes] = merged_classes

  return tensor_dict


def resize_image(image,
                 masks=None,
                 new_height=320,
                 new_width=320,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
  with tf.name_scope(
      'ResizeImage',
      values=[image, new_height, new_width, method, align_corners]):
    new_image = tf.image.resize_images(
      image, tf.stack([new_height, new_width]),
      method=method,
      align_corners=align_corners)
    image_shape = shape_utils.combined_static_and_dynamic_shape(image)
    result = [new_image]
    if masks is not None:
      num_instances = tf.shape(masks)[0]
      new_size = tf.stack([new_height, new_width])

      def resize_masks_branch():
        new_masks = tf.expand_dims(masks, 3)
        new_masks = tf.image.resize_nearest_neighbor(
          new_masks, new_size, align_corners=align_corners)
        new_masks = tf.squeeze(new_masks, axis=3)
        return new_masks

      def reshape_masks_branch():
        # The shape function will be computed for both branches of the
        # condition, regardless of which branch is actually taken. Make sure
        # that we don't trigger an assertion in the shape function when trying
        # to reshape a non empty tensor into an empty one.
        new_masks = tf.reshape(masks, [-1, new_size[0], new_size[1]])
        return new_masks

      masks = tf.cond(num_instances > 0, resize_masks_branch,
                      reshape_masks_branch)
      result.append(masks)

    result.append(tf.stack([new_height, new_width, image_shape[3]]))
    return result


def _get_labels_dict(input_dict):
  """Extracts labels dict from input dict."""
  required_label_keys = [
    fields.InputDataFields.groundtruth_classes,
    fields.InputDataFields.groundtruth_boxes,
    fields.InputDataFields.num_groundtruth_boxes,
  ]
  labels_dict = {}
  for key in required_label_keys:
    labels_dict[key] = input_dict[key]

  optional_label_keys = [
    fields.InputDataFields.groundtruth_keypoints,
    fields.InputDataFields.groundtruth_instance_masks,
    fields.InputDataFields.groundtruth_area,
    fields.InputDataFields.groundtruth_is_crowd,
    fields.InputDataFields.groundtruth_difficult
  ]

  for key in optional_label_keys:
    if key in input_dict:
      labels_dict[key] = input_dict[key]
  if fields.InputDataFields.groundtruth_difficult in labels_dict:
    labels_dict[fields.InputDataFields.groundtruth_difficult] = tf.cast(
      labels_dict[fields.InputDataFields.groundtruth_difficult], tf.int32)
  return labels_dict


def _get_features_dict(input_dict):
  """Extracts features dict from input dict."""
  hash_from_source_id = tf.string_to_hash_bucket_fast(
    input_dict[fields.InputDataFields.source_id], HASH_BINS)
  features = {
    'ref_sec': input_dict['ref_sec'],
    'query': input_dict['query'],
    'query_box': input_dict['query_box'],
    'ref': input_dict['ref'],
    'query_shape': input_dict['query_shape'],
    'query_sec': input_dict['query_sec'],
    fields.InputDataFields.true_image_shape: input_dict[
      fields.InputDataFields.true_image_shape],
    HASH_KEY: tf.cast(hash_from_source_id, tf.int32),
  }
  if fields.InputDataFields.original_image in input_dict:
    features[fields.InputDataFields.original_image] = input_dict[
      fields.InputDataFields.original_image]
  return features


def eval_generator(subset):
  with open('data/%s.json' % subset, 'r') as f:
    data = json.load(f)
  for query, ref in data:
    qf = query['frames']
    qf = ['%04d' % i for i in qf]
    ql = [str(i) for i in query['label']]
    ql = ','.join(ql)
    rf = range(ref['start'], ref['end'])
    rf = ['%04d' % i for i in rf]
    yield query['id'], qf, ql, query['boxes'], ref['id'], rf


def create_eval_input_fn(train_config, train_input_config,
                         model_config, subset):
  def _input_fn(params=None):
    is_training = subset == 'train'
    dataset = tf.data.Dataset.from_generator(
      functools.partial(eval_generator, subset=subset),
      output_types=(
      tf.string, tf.string, tf.string, tf.float32, tf.string, tf.string),
      output_shapes=([], [None], [], [None, 4], [], [None]))
    dataset = dataset.map(functools.partial(eval_read_video, subset=subset),
                          num_parallel_calls=-1).prefetch(-1)

    def transform_and_pad_input_data_fn(tensor_dict):
      """Combines transform and pad operation."""
      model = model_builder.build(model_config, is_training=True)
      image_resizer_config = config_util.get_image_resizer_config(model_config)
      image_resizer_fn = image_resizer_builder.build(image_resizer_config)
      if is_training:
        data_augmentation_options = [
          preprocessor_builder.build(step)
          for step in train_config.data_augmentation_options
        ]
        data_augmentation_fn = functools.partial(
          augment_input_data,
          data_augmentation_options=data_augmentation_options)
        transform_data_fn = functools.partial(
          transform_input_data, model_preprocess_fn=model.preprocess,
          image_resizer_fn=image_resizer_fn,
          num_classes=config_util.get_number_of_classes(model_config),
          data_augmentation_fn=data_augmentation_fn,
          merge_multiple_boxes=train_config.merge_multiple_label_boxes,
          retain_original_image=train_config.retain_original_images)
      else:
        transform_data_fn = functools.partial(
          transform_input_data, model_preprocess_fn=model.preprocess,
          image_resizer_fn=image_resizer_fn,
          num_classes=config_util.get_number_of_classes(model_config),
          data_augmentation_fn=None,
          retain_original_image=train_config.retain_original_images)

      tensor_dict = transform_data_fn(tensor_dict)
      return (_get_features_dict(tensor_dict), _get_labels_dict(tensor_dict))

    devices = _get_local_devices('GPU') or _get_local_devices('CPU')
    batch_size = len(devices)

    dataset = dataset.map(transform_and_pad_input_data_fn,
                          num_parallel_calls=-1).prefetch(-1)

    if is_training:

      def key_func(features, labels):
        id2 = labels['query_sec']
        return tf.to_int64(id2)

      def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data, batch_size)

      dataset = dataset.apply(
        tf.contrib.data.group_by_window(
          key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
    else:
      dataset = batching_func(dataset, batch_size)
    dataset = dataset.prefetch(2)
    return dataset

  return _input_fn


def eval_read_video(query_id, query_frame, query_label, query_box, ref_id,
                    ref_frame, subset):
  query_video = tf.map_fn(
    functools.partial(load_clip, vid=query_id),
    [query_frame], tf.float32, parallel_iterations=1)
  query_shape = tf.shape(query_video)
  query_shape = tf.unstack(query_shape, axis=0)
  query_shape[-1] = 3
  query_video = tf.reshape(query_video, [-1] + query_shape[2:])

  ref_video = tf.map_fn(
    functools.partial(load_clip, vid=ref_id),
    [ref_frame], tf.float32, parallel_iterations=1)
  ref_shape = tf.shape(ref_video)
  ref_shape = tf.unstack(ref_shape, axis=0)
  ref_shape[-1] = 3
  ref_video = tf.reshape(ref_video, [-1] + ref_shape[2:])

  key = tf.string_join([ref_id, ref_frame[0], query_label])
  features = {'query': query_video, 'ref': ref_video,
              fields.InputDataFields.source_id: key}
  ref_label = tf.ones([FLAGS.ref_sec * 3, 1], dtype=tf.int32)
  ref_box = tf.ones([FLAGS.ref_sec * 3, 1, 4])
  num_gt = tf.ones([FLAGS.ref_sec * 3])
  query_sec = tf.shape(query_box)[0]
  query_box = tf.expand_dims(query_box, axis=1)
  query_box = tf.tile(query_box, [1, 3, 1])
  query_box = tf.reshape(query_box, [-1, 4])
  labels = {'query_box': query_box,
            'query_sec': query_sec,
            fields.InputDataFields.num_groundtruth_boxes: num_gt,
            fields.InputDataFields.groundtruth_classes: ref_label,
            fields.InputDataFields.groundtruth_boxes: ref_box,
            'ref_sec': tf.shape(ref_frame)[0]}
  features.update(labels)
  return features
