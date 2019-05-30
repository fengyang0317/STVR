import logging

import numpy as np
import tensorflow as tf
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils


def validate_batch_size_for_multi_gpu(batch_size):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by replicate_model_fn
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: the number of examples processed in each training batch.

  Raises:
    ValueError: if no GPUs are found, or selected batch_size is invalid.
  """
  from tensorflow.python.client import \
    device_lib  # pylint: disable=g-import-not-at-top

  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. '
           'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
           ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)


def get_distribution_strategy(num_gpus, all_reduce_alg=None):
  """Return a DistributionStrategy for running the model.
  Args:
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossTowerOps for available algorithms.
      If None, DistributionStrategy will choose based on device topology.
  Returns:
    tf.contrib.distribute.DistibutionStrategy object.
  """
  if num_gpus == 0:
    return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    if all_reduce_alg:
      return tf.contrib.distribute.MirroredStrategy(
        num_gpus=num_gpus,
        cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
          all_reduce_alg, num_packs=num_gpus))
    else:
      return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def get_eval_metric(iou):
  th = np.arange(0.1, 1.0, 0.1)
  vals = [tf.to_float(iou > i) for i in th]
  metrics = dict([('IoU/%g' % i, tf.metrics.mean(j)) for i, j in zip(th, vals)])
  metrics['IoU/mean'] = tf.metrics.mean(vals[4:])
  return metrics


def get_iou(pred, label):
  pred_l, pred_r = tf.unstack(pred, axis=1)
  for i in range(2, len(pred.shape)):
    label = tf.expand_dims(label, axis=i)
  label_l, label_r = tf.unstack(label, axis=1)
  inter_l = tf.maximum(pred_l, label_l)
  inter_r = tf.minimum(pred_r, label_r)
  inter = tf.maximum(inter_r - inter_l, 0)
  union = pred_r - pred_l + label_r - label_l - inter
  return tf.divide(inter, union, name='iou')


def unstack_batch(tensor_dict, unpad_groundtruth_tensors=True):
  """Unstacks all tensors in `tensor_dict` along 0th dimension.

  Unstacks tensor from the tensor dict along 0th dimension and returns a
  tensor_dict containing values that are lists of unstacked, unpadded tensors.

  Tensors in the `tensor_dict` are expected to be of one of the three shapes:
  1. [batch_size]
  2. [batch_size, height, width, channels]
  3. [batch_size, num_boxes, d1, d2, ... dn]

  When unpad_groundtruth_tensors is set to true, unstacked tensors of form 3
  above are sliced along the `num_boxes` dimension using the value in tensor
  field.InputDataFields.num_groundtruth_boxes.

  Note that this function has a static list of input data fields and has to be
  kept in sync with the InputDataFields defined in core/standard_fields.py

  Args:
    tensor_dict: A dictionary of batched groundtruth tensors.
    unpad_groundtruth_tensors: Whether to remove padding along `num_boxes`
      dimension of the groundtruth tensors.

  Returns:
    A dictionary where the keys are from fields.InputDataFields and values are
    a list of unstacked (optionally unpadded) tensors.

  Raises:
    ValueError: If unpad_tensors is True and `tensor_dict` does not contain
      `num_groundtruth_boxes` tensor.
  """
  unbatched_tensor_dict = {key: tf.unstack(tensor)
                           for key, tensor in tensor_dict.items()}
  if unpad_groundtruth_tensors:
    if (fields.InputDataFields.num_groundtruth_boxes not in
        unbatched_tensor_dict):
      raise ValueError('`num_groundtruth_boxes` not found in tensor_dict. '
                       'Keys available: {}'.format(
        unbatched_tensor_dict.keys()))
    unbatched_unpadded_tensor_dict = {}
    unpad_keys = set([
      # List of input data fields that are padded along the num_boxes
      # dimension. This list has to be kept in sync with InputDataFields in
      # standard_fields.py.
      fields.InputDataFields.groundtruth_boxes,
      fields.InputDataFields.groundtruth_classes,
    ]).intersection(set(unbatched_tensor_dict.keys()))

    for key in unpad_keys:
      unpadded_tensor_list = []
      for num_gt, padded_tensor in zip(
          unbatched_tensor_dict[fields.InputDataFields.num_groundtruth_boxes],
          unbatched_tensor_dict[key]):
        tensor_shape = shape_utils.combined_static_and_dynamic_shape(
          padded_tensor)
        slice_begin = tf.zeros([len(tensor_shape)], dtype=tf.int32)
        slice_size = tf.stack(
          [num_gt] + [-1 if dim is None else dim for dim in tensor_shape[1:]])
        unpadded_tensor = tf.slice(padded_tensor, slice_begin, slice_size)
        unpadded_tensor_list.append(unpadded_tensor)
      unbatched_unpadded_tensor_dict[key] = unpadded_tensor_list

    unbatched_tensor_dict.update(unbatched_unpadded_tensor_dict)

  return unbatched_tensor_dict


def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
  """Returns the subset of variables available in the checkpoint.

  Inspects given checkpoint and returns the subset of variables that are
  available in it.

  TODO(rathodv): force input and output to be a dictionary.

  Args:
    variables: a list or dictionary of variables to find in checkpoint.
    checkpoint_path: path to the checkpoint to restore variables from.
    include_global_step: whether to include `global_step` variable, if it
      exists. Default True.

  Returns:
    A list or dictionary of variables.
  Raises:
    ValueError: if `variables` is not a list or dict.
  """
  if isinstance(variables, list):
    variable_names_map = {variable.op.name: variable for variable in variables}
  elif isinstance(variables, dict):
    variable_names_map = variables
  else:
    raise ValueError('`variables` is expected to be a list or dict.')
  ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
  ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  if not include_global_step:
    ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
  vars_in_ckpt = {}
  for variable_name, variable in sorted(variable_names_map.items()):
    if variable_name.startswith('FirstStageFeatureExtractor'):
      variable_name = variable_name[27:]
    rep = variable_name.find('replica')
    if rep != -1:
      variable_name = variable_name[:rep - 1]
    if variable_name in ckpt_vars_to_shape_map:
      if ckpt_vars_to_shape_map[
        variable_name] == variable.shape.as_list() or np.prod(
        ckpt_vars_to_shape_map[variable_name]) == np.prod(
        variable.shape.as_list()):
        vars_in_ckpt[variable_name] = variable
      else:
        logging.warning('Variable [%s] is available in checkpoint, but has an '
                        'incompatible shape with model variable. Checkpoint '
                        'shape: [%s], model variable shape: [%s]. This '
                        'variable will not be initialized from the checkpoint.',
                        variable_name, ckpt_vars_to_shape_map[variable_name],
                        variable.shape.as_list())
    else:
      logging.warning('Variable [%s] is not available in checkpoint',
                      variable_name)
  if isinstance(variables, list):
    return vars_in_ckpt.values()
  return vars_in_ckpt
