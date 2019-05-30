"""Evaluates the model.

python main.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import functools
import os

import tensorflow as tf
from object_detection import eval_util
from object_detection import inputs
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.model_lib import _prepare_groundtruth_for_eval
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils.config_util import get_configs_from_pipeline_file

from input_func import create_eval_input_fn
from misc_fn import get_variables_available_in_checkpoint
from misc_fn import unstack_batch

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer('intra_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_integer('inter_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_string('job_dir', 'saving', 'job dir')

tf.flags.DEFINE_string('data_dir', '/home/yfeng23/dataset/ava/', 'data dir')

tf.flags.DEFINE_integer('im_size', 320, 'image size')

tf.flags.DEFINE_bool('load_pretrained', True, 'load pretrained')

tf.flags.DEFINE_bool('multi_gpu', False, 'multi gpu')

tf.flags.DEFINE_integer('max_steps', 1000000, 'training steps')

tf.flags.DEFINE_integer('save_summary_steps', 100, 'save summary steps')

tf.flags.DEFINE_integer('save_checkpoint_steps', 5000, 'save ckpt')

tf.flags.DEFINE_string(
  'i3d_ckpt',
  '/home/yfeng23/lab/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
  'i3d ckpt')

tf.flags.DEFINE_string('subset', 'val', 'subset')

FLAGS = tf.flags.FLAGS


def create_model_fn(detection_model_fn, configs, hparams, use_tpu=False):
  """Creates a model function for `Estimator`.

  Args:
    detection_model_fn: Function that returns a `DetectionModel` instance.
    configs: Dictionary of pipeline config objects.
    hparams: `HParams` object.
    use_tpu: Boolean indicating whether model should be constructed for
        use on TPU.

  Returns:
    `model_fn` for `Estimator`.
  """
  train_config = configs['train_config']
  eval_input_config = configs['eval_input_config']
  eval_config = configs['eval_config']

  def model_fn(features, labels, mode, params=None):
    """Constructs the object detection model.

    Args:
      features: Dictionary of feature tensors, returned from `input_fn`.
      labels: Dictionary of groundtruth tensors if mode is TRAIN or EVAL,
        otherwise None.
      mode: Mode key from tf.estimator.ModeKeys.
      params: Parameter dictionary passed from the estimator.

    Returns:
      An `EstimatorSpec` that encapsulates the model and its serving
        configurations.
    """
    params = params or {}
    total_loss, train_op, detections, export_outputs = None, None, None, None
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Make sure to set the Keras learning phase. True during training,
    # False for inference.
    tf.keras.backend.set_learning_phase(is_training)
    detection_model = detection_model_fn(is_training=is_training,
                                         add_summaries=(not use_tpu))
    scaffold = None
    mask = tf.sequence_mask(features['query_sec'] * 24,
                            tf.shape(features['query'])[1])
    features['query'] = tf.boolean_mask(features['query'], mask)
    mask = tf.sequence_mask(features['query_sec'] * 3,
                            tf.shape(features['query_box'])[1])
    features['query_box'] = tf.boolean_mask(features['query_box'], mask)
    true_im_shape = features[fields.InputDataFields.true_image_shape]
    true_im_shape = tf.expand_dims(true_im_shape, axis=1)
    true_im_shape = tf.tile(true_im_shape,
                            [1, tf.shape(features['ref'])[1] // 8, 1])
    features[fields.InputDataFields.true_image_shape] = tf.reshape(
      true_im_shape, [-1, 3])
    # features['ref'] = tf.reshape(features['ref'], [-1, 8, 320, 320, 3])
    features['query_idx'] = tf.zeros([tf.shape(features['query_box'])[0]],
                                     tf.int32)
    features['query_box'] = [features['query_box']]

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      labels = unstack_batch(
        labels,
        unpad_groundtruth_tensors=train_config.unpad_groundtruth_tensors)
      gt_boxes_list = labels[fields.InputDataFields.groundtruth_boxes]
      gt_classes_list = labels[fields.InputDataFields.groundtruth_classes]
      gt_masks_list = None
      if fields.InputDataFields.groundtruth_instance_masks in labels:
        gt_masks_list = labels[
          fields.InputDataFields.groundtruth_instance_masks]
      gt_keypoints_list = None
      if fields.InputDataFields.groundtruth_keypoints in labels:
        gt_keypoints_list = labels[fields.InputDataFields.groundtruth_keypoints]
      gt_weights_list = None
      if fields.InputDataFields.groundtruth_weights in labels:
        gt_weights_list = labels[fields.InputDataFields.groundtruth_weights]
      if fields.InputDataFields.groundtruth_is_crowd in labels:
        gt_is_crowd_list = labels[fields.InputDataFields.groundtruth_is_crowd]
      detection_model.provide_groundtruth(
        groundtruth_boxes_list=gt_boxes_list,
        groundtruth_classes_list=gt_classes_list,
        groundtruth_masks_list=gt_masks_list,
        groundtruth_keypoints_list=gt_keypoints_list,
        groundtruth_weights_list=gt_weights_list)

    prediction_dict = detection_model.predict(features)
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
      detections = detection_model.postprocess(
        prediction_dict, features[fields.InputDataFields.true_image_shape])

    if mode == tf.estimator.ModeKeys.TRAIN:
      if train_config.fine_tune_checkpoint and hparams.load_pretrained:
        if not train_config.fine_tune_checkpoint_type:
          # train_config.from_detection_checkpoint field is deprecated. For
          # backward compatibility, set train_config.fine_tune_checkpoint_type
          # based on train_config.from_detection_checkpoint.
          if train_config.from_detection_checkpoint:
            train_config.fine_tune_checkpoint_type = 'detection'
          else:
            train_config.fine_tune_checkpoint_type = 'classification'
        asg_map = detection_model.restore_map(
          fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
          load_all_detection_checkpoint_vars=(
            train_config.load_all_detection_checkpoint_vars))
        available_var_map = (
          get_variables_available_in_checkpoint(
            asg_map, FLAGS.i3d_ckpt,
            include_global_step=False))
        if use_tpu:
          def tpu_scaffold():
            tf.train.init_from_checkpoint(train_config.fine_tune_checkpoint,
                                          available_var_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          saver = tf.train.Saver(var_list=available_var_map, reshape=True)

          def init_fn(scaffold, session):
            saver.restore(session, FLAGS.i3d_ckpt)

          scaffold = tf.train.Scaffold(init_fn=init_fn)
          # tf.train.init_from_checkpoint(train_config.fine_tune_checkpoint,
          #                              available_var_map)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      losses_dict = detection_model.loss(
        prediction_dict, features[fields.InputDataFields.true_image_shape])
      losses = [loss_tensor for loss_tensor in losses_dict.values()]
      if train_config.add_regularization_loss:
        regularization_losses = tf.get_collection(
          tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
          regularization_loss = tf.add_n(regularization_losses,
                                         name='regularization_loss')
          losses.append(regularization_loss)
          losses_dict['Loss/regularization_loss'] = regularization_loss
      total_loss = tf.add_n(losses, name='total_loss')
      losses_dict['Loss/total_loss'] = total_loss

      if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
          configs['graph_rewriter_config'], is_training=is_training)
        graph_rewriter_fn()

      # TODO(rathodv): Stop creating optimizer summary vars in EVAL mode once we
      # can write learning rate summaries on TPU without host calls.
      global_step = tf.train.get_or_create_global_step()
      training_optimizer, optimizer_summary_vars = optimizer_builder.build(
        train_config.optimizer)

    if mode == tf.estimator.ModeKeys.TRAIN:
      if use_tpu:
        training_optimizer = tf.contrib.tpu.CrossShardOptimizer(
          training_optimizer)
      if FLAGS.multi_gpu:
        training_optimizer = tf.contrib.estimator.TowerOptimizer(
          training_optimizer)

      # Optionally freeze some layers by setting their gradients to be zero.
      trainable_variables = None
      include_variables = (
        train_config.update_trainable_variables
        if train_config.update_trainable_variables else None)
      exclude_variables = (
        train_config.freeze_variables
        if train_config.freeze_variables else None)
      trainable_variables = tf.contrib.framework.filter_variables(
        tf.trainable_variables(),
        include_patterns=include_variables,
        exclude_patterns=exclude_variables)

      clip_gradients_value = None
      if train_config.gradient_clipping_by_norm > 0:
        clip_gradients_value = train_config.gradient_clipping_by_norm

      if not use_tpu:
        for var in optimizer_summary_vars:
          tf.summary.scalar(var.op.name, var)
      summaries = [] if use_tpu else None
      train_op = tf.contrib.layers.optimize_loss(
        loss=total_loss,
        global_step=global_step,
        learning_rate=None,
        clip_gradients=clip_gradients_value,
        optimizer=training_optimizer,
        variables=trainable_variables,
        summaries=summaries,
        name='')  # Preventing scope prefix on all variables.

    if mode == tf.estimator.ModeKeys.PREDICT:
      for k in detections:
        detections[k] = tf.expand_dims(detections[k], axis=0)
      export_outputs = {
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
          tf.estimator.export.PredictOutput(detections)
      }

    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.EVAL:
      scaffold = None
      class_agnostic = (fields.DetectionResultFields.detection_classes
                        not in detections)
      groundtruth = _prepare_groundtruth_for_eval(
        detection_model, class_agnostic)
      use_original_images = fields.InputDataFields.original_image in features
      eval_images = (
        features[fields.InputDataFields.original_image] if use_original_images
        else features[fields.InputDataFields.image])
      eval_dict = eval_util.result_dict_for_single_example(
        eval_images[0:1],
        features[inputs.HASH_KEY][0],
        detections,
        groundtruth,
        class_agnostic=class_agnostic,
        scale_to_absolute=True)

      if class_agnostic:
        category_index = label_map_util.create_class_agnostic_category_index()
      else:
        category_index = label_map_util.create_category_index_from_labelmap(
          eval_input_config.label_map_path)
      img_summary = None
      if not use_tpu and use_original_images:
        detection_and_groundtruth = (
          vis_utils.draw_side_by_side_evaluation_image(
            eval_dict, category_index,
            max_boxes_to_draw=eval_config.max_num_boxes_to_visualize,
            min_score_thresh=eval_config.min_score_threshold,
            use_normalized_coordinates=False))
        img_summary = tf.summary.image('Detections_Left_Groundtruth_Right',
                                       detection_and_groundtruth)

      # Eval metrics on a single example.
      eval_metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
        eval_config,
        category_index.values(),
        eval_dict)
      for loss_key, loss_tensor in iter(losses_dict.items()):
        eval_metric_ops[loss_key] = tf.metrics.mean(loss_tensor)
      for var in optimizer_summary_vars:
        eval_metric_ops[var.op.name] = (var, tf.no_op())
      if img_summary is not None:
        eval_metric_ops['Detections_Left_Groundtruth_Right'] = (
          img_summary, tf.no_op())
      eval_metric_ops = {str(k): v for k, v in eval_metric_ops.items()}

      if eval_config.use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
        keep_checkpoint_every_n_hours = (
          train_config.keep_checkpoint_every_n_hours)
        saver = tf.train.Saver(
          variables_to_restore,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        scaffold = tf.train.Scaffold(saver=saver)

    # EVAL executes on CPU, so use regular non-TPU EstimatorSpec.
    if use_tpu and mode != tf.estimator.ModeKeys.EVAL:
      return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        scaffold_fn=scaffold_fn,
        predictions=detections,
        loss=total_loss,
        train_op=train_op,
        eval_metrics=eval_metric_ops,
        export_outputs=export_outputs)
    else:
      return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=detections,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs,
        scaffold=scaffold)

  return model_fn


def main(_):
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  session_config = tf.ConfigProto(
    allow_soft_placement=True,
    intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
    inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
    gpu_options=tf.GPUOptions(allow_growth=True))

  run_config = tf.estimator.RunConfig(
    session_config=session_config,
    save_checkpoints_steps=FLAGS.save_checkpoint_steps,
    save_summary_steps=FLAGS.save_summary_steps,
    keep_checkpoint_max=100)

  configs = get_configs_from_pipeline_file(
    'data/faster_rcnn_resnet101_pets.config')
  model_config = configs['model']
  eval_config = configs['eval_config']
  eval_input_config = configs['eval_input_config']

  eval_input_fn = create_eval_input_fn(eval_config, eval_input_config,
                                       model_config, FLAGS.subset)

  detection_model_fn = functools.partial(
    model_builder.build, model_config=model_config)

  model_fn = create_model_fn(detection_model_fn, configs, FLAGS, False)

  if FLAGS.multi_gpu:
    model_fn = tf.contrib.estimator.replicate_model_fn(
      model_fn,
      loss_reduction=tf.losses.Reduction.MEAN)

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.job_dir,
    config=run_config,
    params=FLAGS)

  predictions = estimator.predict(input_fn=eval_input_fn)
  ret = list(predictions)
  with open('%s/ret_%s.pkl' % (FLAGS.job_dir, FLAGS.subset), 'w') as f:
    cPickle.dump(ret, f)


if __name__ == '__main__':
  tf.app.run()
