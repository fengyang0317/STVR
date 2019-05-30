import csv
import os
import glob
import multiprocessing

from absl import app
from absl import flags

flags.DEFINE_string('data_dir', '/media/yfeng23/7d807ee6-3c76-4ee8-b24a-5dd529d'
                                'a0144/dataset/ava_dataset_v2/', 'data dir')

flags.DEFINE_string('subset', 'train', 'subset')

flags.DEFINE_integer('num_proc', 2, 'number of process')

FLAGS = flags.FLAGS

dst = None


def initializer():
  current = multiprocessing.current_process()
  id = (current._identity[0] - 1) % 8
  os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % id


def cut_video(i):
  video_path = glob.glob(
    os.path.join(FLAGS.data_dir, FLAGS.subset, i + '.*'))
  assert len(video_path) == 1, i
  video_path = video_path[0]
  ext = os.path.splitext(video_path)[1]
  cmd = ('ffmpeg -ss 900 -i %s -to 901 -vf scale=320:320 -map 0 -segment_time 1'
         ' -force_key_frames "expr:gte(t,n_forced)" -f segment -r 24'
         ' -reset_timestamps 1 -segment_start_number 900 -loglevel error'
         ' %s/%s_%%04d.mkv' % (video_path, dst, i))
  os.system(cmd)


def main(_):
  with open('data/ava_%s_v2.1.csv' % FLAGS.subset, 'r') as f:
    reader = csv.reader(f)
    videos = set()
    for row in reader:
      videos.add(row[0])
  global dst
  dst = os.path.join(FLAGS.data_dir, 'clips')
  if not os.path.exists(dst):
    os.mkdir(dst)
  pool = multiprocessing.Pool(FLAGS.num_proc, initializer)
  pool.map(cut_video, list(videos))


if __name__ == '__main__':
  app.run(main)
