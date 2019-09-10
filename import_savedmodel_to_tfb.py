import sys
import argparse
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.summary import summary

def import_to_tensorboard(savedmodel_dir, tag, log_dir):
  """Load a SavedModel and export it to tensorbloard log dir

  Args:
    savedmodel_dir: The location of the savedmodel
    log_dir: tensorboard log dir
  """

  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(
        sess, [tag], savedmodel_dir)

    log_writer = summary.FileWriter(log_dir)
    log_writer.add_graph(sess.graph)
    print("Start the tensorboard by:"
          "tensorboard --logdir={}".format(log_dir))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert savedmodel to tensorboard log')
  parser.add_argument('--savedmodel_dir', type=str, nargs=1,
                      help='SavedModel directory')
  parser.add_argument('--tag', nargs=1, default=[tf.saved_model.tag_constants.SERVING],
                      help='SavedModel tag')
  parser.add_argument('--log_dir', nargs=1, type=str, help='tensorbloard log dir')
  args = parser.parse_args()
  import_to_tensorboard(args.savedmodel_dir[0], args.tag[0], args.log_dir[0])

