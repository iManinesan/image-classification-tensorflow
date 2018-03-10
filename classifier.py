import os

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


class Classifier:
    def __init__(self):
        self._sess = tf.Session()

        with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        # Loads label file, strips off carriage return
        self._label_lines = [line.rstrip() for line
                             in tf.gfile.GFile("logs/trained_labels.txt")]

        # Feed the image_data as input to the graph and get first prediction
        self._softmax_tensor = self._sess.graph.get_tensor_by_name('final_result:0')

    def classify(self, image_path):
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        predictions = self._sess.run(self._softmax_tensor, \
                                     {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        res = []
        for node_id in top_k:
            human_string = self._label_lines[node_id]
            score = predictions[0][node_id]
            # print('%s (score = %.5f)' % (human_string, score))
            res.append((human_string, score))

        return res
