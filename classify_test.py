import tensorflow as tf
import sys
import os

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


def main():
    # Unpersists graph from file
    with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/trained_labels.txt")]

    sess = tf.Session()

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    match = error = 0
    for root, dirs, files in os.walk("test"):
        for f in files:
            if os.path.basename(f) == '.DS_Store':
                continue

            full_path = root + os.sep + f
            # print('Checking ' + full_path)
            image_data = tf.gfile.FastGFile(full_path, 'rb').read()

            predictions = sess.run(softmax_tensor, \
                                   {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            res = None
            for node_id in top_k[0:1]:  # first
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                # print('%s (score = %.5f)' % (human_string, score))
                res = (human_string, score)

            expected = root.split(os.sep)[1]

            # import pdb; pdb.set_trace()

            if res[0] == expected:
                print('MATCH act={} file={} score={}'.format(res[0], f, res[1]))
                match += 1
            else:
                print('ERR act={} exp={} file={} score={}'.format(res[0], expected, f, res[1]))
                error += 1

    print('RESULT: SUCCESS=' + str(match / (match + error)))



if __name__ == '__main__':
    main()
