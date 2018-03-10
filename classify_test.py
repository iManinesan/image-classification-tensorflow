import sys
import os

from classifier import Classifier


def main():
    classifier = Classifier()
    match = error = 0
    for root, dirs, files in os.walk("test"):
        for f in files:
            if os.path.basename(f) == '.DS_Store':
                continue

            full_path = root + os.sep + f

            res = classifier.classify(full_path)[0]

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
