import sys
import os
import json

import tempfile
import tensorflow as tf
from collections import Counter

import requests

from classifier import Classifier

CLASS_TO_APT_STATE = {
    None: 1,  # TODO: I assume that no pics mean that it's "net otdelki",
              # I'm not sure how often this assumption is right but I don't have time
              # to check it right now.
    'netotd': 1,
    'bab': 2,
    'remont': 3
}


def main(apartments_path, output_path):
    classifier = Classifier()
    match = error = 0
    with open(apartments_path) as input:
        with open(output_path, 'w') as output:
            for line in input:
                apartment_info = json.loads(line)
                state = _get_apt_state(classifier, apartment_info)
                apartment_info['apt_state'] = state

                json.dump(apartment_info, output)
                output.write('\n')



def _get_apt_state(classifier, apartment_info):
    print('Processing apt id={}'.format(apartment_info['id']))

    classes = []
    for image_url in apartment_info['image_urls']:
        # clazz = classifier.classify()

        r = requests.get(image_url)
        with tempfile.NamedTemporaryFile(mode="wb", delete=True) as img:
            img.write(r.content)
            img.seek(0, 0)

            res = classifier.classify(img.name)[0]
            classes.append(res[0])

    classes = [c for c in classes if c not in ['outside', 'plans']]

    if not classes:
        clazz = None
    else:
        clazz, _ = Counter(classes).most_common(1)[0]

    print('class={}'.format(clazz))
    return CLASS_TO_APT_STATE[None]


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
