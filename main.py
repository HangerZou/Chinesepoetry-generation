#! /usr/bin/env python
# -*- coding:utf-8 -*-

from plan import Planner
from predict import Seq2SeqPredictor
import sys
import os
import random
import codecs

import tensorflow as tf
import imp
from utils import DATA_RAW_DIR, DATA_PROCESSED_DIR

topic_type_path = os.path.join(DATA_RAW_DIR, 'topic_type.txt')

tf.app.flags.DEFINE_boolean('cangtou', False, 'Generate Acrostic Poem')

imp.reload(sys)
# sys.setdefaultencoding('utf8')


def get_cangtou_keywords(inputs):
    assert(len(inputs) == 4)
    return [c for c in inputs]


def get_topic_type_keywords(choose):
    topic_list = []
    key_set = []
    with codecs.open(topic_type_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            print("line", line)
            toks = line.strip().split('：')
            topic = toks[0]
            set_words = toks[1]
            # content = content.replace(' ', '')
            set_words = set_words.split('，')
            key_set.append(set_words)
            topic_list.append(topic)
            line = fin.readline()
    c = topic_list.index(choose)
    n = len(key_set[c])
    key = []
    for i in range(1, 3):
        s = key_set[c][random.randint(0, n - 1)]
        key.append(s)
    return key


def main(cangtou=False):
    planner = Planner()
    with Seq2SeqPredictor() as predictor:
        # Run loop
        terminate = False
        while not terminate:
            try:
                # inputs = input('Input Text:\n').decode('utf-8').strip()
                inputs = input('Input Text:\n').strip()
                if not inputs:
                    print('Input cannot be empty!')
                elif inputs.lower() in ['quit', 'exit']:
                    terminate = True
                else:
                    if cangtou:
                        keywords = get_cangtou_keywords(inputs)
                    else:
                        # Generate keywords
                        keywords = planner.plan(inputs)

                    # Generate poem
                    lines = predictor.predict(keywords)

                    # Print keywords and poem
                    print('Keyword:\t\tPoem:')
                    for line_number in range(4):
                        punctuation = '，' if line_number % 2 == 0 else '。'
                        print('{keyword}\t\t{line}{punctuation}'.format(
                            keyword=keywords[line_number],
                            line=lines[line_number],
                            punctuation=punctuation
                        ))
            except EOFError:
                terminate = True
            except KeyboardInterrupt:
                terminate = True
    print('\nTerminated.')


if __name__ == '__main__':
    main(cangtou=tf.app.flags.FLAGS.cangtou)
