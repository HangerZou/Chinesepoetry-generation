#!/usr/bin/env python
# coding: utf-8


import json

import tensorflow as tf
import types
from data_utils import prepare_batch_predict_data
from model import Seq2SeqModel
from rhyme import RhymeUtil
from vocab import get_vocab, ints_to_sentence

# Data loading parameters
tf.app.flags.DEFINE_boolean('rev_data', True, 'Use reversed training data')
tf.app.flags.DEFINE_boolean('align_data', True, 'Use aligned training data')
tf.app.flags.DEFINE_boolean('prev_data', True, 'Use training data with previous sentences')
tf.app.flags.DEFINE_boolean('align_word2vec', True, 'Use aligned word2vec model')

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 1, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('decode_batch_size', 80, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_decode_step', 500, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_string('model_path', None, 'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('model_dir', None, 'Path to load model checkpoints')
tf.app.flags.DEFINE_string('predict_mode', 'greedy', 'Decode helper to use for predicting')
tf.app.flags.DEFINE_string('decode_input', 'data/newstest2012.bpe.de', 'Decoding input path')
tf.app.flags.DEFINE_string('decode_output', 'data/newstest2012.bpe.de.trans', 'Decoding output path')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')


FLAGS = tf.app.flags.FLAGS


#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in list(d.items()))


def load_config(FLAGS):
    if FLAGS.model_path is not None:
        checkpoint_path = FLAGS.model_path
        print('Model path specified at: {}'.format(checkpoint_path))
    elif FLAGS.model_dir is not None:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir + '/')
        print('Model dir specified, using the latest checkpoint at: {}'.format(checkpoint_path))
    else:
        checkpoint_path = tf.train.latest_checkpoint('model/')
        print('Model path not specified, using the latest checkpoint at: {}'.format(checkpoint_path))

    FLAGS.model_path = checkpoint_path

    # Load config saved with model
    config = json.load(open('%s.json' % FLAGS.model_path, 'r'))
    #config = unicode_to_utf8(config_unicode)

    # Overwrite flags
    config1 = {'rev_data': True,'align_data': True,'prev_data': True,'align_word2vec': True,'beam_width': 1,'decode_batch_size':80,'write_n_best': False,'max_decode_step': 500,'model_path': None, 'model_dir': None,'predict_mode': 'greedy','decode_input':'data/newstest2012.bpe.de','decode_output': 'data/newstest2012.bpe.de.trans' }
    for key, value in list(config1.items()):
        config[key] = value
    
    return config


def load_model(session, model, saver):
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print('Reloading model parameters..')
        model.restore(session, saver, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


class Seq2SeqPredictor:
    def __init__(self):
        # Load model config
        config =dict()
       # config = {'cangtou_data': False, 'rev_data': True, 'align_data': True, 'prev_data': True, 'align_word2vec': True, 'cell_type': 'lstm', 'attention_type': 'bahdanau', 'hidden_units': 128, 'depth': 4, 'embedding_size': 128, 'num_encoder_symbols': 30000, 'num_decoder_symbols': 30000, 'vocab_size': 6000, 'use_residual': True, 'attn_input_feeding': False, 'use_dropout': True, 'dropout_rate': 0.3, 'learning_rate': 0.0002, 'max_gradient_norm': 1.0, 'batch_size': 64, 'max_epochs': 10000, 'max_load_batches': 20, 'max_seq_length': 50, 'display_freq': 100, 'save_freq': 100, 'valid_freq': 1150000, 'optimizer': 'adam', 'model_dir': 'model', 'summary_dir': 'model/summary', 'model_name': 'translate.ckpt', 'shuffle_each_epoch': True, 'sort_by_length': True, 'use_fp16': False, 'bidirectional': True, 'train_mode': 'ground_truth', 'sampling_probability': 0.1, 'start_token': 0, 'end_token': 5999, 'allow_soft_placement': True, 'log_device_placement': False, 'rev_data': True, 'align_data': True, 'prev_data': True, 'align_word2vec': True, 'beam_width': 1, 'decode_batch_size': 80, 'write_n_best': False, 'max_decode_step': 500, 'model_path': None, 'model_dir': None, 'predict_mode': 'greedy', 'decode_input': 'data/newstest2012.bpe.de', 'decode_output': 'data/newstest2012.bpe.de.trans'}
        config = load_config(FLAGS)
        #print("config",config)
        config_proto = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )

        self.sess = tf.Session(config=config_proto)

        # Build the model
        self.model = Seq2SeqModel(config, 'predict')

        # Create saver
        # Using var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list=None)

        # Reload existing checkpoint
        load_model(self.sess, self.model, saver)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def predict(self, keywords):
        sentences = []
        for idx,keyword in enumerate(keywords):
            source, source_len = prepare_batch_predict_data(keyword,
                                                            previous=sentences,
                                                            prev=FLAGS.prev_data,
                                                            rev=FLAGS.rev_data,
                                                            align=FLAGS.align_data)

            # flag_pre = True
            # while flag_pre:
            predicted_batch = self.model.predict(
                self.sess,
                encoder_inputs=source,
                encoder_inputs_length=source_len
            )
            # print("p",predicted_batch)
            predicted_line = predicted_batch[0] # predicted is a batch of one line
            predicted_line_clean = predicted_line[:-1] # remove the end token
            predicted_ints = [x[0] for x in predicted_line_clean] # Flatten from [time_step, 1] to [time_step]
            predicted_sentence = ints_to_sentence(predicted_ints)

            if FLAGS.rev_data:
                predicted_sentence = predicted_sentence[::-1]
            # if idx == 0 or idx ==2:
            #     flag_pre = False
            # if idx == 1:
            #     print("p",predicted_sentence[-1])
            #     lis1 = RhymeUtil.get_possible_rhyme_categories(ch = predicted_sentence[-1])
            #     flag_pre = False
            # if idx == 3:
            #     lis3 = RhymeUtil.get_possible_rhyme_categories(ch = predicted_sentence[-1])
            #     for ch in lis3:
            #         if ch in lis1:
            #             flag_pre = False

            sentences.append(predicted_sentence)
        return sentences


def main(_):
    KEYWORDS = [
        '楚',
        '收拾',
        '思乡',
        '相随'
    ]

    with Seq2SeqPredictor() as predictor:
        lines = predictor.predict(KEYWORDS)
        for line in lines:
            print(line)

if __name__ == '__main__':
    tf.app.run()
