from urllib.parse import urlparse

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import tqdm
#import tensorflow_hub as hub
import tensorflow as tf
#import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil

np.set_printoptions(suppress=True)

import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('./transformers')
sys.path.append('./sacremoses')
tf.keras.backend.clear_session()
from transformers import *
print(tf.__version__)

PATH = './'
# BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
# tokenizer = tokenization.FullTokenizer(BERT_PATH+'/assets/vocab.txt', True)
BERT_PATH = './roberta-base/'

tokenizer = RobertaTokenizer(BERT_PATH + "roberta-base-vocab.json", BERT_PATH + "roberta-base-merges.txt")
MAX_SEQUENCE_LENGTH = 512

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
#print('\noutput categories:\n\t', output_categories)
#print('\ninput categories:\n\t', input_categories)
output_categories_question =list(df_train.columns[11:32])
output_categories_answer = list(df_train.columns[32:])
input_categories_question = list(df_train.columns[[1,2]])
input_categories_answer = list(df_train.columns[[1,5]])
print('\noutput categories_question:\n\t', output_categories_question)
print('\noutput categories_answer:\n\t', output_categories_answer)
print('\ninput categories_question:\n\t', input_categories_question)
print('\ninput categories_answer:\n\t', input_categories_answer)


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))



def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [1] * (max_seq_length - len(token_ids))
    return input_ids


def _trim_input(title, question, max_sequence_length,
                t_max_len=100, head=128, tail=281, Q=True, q_max_len=239, a_max_len=239):
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    # a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    # a_len = len(a)

    if (t_len + q_len + 4) > max_sequence_length:
        if t_max_len > t_len:
            t_new_len = t_len
            q_head = head
            q_tail = 508 - q_head - t_new_len
        else:
            t_new_len = t_max_len
            if (t_new_len + q_len + 4) > max_sequence_length:
                q_head = head
                q_tail = 508 - q_head - t_new_len
            else:
                t = t[:t_new_len]

                return t, q
        t = t[:t_new_len]
        q = q[:q_head] + q[-q_tail:]
        return t, q
    else:
        return t, q


def _convert_to_bert_inputs(title, question, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["<s>"] + title + ["</s>"] + ["</s>"] + question + ["</s>"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    #input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks]


def compute_input_arays(df, columns, tokenizer, max_sequence_length, Q=True):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        if Q:
            t, q = instance.question_title, instance.question_body
            t, q = _trim_input(t, q, max_sequence_length)
            ids, masks= _convert_to_bert_inputs(t, q, tokenizer, max_sequence_length)
            input_ids.append(ids)
            input_masks.append(masks)

        else:
            t, a = instance.question_title, instance.answer
            t, a = _trim_input(t, a, max_sequence_length)
            ids, masks = _convert_to_bert_inputs(t, a, tokenizer, max_sequence_length)
            input_ids.append(ids)
            input_masks.append(masks)


    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, valid_data, test_data, batch_size=16, fold=None, Q=True):

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        self.rho = 0
        self.batch_size = batch_size
        self.fold = fold
        self.Q = Q

    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        # self.test_predictions = []
        self.model.stop_training = False

    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))

        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))

        print("\nvalidation rho: %.4f" % rho_val)
        if rho_val >= self.rho:
            self.rho = rho_val
        else:
            self.model.stop_training = True
        if self.fold is not None:
            if epoch >= 0:
                if self.Q:
                    self.model.save_weights(f'R_F_B_Q/roberta-base-features-question-{fold}-{epoch}.h5py')
                else:
                    self.model.save_weights(f'R_F_B_A/roberta-base-features-answer-{fold}-{epoch}.h5py')

        # self.test_predictions.append(
        #     self.model.predict(self.test_inputs, batch_size=self.batch_size)
        # )

def train_and_predict(model, train_data, valid_data, test_data,
                      learning_rate, epochs, batch_size, loss_function, fold, Q=True):
    if Q:
        custom_callback = CustomCallback(
            valid_data=(valid_data[0], valid_data[1]),
            test_data=test_data,
            batch_size=batch_size,
            fold=5,
            Q=True)
    else:
        custom_callback = CustomCallback(
            valid_data=(valid_data[0], valid_data[1]),
            test_data=test_data,
            batch_size=batch_size,
            fold=5,
            Q=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
        optimizer,
        loss_scale='dynamic')
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, steps_per_epoch=None,
              batch_size=batch_size, callbacks=[custom_callback])

    return custom_callback
targets = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'
    ]

# input_columns = ['question_title', 'question_body']
input_columns = ['question_title', 'answer']

find = re.compile(r"^[^.]*")

df_train['netloc'] = df_train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
df_test['netloc'] = df_test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

features = ['netloc', 'category']
merged = pd.concat([df_train[features], df_train[features]])
ohe = OneHotEncoder()
ohe.fit(merged)

features_train = ohe.transform(df_train[features]).toarray()
# features_test = ohe.transform(df_test[features]).toarray()

module_url = "./universalsentenceencoderlarge4/"
embed = hub.load(module_url)

embeddings_train = {}
embeddings_test = {}

for text in input_columns:
    print(text)
    train_text = df_train[text].str.replace('?', '.').str.replace('!', '.').tolist()
    # test_text = df_test[text].str.replace('?', '.').str.replace('!', '.').tolist()

    curr_train_emb = []
    # curr_test_emb = []
    batch_size = 4
    ind = 0
    while ind * batch_size < len(train_text):
        curr_train_emb.append(embed(train_text[ind * batch_size: (ind + 1) * batch_size])["outputs"].numpy())
        ind += 1

    # ind = 0
    # while ind * batch_size < len(test_text):
    #     curr_test_emb.append(embed(test_text[ind * batch_size: (ind + 1) * batch_size])["outputs"].numpy())
    #     ind += 1

    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)
    # embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)

del embed
K.clear_session()
gc.collect()

l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)

cos_dist = lambda x, y: (x*y).sum(axis=1)

# dist_features_train = np.array([
#     l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
#     cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding'])
# ]).T

dist_features_train = np.array([
    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
]).T

# dist_features_test = np.array([
#     l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
#     cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding'])
# ]).T
#
# dist_features_test = np.array([
#     l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
#     cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding'])
# ]).T

feature_embeddings = np.hstack([item for k, item in embeddings_train.items()] + [features_train, dist_features_train])
print("feature_shape:")
print(feature_embeddings.shape)
print("----------------------------------------------------")
# X_test = np.hstack([item for k, item in embeddings_test.items()] + [features_test, dist_features_test])
# y_train = train[targets].values
feature_shape = feature_embeddings.shape[1]
print(feature_shape)




def bert_model_answer():
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    feature_embeddings = tf.keras.layers.Input(
        (feature_shape,), dtype=tf.float32, name='feature_embeddings')


    bert_model = TFRobertaModel.from_pretrained(BERT_PATH, output_hidden_states=True)

    hidden_output = bert_model([input_word_ids, input_masks])[2]
    last_cat_sequence = tf.concat(
        (hidden_output[-1], hidden_output[-2], hidden_output[-3]),
        2,
    )
    x = tf.keras.layers.GlobalAveragePooling1D()(last_cat_sequence)
    x = tf.concat((x, feature_embeddings), 1)
    out = tf.keras.layers.Dense(9, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, feature_embeddings], outputs=out)

    return model

class CustomSchedule(tf.keras.optimizers.schedules.PolynomialDecay):

    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 end_learning_rate=0.0001,
                 power=1.0,
                 cycle=False,
                 name=None,
                 num_warmup_steps=1000):
        # Since we have a custom __call__() method, we pass cycle=False when calling `super().__init__()` and
        # in self.__call__(), we simply do `step = step % self.decay_steps` to have cyclic behavior.
        super(CustomSchedule, self).__init__(initial_learning_rate, decay_steps, end_learning_rate, power, cycle=False,
                                             name=name)

        self.num_warmup_steps = num_warmup_steps

        self.cycle = tf.constant(cycle, dtype=tf.bool)

    def __call__(self, step):
        """ `step` is actually the step index, starting at 0.
        """

        # For cyclic behavior
        step = tf.cond(self.cycle and step >= self.decay_steps, lambda: step % self.decay_steps, lambda: step)

        learning_rate = super(CustomSchedule, self).__call__(step)

        # Copy (including the comments) from original bert optimizer with minor change.
        # Ref: https://github.com/google-research/bert/blob/master/optimization.py#L25

        # Implements linear warmup: if global_step < num_warmup_steps, the
        # learning rate will be `global_step / num_warmup_steps * init_lr`.
        if self.num_warmup_steps > 0:
            steps_int = tf.cast(step, tf.int32)
            warmup_steps_int = tf.constant(self.num_warmup_steps, dtype=tf.int32)

            steps_float = tf.cast(steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            # The first training step has index (`step`) 0.
            # The original code use `steps_float / warmup_steps_float`, which gives `warmup_percent_done` being 0,
            # and causing `learning_rate` = 0, which is undesired.
            # For this reason, we use `(steps_float + 1) / warmup_steps_float`.
            # At `step = warmup_steps_float - 1`, i.e , at the `warmup_steps_float`-th step,
            # `learning_rate` is `self.initial_learning_rate`.
            warmup_percent_done = (steps_float + 1) / warmup_steps_float

            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

            is_warmup = tf.cast(steps_int < warmup_steps_int, tf.float32)
            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        return learning_rate

gkf = GroupKFold(n_splits=8).split(X=df_train.question_body, groups=df_train.question_body) ############## originaln_splits=5

outputs_question = compute_output_arrays(df_train, output_categories_question)
inputs_question = compute_input_arays(df_train, input_categories_question, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs_question = compute_input_arays(df_test, input_categories_question, tokenizer, MAX_SEQUENCE_LENGTH)

outputs_answer = compute_output_arrays(df_train, output_categories_answer)
inputs_answer = compute_input_arays(df_train, input_categories_answer, tokenizer, MAX_SEQUENCE_LENGTH,Q=False)
test_inputs_answer = compute_input_arays(df_test, input_categories_answer, tokenizer, MAX_SEQUENCE_LENGTH,Q=False)

# num_train_steps = 4863 * 8 / 4

histories_answer = []
#strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

for fold, (train_idx, valid_idx) in enumerate(gkf):
    num_train_steps = len(train_idx) * 8 / 16
    if fold >= 0 :
        learning_rate = CustomSchedule(
            initial_learning_rate=3e-5,
            decay_steps=num_train_steps,
            end_learning_rate=0,
            power=1.0,
            cycle=False,
            num_warmup_steps=int(0.06 * num_train_steps)
        )
        # will actually only do 3 folds (out of 5) to manage < 2h
        if fold < 30:
            # if fold in [0, 1, 5, 7]:
            K.clear_session()
            model = bert_model_answer()


            train_inputs = [inputs_answer[i][train_idx] for i in range(2)]
            train_inputs.append(feature_embeddings[train_idx])
            train_outputs = outputs_answer[train_idx]

            # valid_inputs = [inputs_question[i][valid_idx] for i in range(3)]
            # valid_inputs.append(feature_embeddings[valid_idx])
            # valid_outputs = outputs_question[valid_idx]
            valid_inputs = [inputs_answer[i][valid_idx] for i in range(2)]
            valid_inputs.append(feature_embeddings[valid_idx])
            valid_outputs = outputs_answer[valid_idx]
            # history contains two lists of valid and test preds respectively:
            #  [valid_predictions_{fold}, test_predictions_{fold}]
            history = train_and_predict(model,
                                        train_data=(train_inputs, train_outputs),
                                        valid_data=(valid_inputs, valid_outputs),
                                        test_data=test_inputs_answer,
                                        learning_rate=learning_rate, epochs=8, batch_size=16,
                                        loss_function='binary_crossentropy', fold=fold, Q=False)

