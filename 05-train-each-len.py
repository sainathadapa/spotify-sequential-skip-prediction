import os
import sys
import random
import numpy as np
import pandas as pd
import keras.layers as kl
import keras.optimizers as ko
import keras.callbacks as kc
import keras.backend as K
from keras.utils import Sequence
from keras.models import Model

cuda_dev = sys.argv[1]
track_len = int(sys.argv[2])

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_dev

track_features = pd.read_pickle('./data/track_features.pkl.gz')
track_ids_slnos = pd.read_pickle('./data/track_ids_slnos.pkl.gz')


def prepare_data_1(traindf):
    for_feats = traindf.loc[
        lambda x: x.session_position <= (x.session_length/2)
    ].copy()

    for_feats.skip_1 = for_feats.skip_1.astype('int64')
    for_feats.skip_2 = for_feats.skip_2.astype('int64')
    for_feats.skip_3 = for_feats.skip_3.astype('int64')
    for_feats.not_skipped = for_feats.not_skipped.astype('int64')
    for_feats.hist_user_behavior_is_shuffle = for_feats.hist_user_behavior_is_shuffle.astype('int64')
    for_feats.premium = for_feats.premium.astype('int64')

    for_feats.date = pd.to_datetime(for_feats.date)
    for_feats['wkdy'] = for_feats.date.dt.dayofweek
    for_feats['day'] = for_feats.date.dt.day
    for_feats['month'] = for_feats.date.dt.month
    for_feats['year'] = for_feats.date.dt.year
    for_feats.drop(columns=['date'], inplace=True)

    for_feats.drop(columns=['track_id_clean'], inplace=True)

    where_to_replace = for_feats.hist_user_behavior_reason_start.isin([
        'endplay', 'popup', 'uriopen', 'clickside'
    ]).copy()
    for_feats.loc[where_to_replace, 'hist_user_behavior_reason_start'] = 'merged'

    where_to_replace = for_feats.hist_user_behavior_reason_end.isin([
        'clickrow', 'appload', 'popup', 'uriopen', 'clickside', 'logout'
    ]).copy()
    for_feats.loc[where_to_replace, 'hist_user_behavior_reason_end'] = 'merged'

    for_feats.sort_values(['session_id', 'session_position'], inplace=True)

    traindf = traindf.loc[
        lambda x: x.session_position > (x.session_length/2)
    ].sort_values(['session_id', 'session_position'])

    traindf = traindf.loc[:, [
        'session_id', 'session_position', 'session_length',
        'track_id_clean', 'track_slno', 'skip_2'
    ]].copy()

    traindf.sort_values(['session_id', 'session_position'], inplace=True)

    return (traindf.reset_index(drop=True),
            for_feats.reset_index(drop=True))


tmp = pd.read_pickle('./data/final_samples/training_set_sample_data_l{}.pkl'.format(track_len))
tmp = pd.merge(tmp, track_ids_slnos, on=['track_id_clean'], how='inner')
tmp.sort_values(['session_id', 'session_position'], inplace=True)

sample_ids = tmp.session_id.drop_duplicates().sample(frac=0.2).tolist()

train_df, train_feats = prepare_data_1(
    tmp
    .loc[lambda x: ~x.session_id.isin(sample_ids)]
    .sort_values(['session_id', 'session_position'])
    .reset_index(drop=True))

test_df, test_feats = prepare_data_1(
    tmp
    .loc[lambda x: x.session_id.isin(sample_ids)]
    .sort_values(['session_id', 'session_position'])
    .reset_index(drop=True))

cols_to_select = [
    'context_switch',
    'context_type',
    'day',
    'hist_user_behavior_is_shuffle',
    'hist_user_behavior_n_seekback',
    'hist_user_behavior_n_seekfwd',
    'hist_user_behavior_reason_end',
    'hist_user_behavior_reason_start',
    'hour_of_day',
    'long_pause_before_play',
    'month',
    'no_pause_before_play',
    'not_skipped',
    'premium',
    'session_position',
    'short_pause_before_play',
    'skip_1',
    'skip_2',
    'skip_3',
    'wkdy']

train_feats_dummies = pd.get_dummies(train_feats.loc[:, cols_to_select])
test_feats_dummies = pd.get_dummies(test_feats.loc[:, cols_to_select])

train_feats.reset_index(drop=False, inplace=True)
train_feats['index'] += 1
train_feats.set_index('index', inplace=True, drop=True, verify_integrity=True)

test_feats.reset_index(drop=False, inplace=True)
test_feats['index'] += train_feats.index.max() + 1
test_feats.set_index('index', inplace=True, drop=True, verify_integrity=True)

train_seq = train_feats.reset_index().groupby('session_id')['index'].apply(lambda x: x.tolist()).tolist()
test_seq = test_feats.reset_index().groupby('session_id')['index'].apply(lambda x: x.tolist()).tolist()

train_track_seq = train_feats.groupby('session_id')['track_slno'].apply(lambda x: x.tolist()).tolist()
test_track_seq = test_feats.groupby('session_id')['track_slno'].apply(lambda x: x.tolist()).tolist()

train_df.skip_2 = train_df.skip_2.astype('int64')
test_df.skip_2 = test_df.skip_2.astype('int64')

train_pre_pred = train_feats.groupby('session_id')['skip_2'].apply(lambda x: x.tolist()).tolist()
test_pre_pred = test_feats.groupby('session_id')['skip_2'].apply(lambda x: x.tolist()).tolist()

train_to_pred_y = train_df.groupby('session_id')['skip_2'].apply(lambda x: x.tolist()).tolist()
test_to_pred_y = test_df.groupby('session_id')['skip_2'].apply(lambda x: x.tolist()).tolist()

train_to_pred_tracks = train_df.groupby('session_id')['track_slno'].apply(lambda x: x.tolist()).tolist()
test_to_pred_tracks = test_df.groupby('session_id')['track_slno'].apply(lambda x: x.tolist()).tolist()

cols_order = sorted(train_feats_dummies.columns.values)

session_embed_mat = np.concatenate([
    np.zeros((1, 38)),
    train_feats_dummies.loc[:, cols_order].values,
    test_feats_dummies.loc[:, cols_order].values])

track_embed_mat = np.concatenate([
    np.zeros((1, 29)),
    track_features.values
])

session_embed = kl.Embedding(
    input_dim=session_embed_mat.shape[0],
    output_dim=session_embed_mat.shape[1],
    weights=[session_embed_mat],
    trainable=False,
    mask_zero=False,
    name='session_embed')

track_embed = kl.Embedding(
    input_dim=track_embed_mat.shape[0],
    output_dim=track_embed_mat.shape[1],
    weights=[track_embed_mat],
    trainable=False,
    mask_zero=False,
    name='track_embed')

session_bn = kl.BatchNormalization(name='bn1')
session_transformer = kl.Dense(64, activation='relu', name='session_transformer')

session_input = kl.Input(shape=(None,), dtype='int64', name='session_input')
x1 = session_embed(session_input)
x1 = session_bn(x1)
x1 = session_transformer(x1)

track_bn = kl.BatchNormalization(name='track_bn')
track_transformer = kl.Dense(64, activation='relu', name='track_transformer')

prehist_tracks_input = kl.Input(shape=(None,), dtype='int64', name='prehist_tracks_input')
x2 = track_embed(prehist_tracks_input)
x2 = track_bn(x2)
x2 = track_transformer(x2)

topred_tracks_input = kl.Input(shape=(None,), dtype='int64', name='topred_tracks_input')
x3 = track_embed(topred_tracks_input)
x3 = track_bn(x3)
x3 = track_transformer(x3)

x = kl.concatenate([x1, x2], axis=-1)
lstm1 = kl.Bidirectional(kl.CuDNNLSTM(64, return_sequences=False, return_state=False, name='lstm1'))
prehist_sc_1 = lstm1(x)

x = kl.concatenate([x2, x3], axis=1)
lstm2 = kl.Bidirectional(kl.CuDNNLSTM(64, return_sequences=False, return_state=False, name='lstm2'))
prehist_sc_2 = lstm2(x)

prehist_sc = kl.concatenate([prehist_sc_1, prehist_sc_2])


def repeat_vector(args):
    layer_to_repeat = args[0]
    sequence_layer = args[1]
    return kl.RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


prehist_sc_rep = kl.Lambda(repeat_vector, output_shape=(None, 256))([
    prehist_sc,
    x2
])

x = kl.concatenate([
    prehist_sc_rep,
    x2
])

base_transformer = kl.Dense(256, activation='relu', name='base_transformer')
encoder_2 = kl.Bidirectional(kl.CuDNNLSTM(256, return_sequences=False, return_state=True, name='encoder_2'))

x = base_transformer(x)
_, fwd_sh, fwd_sc, bck_sh, bck_sc = encoder_2(x)

fwd_sh = kl.Dropout(0.2)(fwd_sh)
fwd_sc = kl.Dropout(0.2)(fwd_sc)
bck_sh = kl.Dropout(0.2)(bck_sh)
bck_sc = kl.Dropout(0.2)(bck_sc)

topred_prev_pred_input = kl.Input(shape=(1, 1), dtype='float32', name='topred_prev_pred_input')

decoder_2 = kl.Bidirectional(kl.CuDNNLSTM(256, return_sequences=True, return_state=True, name='decoder_2'))
decoder_3 = kl.CuDNNLSTM(64, return_sequences=True, return_state=True, name='decoder_3')
decoder_4 = kl.Dropout(0.5)
decoder_5 = kl.Dense(1, activation='sigmoid', name='decoder_5')

all_ouputs = []
x = kl.concatenate([
    kl.RepeatVector(1)(prehist_sc),
    kl.RepeatVector(1)(kl.Lambda(lambda x: x[:, 0])(x3))
])
x = base_transformer(x)
x, fwd_sh, fwd_sc, bck_sh, bck_sc = decoder_2(x, initial_state=[fwd_sh, fwd_sc, bck_sh, bck_sc])
x = kl.concatenate([x, topred_prev_pred_input])
x, sh, sc = decoder_3(x)
x = decoder_4(x)
oup = decoder_5(x)
all_ouputs.append(oup)

for i in range(1, int(np.ceil(track_len / 2))):
    x = kl.concatenate([
        kl.RepeatVector(1)(prehist_sc),
        kl.RepeatVector(1)(kl.Lambda(lambda x: x[:, i])(x3))
    ])
    x = base_transformer(x)
    x, fwd_sh, fwd_sc, bck_sh, bck_sc = decoder_2(x, initial_state=[fwd_sh, fwd_sc, bck_sh, bck_sc])
    x = kl.concatenate([x, oup])
    x, sh, sc = decoder_3(x, initial_state=[sh, sc])
    x = decoder_4(x)
    oup = decoder_5(x)
    all_ouputs.append(oup)

out_combined = kl.Lambda(lambda x: K.concatenate(x, axis=1))(all_ouputs)
model = Model(inputs=[session_input,
                      prehist_tracks_input,
                      topred_tracks_input,
                      topred_prev_pred_input],
              outputs=[out_combined])
model.compile(optimizer=ko.RMSprop(lr=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'],
              sample_weight_mode='temporal')

track_weights = {
    5: [1.47948164, 1.15550756, 0.93952484, 0.7775378, 0.64794816],
    6: [1.53926702, 1.22513089, 1.01570681, 0.85863874, 0.73298429,
        0.62827225],
    7: [1.59110833, 1.28428303, 1.07973283, 0.92632018, 0.80359006,
        0.70131497, 0.61365059],
    8: [1.63699919, 1.33584297, 1.13507215, 0.98449404, 0.86403155,
        0.76364614, 0.67760151, 0.60231245],
    9: [1.6782454, 1.38162748, 1.18388219, 1.03557323, 0.91692605,
        0.81805341, 0.73330543, 0.65915095, 0.59323586],
    10: [1.7157535, 1.42285967, 1.22759711, 1.08115019, 0.96399265,
         0.86636138, 0.78267742, 0.70945396, 0.64436644, 0.58578768]
}


class TrainGenerator(Sequence):
    def __init__(self,
                 session_seq,
                 prehist_track_seq,
                 prehist_pred_seq,
                 topred_track_seq,
                 y_seq,
                 batch_size,
                 shuffle=True):

        self.session_seq = session_seq
        self.prehist_track_seq = prehist_track_seq
        self.prehist_pred_seq = prehist_pred_seq
        self.topred_track_seq = topred_track_seq
        self.y_seq = y_seq
        self.batch_size = batch_size

        self.indices = list(range(len(self.session_seq)))

        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.session_seq) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

    def __getitem__(self, i):
        start = self.batch_size * i
        end = min(start + self.batch_size, len(self.session_seq))
        this_batch_ids = self.indices[start:end]

        x1_batch = np.array([self.session_seq[i] for i in this_batch_ids])
        x2_batch = np.array([self.prehist_track_seq[i] for i in this_batch_ids])
        x3_batch = np.array([self.topred_track_seq[i] for i in this_batch_ids])
        x4_batch = np.array([
            [self.prehist_pred_seq[i][0]] + self.prehist_pred_seq[i][:-1]
            for i in this_batch_ids
        ])
        x4_batch = np.expand_dims(x4_batch, -1).astype('float32')
        x5_batch = np.array([
            [self.prehist_pred_seq[i][-1]]
            for i in this_batch_ids
        ])
        x5_batch = np.expand_dims(x5_batch, -1).astype('float32')
        y_batch = np.array([self.y_seq[i] for i in this_batch_ids])

        this_weights = track_weights[int(np.ceil(track_len / 2))]
        sample_weights = np.array([this_weights for _ in range(x1_batch.shape[0])])

        return {
            'session_input': x1_batch,
            'prehist_tracks_input': x2_batch,
            'topred_tracks_input': x3_batch,
            'topred_prev_pred_input': x5_batch
        }, np.expand_dims(y_batch, -1), sample_weights


train_batch_size = 2048
val_batch_size = 2048

train_generator = TrainGenerator(
    session_seq=train_seq,
    prehist_track_seq=train_track_seq,
    prehist_pred_seq=train_pre_pred,
    topred_track_seq=train_to_pred_tracks,
    y_seq=train_to_pred_y,
    batch_size=train_batch_size,
    shuffle=True)

val_generator = TrainGenerator(
    session_seq=test_seq,
    prehist_track_seq=test_track_seq,
    prehist_pred_seq=test_pre_pred,
    topred_track_seq=test_to_pred_tracks,
    y_seq=test_to_pred_y,
    batch_size=val_batch_size,
    shuffle=False)

callbacks = [
    kc.EarlyStopping(monitor='val_loss',
                     patience=15,
                     verbose=1,
                     min_delta=1e-4,
                     mode='min'),
    kc.ReduceLROnPlateau(monitor='val_loss',
                         factor=0.1,
                         patience=4,
                         verbose=1,
                         epsilon=1e-4,
                         mode='min'),
    kc.ModelCheckpoint(monitor='val_loss',
                       filepath='model_weights_l{}.hdf5'.format(track_len),
                       save_best_only=True,
                       save_weights_only=True)
]

hist = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    verbose=2,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=len(val_generator))

model.load_weights('./model_weights_l{}.hdf5'.format(track_len))

model.evaluate_generator(
    val_generator,
    steps=len(val_generator),
    workers=1,
    use_multiprocessing=False
)


class TestGenerator(Sequence):
    def __init__(self,
                 session_seq,
                 prehist_track_seq,
                 prehist_pred_seq,
                 topred_track_seq,
                 batch_size,
                 shuffle=True):

        self.session_seq = session_seq
        self.prehist_track_seq = prehist_track_seq
        self.prehist_pred_seq = prehist_pred_seq
        self.topred_track_seq = topred_track_seq
        self.batch_size = batch_size

        self.indices = list(range(len(self.session_seq)))

    def __len__(self):
        return int(np.ceil(len(self.session_seq) / self.batch_size))

    def __getitem__(self, i):
        start = self.batch_size * i
        end = min(start + self.batch_size, len(self.session_seq))
        this_batch_ids = self.indices[start:end]

        x1_batch = np.array([self.session_seq[i] for i in this_batch_ids])
        x2_batch = np.array([self.prehist_track_seq[i] for i in this_batch_ids])
        x3_batch = np.array([self.topred_track_seq[i] for i in this_batch_ids])
        x4_batch = np.array([
            [self.prehist_pred_seq[i][0]] + self.prehist_pred_seq[i][:-1]
            for i in this_batch_ids
        ])
        x4_batch = np.expand_dims(x4_batch, -1).astype('float32')
        x5_batch = np.array([
            [self.prehist_pred_seq[i][-1]]
            for i in this_batch_ids
        ])
        x5_batch = np.expand_dims(x5_batch, -1).astype('float32')
        return {
            'session_input': x1_batch,
            'prehist_tracks_input': x2_batch,
            'topred_tracks_input': x3_batch,
            'topred_prev_pred_input': x5_batch
        }


test_generator = TestGenerator(
    session_seq=test_seq,
    prehist_track_seq=test_track_seq,
    prehist_pred_seq=test_pre_pred,
    topred_track_seq=test_to_pred_tracks,
    batch_size=val_batch_size)

preds = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers=1,
    use_multiprocessing=False
)

preds_flat = []
for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
        preds_flat.append(preds[i, j, 0])

test_df['pred'] = pd.Series(preds_flat, index=test_df.index)
test_df['label'] = (test_df.pred >= 0.5).astype('int64')

gt = test_df.groupby('session_id')['skip_2'].apply(lambda x: x.tolist()).tolist()
subm = test_df.groupby('session_id')['label'].apply(lambda x: x.tolist()).tolist()


def evaluate(submission, groundtruth):
    ap_sum = 0.0
    first_pred_acc_sum = 0.0
    counter = 0
    for sub, tru in zip(submission, groundtruth):
        if len(sub) != len(tru):
            raise Exception('Line {} should contain {} predictions, but instead contains '
                            '{}'.format(counter+1, len(tru), len(sub)))
        ap_sum += ave_pre(sub, tru, counter)
        first_pred_acc_sum += sub[0] == tru[0]
        counter += 1
    ap = ap_sum/counter
    first_pred_acc = first_pred_acc_sum/counter
    return ap, first_pred_acc


def ave_pre(submission, groundtruth, counter):
    s = 0.0
    t = 0.0
    c = 1.0
    for x, y in zip(submission, groundtruth):
        if x != 0 and x != 1:
            raise Exception('Invalid prediction in line {}, should be 0 or 1'.format(counter))
        if x == y:
            s += 1.0
            t += s / c
        c += 1
    return t/len(groundtruth)


print(evaluate(subm, gt))
