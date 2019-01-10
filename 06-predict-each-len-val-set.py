import sys
import os
import numpy as np
import pandas as pd
from keras.utils import Sequence
from keras.models import Model
import keras.layers as kl
import keras.optimizers as ko
import keras.backend as K

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


tmp = pd.read_pickle('./data/training_set_sample_data_2.pkl.gz')
tmp = tmp.loc[lambda x: x.session_length == track_len]
tmp.sort_values(['session_id', 'session_position'], inplace=True)
tmp.reset_index(drop=True, inplace=True)
tmp = pd.merge(tmp, track_ids_slnos, on=['track_id_clean'], how='inner')
tmp.sort_values(['session_id', 'session_position'], inplace=True)
train_df, train_feats = prepare_data_1(tmp)

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

train_feats.reset_index(drop=False, inplace=True)
train_feats['index'] += 1
train_feats.set_index('index', inplace=True, drop=True, verify_integrity=True)

train_seq = train_feats.reset_index().groupby('session_id')['index'].apply(lambda x: x.tolist()).tolist()
train_track_seq = train_feats.groupby('session_id')['track_slno'].apply(lambda x: x.tolist()).tolist()
train_df.skip_2 = train_df.skip_2.astype('int64')
train_pre_pred = train_feats.groupby('session_id')['skip_2'].apply(lambda x: x.tolist()).tolist()
train_to_pred_y = train_df.groupby('session_id')['skip_2'].apply(lambda x: x.tolist()).tolist()
train_to_pred_tracks = train_df.groupby('session_id')['track_slno'].apply(lambda x: x.tolist()).tolist()

cols_order = [
    'context_switch',
    'context_type_catalog',
    'context_type_charts',
    'context_type_editorial_playlist',
    'context_type_personalized_playlist',
    'context_type_radio',
    'context_type_user_collection',
    'day',
    'hist_user_behavior_is_shuffle',
    'hist_user_behavior_n_seekback',
    'hist_user_behavior_n_seekfwd',
    'hist_user_behavior_reason_end_backbtn',
    'hist_user_behavior_reason_end_endplay',
    'hist_user_behavior_reason_end_fwdbtn',
    'hist_user_behavior_reason_end_merged',
    'hist_user_behavior_reason_end_remote',
    'hist_user_behavior_reason_end_trackdone',
    'hist_user_behavior_reason_start_appload',
    'hist_user_behavior_reason_start_backbtn',
    'hist_user_behavior_reason_start_clickrow',
    'hist_user_behavior_reason_start_fwdbtn',
    'hist_user_behavior_reason_start_merged',
    'hist_user_behavior_reason_start_playbtn',
    'hist_user_behavior_reason_start_remote',
    'hist_user_behavior_reason_start_trackdone',
    'hist_user_behavior_reason_start_trackerror',
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


session_embed_mat_sizes = {
    10: 12495071,
    11: 12495071,
    12: 14994085,
    13: 14994085,
    14: 17493099,
    15: 17493099,
    16: 19992113,
    17: 19992113,
    18: 19992113,
    19: 22491127,
    20: 22491127
}

session_embed_mat = np.concatenate([
    np.zeros((1, 38)),
    train_feats_dummies.loc[:, cols_order].values,
    np.zeros((session_embed_mat_sizes[track_len] - 1 - train_feats_dummies.shape[0], 38))
])

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
x1.shape

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
model.compile(optimizer=ko.RMSprop(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights('./model_weights_76_l{}.hdf5'.format(track_len))


class TestGenerator(Sequence):
    def __init__(self,
                 session_seq,
                 prehist_track_seq,
                 prehist_pred_seq,
                 topred_track_seq,
                 batch_size):

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
        return [
            x1_batch,
            x2_batch,
            x3_batch,
            x5_batch
        ]


test_generator = TestGenerator(
    session_seq=train_seq,
    prehist_track_seq=train_track_seq,
    prehist_pred_seq=train_pre_pred,
    topred_track_seq=train_to_pred_tracks,
    batch_size=2048)

assert model.layers[3].name == 'session_embed'
assert model.layers[4].name == 'track_embed'

model.layers[3].set_weights([session_embed_mat])
model.layers[4].set_weights([track_embed_mat])

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

assert len(preds_flat) == train_df.shape[0]
train_df['pred'] = pd.Series(preds_flat, index=train_df.index)
train_df.to_pickle('76-l{}-eval-preds.pkl'.format(track_len))
