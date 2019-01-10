import sys
import pickle
import pandas as pd

one_file = sys.argv[1]
track_len = int(sys.argv[2])

track_features = pd.read_pickle('./data/track_features.pkl.gz')
track_ids_slnos = pd.read_pickle('./data/track_ids_slnos.pkl.gz')


def process_for_feats(for_feats):
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

    return for_feats.reset_index(drop=True)


tmp = pd.read_csv('./data/test_set/log_prehistory_{}'.format(one_file))
tmp = tmp.loc[lambda x: x.session_length == track_len]
tmp = pd.merge(tmp, track_ids_slnos, on=['track_id_clean'], how='inner')
tmp.sort_values(['session_id', 'session_position'], inplace=True)
tmp.reset_index(drop=True, inplace=True)
train_feats = process_for_feats(tmp)

tmp = pd.read_csv('./data/test_set/log_input_{}'.format(one_file))
tmp = tmp.loc[lambda x: x.session_length == track_len]
tmp = pd.merge(tmp, track_ids_slnos, on=['track_id_clean'], how='inner')
tmp.sort_values(['session_id', 'session_position'], inplace=True)
tmp.reset_index(drop=True, inplace=True)
train_df = tmp

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
train_pre_pred = train_feats.groupby('session_id')['skip_2'].apply(lambda x: x.tolist()).tolist()
train_to_pred_tracks = train_df.groupby('session_id')['track_slno'].apply(lambda x: x.tolist()).tolist()

with open('./tmp/{}-{}.pkl'.format(one_file, track_len), 'wb') as f:
    pickle.dump((
        train_feats, train_feats_dummies,
        track_features, train_seq,
        train_track_seq, train_to_pred_tracks,
        train_df, train_pre_pred), f)
