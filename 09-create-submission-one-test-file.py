import sys
import pandas as pd

one_file = sys.argv[1]

df = pd.concat([
    pd.read_pickle('./pred/{}-{}.pkl'.format(one_file, i))
    for i in range(10, 21)
])

sid_index = pd.read_csv('./data/test_set/log_input_{}'.format(one_file))\
    .reset_index(drop=False)\
    .rename(columns={'index': 'sid_index'})\
    .groupby('session_id')['sid_index'].first().reset_index()

df.pred = (df.pred >= 0.5).astype('int64').astype('str')
df.sort_values(['session_id', 'session_position'], inplace=True)
tosave = df.groupby('session_id')['pred'].apply(lambda x: ''.join(x.values)).reset_index()
tosave = pd.merge(tosave, sid_index, on='session_id', how='inner').sort_values('sid_index')
tosave.loc[:, ['pred']].to_csv('./s/{}'.format(one_file), index=False, header=False)
