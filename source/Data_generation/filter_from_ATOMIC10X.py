import json

import pandas as pd
from tqdm import tqdm

data = open('../../data/downloaded/ATOMIC10X.jsonl', 'r').readlines()

ATOMIC_relations = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']

atomic10x_sep_dict = {
    'head': [],
    'relation': [],
    'tail': [],
    'split': [],
    'score': []
}
for l in tqdm(data):
    data1 = json.loads(l)
    if not data1['relation'] in ATOMIC_relations:
        continue
    elif data1['p_valid_model'] < 0.5:
        continue
    else:
        atomic10x_sep_dict['head'].append(data1['head'])
        atomic10x_sep_dict['relation'].append(data1['relation'])
        atomic10x_sep_dict['tail'].append(data1['tail'])
        atomic10x_sep_dict['split'].append(data1['split'])
        atomic10x_sep_dict['score'].append(data1['p_valid_model'])

atomic10x_sep_df = pd.DataFrame.from_dict(atomic10x_sep_dict)
atomic10x_sep_df.to_csv('../../data/atomic/ATOMIC10X_sep.csv', index=False)

atomic10x_agg_dict = {
    'event': [], 'oEffect': [], 'oReact': [], 'oWant': [], 'xAttr': [], 'xEffect': [], 'xIntent': [], 'xNeed': [],
    'xReact': [], 'xWant': [], 'prefix': [], 'split': []
}
atomic10x_sep_df['split'] = atomic10x_sep_df['split'].apply(lambda x: {'train': 'trn', 'val': 'dev', 'test': 'tst'}[x])
empty = json.dumps([])
for h in tqdm(atomic10x_sep_df['head'].unique()):
    atomic10x_head = atomic10x_sep_df[atomic10x_sep_df['head'] == h].reset_index(drop=True)
    atomic10x_agg_dict['event'].append(h)
    atomic10x_agg_dict['split'].append(atomic10x_head['split'][0])
    atomic10x_agg_dict['prefix'].append(empty)
    for r in ATOMIC_relations:
        if r in atomic10x_head['relation'].values:
            atomic10x_agg_dict[r].append(
                json.dumps(atomic10x_head[atomic10x_head['relation'] == r]['tail'].values.tolist()))
        else:
            atomic10x_agg_dict[r].append(empty)

atomic10x_agg = pd.DataFrame.from_dict(atomic10x_agg_dict)
atomic10x_agg.to_csv('../../data/atomic/ATOMIC10X_agg.csv', index=False)
