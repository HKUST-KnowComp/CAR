import json
import os.path

import pandas as pd
from tqdm import tqdm

if not os.path.exists('../../data/data_and_models_2.1/ATOMIC10X_with_critic_sep_line.csv'):

    atomic10x = open('../../data/data_and_models_2.1/ATOMIC10X.jsonl', 'r').readlines()

    atomic10x_sep_line_dict = {
        'head': [],
        'relation': [],
        'tail': [],
        'critic': [],
        'split': []
    }
    for a in atomic10x:
        data = json.loads(a)
        atomic10x_sep_line_dict['head'].append(data['head'])
        atomic10x_sep_line_dict['relation'].append(data['relation'])
        atomic10x_sep_line_dict['tail'].append(data['tail'])
        atomic10x_sep_line_dict['critic'].append(data['p_valid_model'])
        atomic10x_sep_line_dict['split'].append(data['split'])

    atomic10x_sep_line_df = pd.DataFrame(atomic10x_sep_line_dict)

    atomic10x_sep_line_df.sort_values(by=['head']).to_csv(
        '../../data/data_and_models_2.1/ATOMIC10X_with_critic_sep_line.csv', index=False)
else:
    atomic10x_sep_line_df = pd.read_csv(
        '../../data/data_and_models_2.1/ATOMIC10X_with_critic_sep_line.csv', index_col=None)

relation_list = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']

atomic10x_agg_dict = {
    'event': [],
    'oEffect': [],
    'oReact': [],
    'oWant': [],
    'xAttr': [],
    'xEffect': [],
    'xIntent': [],
    'xNeed': [],
    'xReact': [],
    'xWant': [],
    'prefix': [],
    'split': [],
    'critic': []
}

# build a mapping dict with keys being each unique event in atomic10x_sep_line_df and values being the row indexes with such event
atomic10x_sep_line_df_event_dict = {}
for i, h in enumerate(tqdm(atomic10x_sep_line_df['head'])):
    if h not in atomic10x_sep_line_df_event_dict:
        atomic10x_sep_line_df_event_dict[h] = [i]
    else:
        atomic10x_sep_line_df_event_dict[h].append(i)

for h in tqdm(atomic10x_sep_line_df['head'].unique()):
    atomic10x_head = atomic10x_sep_line_df.iloc[atomic10x_sep_line_df_event_dict[h]].reset_index(drop=True)
    atomic10x_agg_dict['event'].append(h)
    atomic10x_agg_dict['prefix'].append(json.dumps([j for j in h.split() if
                                                    j.lower() not in ['all', 'personx', 'new', 'will', 'must',
                                                                      'themselves',
                                                                      'were', 'was', 'the', 'can', 'if', 'but', 'under',
                                                                      'should',
                                                                      'being', 'by', 'into', 'also', 'and', 'at',
                                                                      'have', 'same',
                                                                      'them', 'more', 'where', 'over', 'some', 'theirs',
                                                                      'are',
                                                                      'there', 'these', 'may', 'no', 'like', 'personz',
                                                                      'is',
                                                                      'person x', 'your', 'whom', 'up', 'just', 'of',
                                                                      'many',
                                                                      'than', 'because', 'would', 'an', 'between',
                                                                      'any', 'it',
                                                                      'against', 'you', 'person y', 'well', 'itself',
                                                                      'about',
                                                                      'my', 'for', 'only', 'before', 'both', 'had',
                                                                      'persony',
                                                                      'she', 'am', 'shall', 'when', 'he', 'might', 'z',
                                                                      'has',
                                                                      'which', 'his', 'then', 'we', 'y', 'person z',
                                                                      'much',
                                                                      'such', 'on', 'could', 'yourself', 'through',
                                                                      'their', 'to',
                                                                      'x', 'other', 'now', 'in', 'its', 'others',
                                                                      'from', 'those',
                                                                      'with', 'most', 'our', 'that', 'a', 'while',
                                                                      'her', 'very',
                                                                      'they', 'this', 'without', 'after', 'once', 'not',
                                                                      'out',
                                                                      'do', 'myself', 'even', 'or', 'who', 'be',
                                                                      "personx's",
                                                                      "personz's", "persony's", "person x's",
                                                                      "person z's",
                                                                      "person y's", "x's", "y's", "z's"]]))
    atomic10x_agg_dict['split'].append(atomic10x_head['split'][0])
    atomic10x_agg_dict['critic'].append(atomic10x_head['critic'][0])
    # print(atomic10x_head)
    for r in relation_list:
        # print(r, atomic10x_head['relation'], r in atomic10x_head['relation'].values)
        if r not in atomic10x_head['relation'].values:
            atomic10x_agg_dict[r].append(json.dumps([]))
        else:
            atomic10x_agg_dict[r].append(
                json.dumps(atomic10x_head[atomic10x_head['relation'] == r]['tail'].values.tolist()))

atomic10x_agg = pd.DataFrame(atomic10x_agg_dict)
atomic10x_agg.drop(['critic'], axis=1).to_csv('../../data/data_and_models_2.1/ATOMIC10X_with_critic_agg.csv',
                                              index=False)
for threshold in tqdm([0.9, 0.8, 0.7, 0.5]):
    atomic10x_agg[atomic10x_agg['critic'] >= threshold].drop(['critic'], axis=1).to_csv(
        '../../data/data_and_models_2.1/ATOMIC10X_with_critic_agg_{}.csv'.format(threshold), index=False)
