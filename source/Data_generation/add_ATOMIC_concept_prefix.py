import json
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange


def remove_mid_bracket(string):
    string = string.replace('(', '')
    string = string.replace(')', '')
    return string


def get_instance(string):
    return string.split('[')[-1].split(']')[0]


pd.set_option('display.max_columns', None)
atomic = pd.read_csv('../../data/atomic/v4_atomic_all_agg.csv', index_col=None)
atomic = atomic[~atomic.event.str.contains('_')].reset_index(drop=True)

concepts_annotated = pd.read_csv('../../data/abstractATOMIC/head_annotated.csv', index_col=None)
concepts_annotated = concepts_annotated[concepts_annotated.score >= 3].reset_index(drop=True)
concepts_pseudo = pd.read_csv('../../data/abstractATOMIC/head_unlabeled_prediction.csv', index_col=None)

concepts_pseudo = concepts_pseudo[concepts_pseudo.score >= 0.8].reset_index(drop=True)
concept = pd.concat([concepts_annotated, concepts_pseudo], ignore_index=True).drop_duplicates(
    subset=['head', 'concept']).sort_values(by=['h_id', 'score'], ascending=False).reset_index(drop=True)

if not os.path.exists('../../data/abstractATOMIC/instance2concept_dict.npy'):
    concept2instance_dict, instance2concept_dict = {}, {}
    concept['instance'] = concept['head'].apply(lambda x: get_instance(x))
    for c in tqdm(concept['instance'].unique(), desc="Adding instance dict"):
        concept_c = concept[concept['instance'] == c].reset_index(drop=True)
        instance2concept_dict[c] = concept_c['concept_text'].drop_duplicates().tolist()

    np.save('../../data/abstractATOMIC/instance2concept_dict.npy', instance2concept_dict)
else:
    instance2concept_dict = np.load('../../data/abstractATOMIC/instance2concept_dict.npy', allow_pickle=True).item()

for i in trange(len(atomic), desc="adding concept to atomic prefix"):
    event = atomic.loc[i, 'event']
    current_prefix = json.loads(atomic.loc[i, 'prefix'])
    for k in instance2concept_dict:
        if k in event:
            current_prefix.extend(instance2concept_dict[k])
    atomic.loc[i, 'prefix'] = json.dumps(list(set(current_prefix)))
atomic.to_csv('../../data/atomic/v4_atomic_all_agg_concept_prefix_woWC.csv', index=False)
atomic[atomic.split == 'trn'].to_csv('../../data/atomic/v4_atomic_agg_concept_prefix_woWC_trn.csv', index=False)
atomic[atomic.split == 'dev'].to_csv('../../data/atomic/v4_atomic_agg_concept_prefix_woWC_dev.csv', index=False)
atomic[atomic.split == 'tst'].to_csv('../../data/atomic/v4_atomic_agg_concept_prefix_woWC_tst.csv', index=False)
