import json
from collections import Counter

import pandas as pd
from tqdm import tqdm

ATOMIC_relations = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
original_atomic = pd.read_csv('../../data/atomic/v4_atomic_all_agg.csv', index_col=None)

annotated = pd.read_csv('../../data/abstractATOMIC/triple_annotated.csv', index_col=None)
annotated = annotated[annotated.label == 1].reset_index(drop=True)
prediction = pd.read_csv('../../data/abstractATOMIC/triple_unlabeled_prediction_trn.csv', index_col=None)


def transform_into_atomic_form(target_csv: pd.DataFrame):
    """
    This function converts a triple-lined csv data into ATOMIC_agg format.
    :param target_csv: The csv to be transformed. with each line being one piece of knowledge.
    :return: The ATOMIC_agg format data.
    """
    append_dict = {i: [] for i in ['event'] + ATOMIC_relations + ['prefix', 'split']}
    for i in tqdm(target_csv['h_id'].unique()):
        triples = target_csv[target_csv.h_id == i].reset_index(drop=True)
        append_dict['event'].append(triples.loc[0, 'head'].replace('[', '').replace(']', ''))
        append_dict['split'].append(original_atomic.loc[i, 'split'])
        append_dict['prefix'].append(original_atomic.loc[i, 'prefix'])

        for r in ATOMIC_relations:
            if r in triples['relation'].unique():
                triples_r = triples[triples.relation == r].reset_index(drop=True)
                append_dict[r].append(json.dumps(list(triples_r['tail'].unique())))
            else:
                append_dict[r].append(json.dumps(['none']))
        # print(append_dict)
    return pd.DataFrame.from_dict(append_dict)


for threshold in tqdm([0.99, 0.98, 0.97, 0.95, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]):
    prediction_threshold = prediction[prediction.score >= threshold].reset_index(drop=True)
    print(Counter(prediction_threshold['relation']))
    merge_annotated = pd.concat([annotated, prediction_threshold], ignore_index=True)
    for id, data in enumerate([prediction_threshold]):  # [prediction_threshold, merge_annotated]
        # transform_into_atomic_form(data).to_csv(
        #     '../../data/MixedATOMIC/{}_{}_only.csv'.format(['pseudo', 'pseudo+annotated'][id], threshold),
        #     index=False)
        total = pd.concat([original_atomic, transform_into_atomic_form(data)], ignore_index=True)
        total.to_csv(
            '../../data/MixedATOMIC/ATOMIC_with_{}_{}.csv'.format(['pseudo', 'pseudo+annotated'][id], threshold),
            index=False)

# pd.concat([original_atomic, transform_into_atomic_form(annotated)], ignore_index=True).to_csv(
#     '../../data/MixedATOMIC/ATOMIC_with_annotated.csv', index=False)
