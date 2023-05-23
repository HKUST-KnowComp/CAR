import glob
import json
import os

from tqdm import tqdm

folders = glob.glob('../../data/MixedATOMIC_SyntheticQA/ATOMIC_with_pseudo_*')
for f in tqdm(folders):
    if not os.path.exists(f.replace('MixedATOMIC_SyntheticQA', 'MixedATOMIC_SynQA_NoWildcard')):
        os.makedirs(f.replace('MixedATOMIC_SyntheticQA', 'MixedATOMIC_SynQA_NoWildcard'), exist_ok=True)
        new_save_path = f.replace('MixedATOMIC_SyntheticQA', 'MixedATOMIC_SynQA_NoWildcard')
        for split in ['train', 'dev']:
            data = open('{}/{}_random.jsonl'.format(f, split), 'r').readlines()
            qa = [json.loads(i) for i in tqdm(data) if '_' not in json.loads(i)['context']]
            data2 = open('{}/{}_random.jsonl'.format(new_save_path, split), 'w')
            data2.writelines([json.dumps(i) + '\n' for i in qa])
            data2.close()
