import glob
import os

from tqdm import tqdm

eval_tasks = [
    ("socialiqa", "../../tasks/socialiqa_dev.jsonl"),
    ("winogrande", "../../tasks/winogrande_dev.jsonl"),
    ("piqa", "../../tasks/piqa_dev.jsonl"),
    ("commonsenseqa", "../../tasks/commonsenseqa_dev.jsonl"),
    ("anli", "../../tasks/anli_dev.jsonl")
]

total_models_to_eval = 0
for f in glob.glob('../Training/Output*'):
    for models in glob.glob('{}/roberta*'.format(f)):
        total_models_to_eval += 1
    for models in glob.glob('{}/deberta-v3-large*'.format(f)):
        total_models_to_eval += 1

progress_bar = tqdm(total=total_models_to_eval)

output_folders = glob.glob('../Training/Output*')
for f in output_folders:
    output_split = f.split('_')[-1]

    for models in glob.glob('{}/roberta*'.format(f)):
        training_data = models.split('_')[-1]
        if not os.path.exists("./eval_results/{}_{}_roberta-large".format(output_split, training_data)):
            for reader, dataset in eval_tasks:
                os.system(
                    """python evaluate_RoBERTa.py --lm {} --dataset_file {} --out_dir {} --device 5 --reader {}""".format(
                        models, dataset, "./eval_results/{}_{}_roberta-large".format(output_split, training_data),
                        reader))
        progress_bar.update(1)

    for models in glob.glob('{}/deberta-v3-large*'.format(f)):
        training_data = models.split('_')[-1]
        if not os.path.exists("./eval_results/{}_{}_deberta-v3-large".format(output_split, training_data)):
            for reader, dataset in eval_tasks:
                if reader != 'piqa':
                    os.system(
                        """python evaluate_DeBERTa.py --lm {} --dataset_file {} --out_dir {} --device 5 --reader {}""".format(
                            models, dataset,
                            "./eval_results/{}_{}_deberta-v3-large".format(output_split, training_data),
                            reader))
                else:
                    os.system(
                        """python evaluate_DeBERTa_60MSL.py --lm {} --dataset_file {} --out_dir {} --device 5 --reader {}""".format(
                            models, dataset,
                            "./eval_results/{}_{}_deberta-v3-large".format(output_split, training_data),
                            reader))
        progress_bar.update(1)
