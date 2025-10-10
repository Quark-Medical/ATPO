import sys
import random
import os
import pandas as pd
from datasets import Dataset

def judge_has_fraction(in_file_list):
    try:
        for i in range(len(in_file_list)):
            if i % 2 == 0:
                in_file_list[i] = float(in_file_list[i])
    except:
        return None
    return True

def find_true_file(in_file):
    if os.path.exists(in_file + '.jsonl'):
        in_file = in_file + '.jsonl'
    elif os.path.exists(in_file + '.json'):
        in_file = in_file + '.json'
    else:
        raise Exception('File not found, [{}]'.format(in_file))
    return in_file

def make_map_fn(split):
    def process_fn(example, idx):
        answer_raw = example['ground_truth']
        data = {
            "index": int(example['index']),
            "agent_name": example['agent_name'],
            "prompt": example['prompt'],
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_raw
            },
            "extra_info": {
                'split': split,
                'index': int(example['index']),
                'ground_truth': answer_raw,
                "prompt": example['prompt']
            },
            "data_source": "mediq"
        }
        for k, v in example.get('extra_info', {}).items():
            if k not in data['extra_info']:
                data['extra_info'][k] = v
        if 'API_KEY' in os.environ:
            data['api_key'] = os.environ['API_KEY']
        return data

    return process_fn


def merge_frac_data(split_type, out_file, in_file_list):
    df_list = list()
    for i in range(len(in_file_list) // 2):
        frac = float(in_file_list[2 * i])
        in_file = in_file_list[2 * i + 1]
        in_file = find_true_file(in_file)
        df = pd.read_json(in_file, lines=True)
        sampled_df = df.sample(frac=frac, random_state=42)
        print(f'success add data {len(sampled_df)} from {in_file_list} with frac {frac}!')
        df_list.append(sampled_df)
    df = pd.concat(df_list)
    df = df.fillna('')
    # df.to_json(out_file, lines=True, force_ascii=False, orient='records')
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(function=make_map_fn(split_type), with_indices=True)
    dataset.to_parquet(out_file)
    print(f"processed {split_type}_dataset {dataset}")


def merge_files(split_type, out_file, in_file_list):
    df_list = list()
    for in_file in in_file_list:
        in_file = find_true_file(in_file)
        sampled_df = pd.read_json(in_file, lines=True)
        print(f'success add data {len(sampled_df)} from {in_file_list}!')
        df_list.append(sampled_df)
    df = pd.concat(df_list)
    df = df.fillna('')
    # df.to_json(out_file, lines=True, force_ascii=False, orient='records')
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(function=make_map_fn(split_type), with_indices=True)
    dataset.to_parquet(out_file)
    print(f"processed {split_type}_dataset {dataset}")


if __name__ == '__main__':
    split_type = sys.argv[1]
    out_file = sys.argv[2]
    in_file_list = sys.argv[3:]

    print('[mix] out_file: [{}], in_file_list: [{}]'.format(out_file, in_file_list))
    has_fraction = judge_has_fraction(in_file_list)

    if has_fraction:
        merge_frac_data(split_type, out_file, in_file_list)
    else:
        merge_files(split_type, out_file, in_file_list)
