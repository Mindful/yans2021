import argparse
import logging
import re
from pathlib import Path

from generationary.converters.utils import Context, read_contexts_with_line
from tqdm import tqdm
import csv
import numpy as np


special_token_re = re.compile(r'<.*?>\s')

with open('count_1w.txt') as countfile:
    reader = csv.reader(countfile, delimiter='\t')
    word_freq = {word: int(freq) for word, freq in reader}


def compute_length(context: Context) -> int:
    sentence = special_token_re.sub('', context.line_src())
    return len(sentence)


def compute_freq(context: Context) -> int:
    return word_freq.get(context.meta['lemma'], 0)


split_metrics = {
    'freq': compute_freq,
    'length': compute_length
}



def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metric', choices=list(split_metrics.keys()), required=True)
    parser.add_argument('--split_count', required=False, default=4)

    args = parser.parse_args()
    if 100 % args.split_count != 0:
        raise Exception("100 must be divisible by split_count")

    compute_metric = split_metrics[args.metric]

    contexts = []
    for context, line in tqdm(read_contexts_with_line(args.input), 'reading contexts'):
        context.metric = compute_metric(context)
        context.line = line
        contexts.append(context)

    contexts.sort(key=lambda x: x.metric)
    print(f'Found {len(contexts)} contexts')
    metric_values = [x.metric for x in contexts]
    split_percentage = 100 // args.split_count
    splits = []

    floor = 0
    for i in range(1, args.split_count+1):
        ceiling = np.percentile(metric_values, i * split_percentage)
        if i == args.split_count:
            ceiling += 1  # raise the ceiling so that we get the last element in the array
        splits.append([c for c in contexts if floor <= c.metric < ceiling])
        floor = ceiling

    for idx, split in enumerate(splits):
        print(f'Split {idx} of size {len(split)}')

    assert sum(len(x) for x in splits) == len(contexts)

    output = Path(args.output)
    output.mkdir(exist_ok=True)
    for i in range(1, args.split_count+1):
        fname = f'split_{i}.txt'
        outfile = output / Path(fname)
        with outfile.open('w') as f:
            f.writelines(x.line + '\n' for x in splits[i-1])

        print(f'Wrote {fname}')

main()


