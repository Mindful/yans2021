import argparse
import logging
import re
from generationary.converters.utils import read_contexts, Context
from tqdm import tqdm
import csv


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

    args = parser.parse_args()

    compute_metric = split_metrics[args.metric]

    contexts = []
    for context in tqdm(read_contexts(args.input), 'reading contexts'):
        context.metric = compute_metric(context)
        contexts.append(context)

    count = 0
    for x in contexts:
        if x.metric > 0:
            count += 1
            if count > 20:
                break
            else:
                sentence = special_token_re.sub('', x.line_src())
                print(x.meta['lemma'], sentence, x.metric)

main()


