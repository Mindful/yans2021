import argparse
import logging
from typing import Iterable

import datasets
import spacy
from datasets import tqdm

from data.db import DbConnection, WriteBuffer
from data.input import RawFileReader


def to_sentences(dataset: datasets.Dataset) -> Iterable[str]:
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe('sentencizer')
    nlp.select_pipes(enable='sentencizer')

    for doc in nlp.pipe(example['text'] for example in dataset):
        for sentence in doc.sents:
            yield sentence.text


datasets_dict = {
    'wiki': lambda: tqdm(to_sentences(datasets.load_dataset('wikipedia', '20200501.en')['train']))
}


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run', required=False, default='default_run')

    args = parser.parse_args()

    if '.' in args.input:
        logger.info(f'Reading sentences from file {args.input}')
        sentence_iter = RawFileReader(args.input)
    else:
        sentence_iter = datasets_dict[args.input]()

    db = DbConnection(args.run)
    write_buffer = WriteBuffer('sentence', db.save_sentences)

    for sentence in sentence_iter:
        write_buffer.add(sentence)

    write_buffer.flush()

if __name__ == '__main__':
    main()