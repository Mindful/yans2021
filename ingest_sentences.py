import argparse
import logging

import datasets
import spacy
from datasets import tqdm
from bloom_filter2 import BloomFilter

from data.db import DbConnection, WriteBuffer
from data.input import RawFileReader, JsonFileReader


def to_sentences(dataset: datasets.Dataset) -> datasets.Dataset:
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe('sentencizer')
    nlp.select_pipes(enable='sentencizer')

    return dataset.map(lambda x: {'sents': [y.text for y in nlp(x['text']).sents]},
                       remove_columns=['title', 'text'], num_proc=10, batch_size=100)


datasets_dict = {
    # raw HF wiki data has tons of garbage, not worth using
    # cirusssearch dump is much better (and can be used with json reader)
    'hf_wiki': lambda: datasets.load_dataset('wikipedia', '20200501.en')['train']
}


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run', required=False, default='default_run')

    args = parser.parse_args()

    seen_sents = BloomFilter(max_elements=100000000, error_rate=0.001)

    if '.txt' in args.input:
        logger.info(f'Reading sentences from file {args.input}')
        sentence_iter = RawFileReader(args.input)
    elif '.json' in args.input:
        logger.info(f'Reading json file {args.input}')
        sentence_iter = JsonFileReader(args.input)
    else:
        dataset = datasets_dict[args.input]()
        logger.info('Sentencizing dataset')
        sentences_dataset = to_sentences(dataset)
        sentence_iter = (sent for example in tqdm(sentences_dataset) for sent in example['sents'])

    db = DbConnection(args.run+'_sentences')
    write_buffer = WriteBuffer('sentence', db.save_sentences)

    for sentence in sentence_iter:
        if sentence not in seen_sents:
            seen_sents.add(sentence)
            write_buffer.add(sentence)

    write_buffer.flush()


if __name__ == '__main__':
    main()
