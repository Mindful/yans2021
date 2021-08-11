import argparse
import logging
from bloom_filter2 import BloomFilter

from data.db import DbConnection, WriteBuffer
from data.input import RawFileReader, JsonFileReader


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run', required=False, default='default_run')
    parser.add_argument('--lines', required=False, type=int)

    args = parser.parse_args()

    seen_sents = BloomFilter(max_elements=200000000, error_rate=0.001)

    if '.txt' in args.input:
        logger.info(f'Reading sentences from file {args.input}')
        sentence_iter = RawFileReader(args.input, total_lines=args.lines)
    elif '.json' in args.input:
        logger.info(f'Reading json file {args.input}')
        sentence_iter = JsonFileReader(args.input, total_lines=args.lines)
    else:
        raise RuntimeError("Unknown input type")

    db = DbConnection(args.run+'_sentences')
    write_buffer = WriteBuffer('sentence', db.save_sentences)

    for sentence in sentence_iter:
        if sentence not in seen_sents:
            seen_sents.add(sentence)
            write_buffer.add(sentence)

    write_buffer.flush()


if __name__ == '__main__':
    main()
