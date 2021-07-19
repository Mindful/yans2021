import argparse
import os
import logging
import queue

import spacy
from multiprocessing import Process, Queue

from tqdm import tqdm

from nlp.parsing import EmbeddingExtractor, reduction_function
from data.db import DbConnection, Word, WriteBuffer
import torch


def embedding_executor(q: Queue, process_num: int, bound: range, reduction: str, run: str):
    spacy.require_gpu(process_num)
    extractor = EmbeddingExtractor(embedding_reducer=reduction_function[reduction])

    db = DbConnection(run)
    write_buffer = WriteBuffer('word', db.save_words)

    sentence_generator = ((text, ident) for ident, text in db.read_sentences(use_tqdm=False, bound=bound))
    for doc, ident in extractor.nlp.pipe(sentence_generator, batch_size=500, as_tuples=True):
        try:
            word_gen = (Word(token.text, token.lemma_, token.pos, ident, embedding)
                        for token, embedding in extractor.get_word_embeddings(doc))
            write_buffer.add_many(word_gen)
        except Exception as e:
            print(doc)
            raise e

        q.put(1)

    write_buffer.flush()


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=False, default='default_run')
    parser.add_argument('--reduction', required=False, default='first')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parser.parse_args()

    db = DbConnection(args.run)
    logger.info('Counting sentences')
    total_sents = db.count_sentences()
    logger.info(f'Found {total_sents} sentences')

    devices = torch.cuda.device_count()
    batch_size = total_sents // devices
    process_metadata = [
        (idx, idx * batch_size, ((idx + 1) * batch_size) if idx != devices - 1 else total_sents)
        for idx in range(devices)
    ]
    q = Queue()

    processes = [
        Process(target=embedding_executor, args=(q, num, range(start, stop), args.reduction, args.run))
        for num, start, stop in process_metadata
    ]

    for proc in processes:
        proc.start()

    pbar = tqdm(total=total_sents, desc='processing sentences')
    while pbar.n < total_sents:
        try:
            pbar.update(q.get(timeout=10))
        except queue.Empty:
            logging.error('Empty queue')
            break

    pbar.close()
    logging.info('Cleaning up')

    for proc in processes:
        proc.join()

    logging.info('Done')


if __name__ == '__main__':
    main()
