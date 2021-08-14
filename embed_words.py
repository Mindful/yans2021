import argparse
import csv
import os
import logging
import queue
from collections import Counter

import spacy
from multiprocessing import Process, Queue

from tqdm import tqdm

from nlp.embedding import EmbeddingExtractor, reduction_function
from data.db import DbConnection, Word, WriteBuffer

MAX_PER_LEMMA = 1000
MAX_TOTAL = 10000000

ABORT_INSTRUCTION = -1


def get_target_words() -> set[str]:
    with open('count_1w.txt', 'r') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    return set(x[0] for x in rows)


def embedding_executor(word_queue: Queue, instruction_queue: Queue, word_set: set[str], process_num: int,
                       sentence_bound: range, reduction: str, run: str):
    logging.info(f'Acquiring GPU {process_num}')
    spacy.require_gpu(process_num)
    extractor = EmbeddingExtractor(embedding_reducer=reduction_function[reduction])

    db = DbConnection(run + '_sentences')
    banned_lemmas = set()

    sentence_generator = ((text, ident) for ident, text in db.read_sentences(use_tqdm=False, bound=sentence_bound))
    for doc, ident in extractor.nlp.pipe(sentence_generator, batch_size=500, as_tuples=True):
        if not instruction_queue.empty():
            instruction = instruction_queue.get_nowait()
            if instruction == ABORT_INSTRUCTION:
                logging.info(f'Proc {process_num} quitting early')
                return
            else:  # it's a lemma we have more than enough of
                banned_lemmas.add(instruction)
        try:
            word_gen = (Word(token.text, token.lemma_.lower(), token.pos, ident, embedding, None)
                        for token, embedding in extractor.get_word_embeddings(doc))
            for word in word_gen:
                if word.lemma not in banned_lemmas and (word.lemma in word_set or word.text.lower() in word_set):
                    word_queue.put(1, block=True, timeout=None)
        except Exception as e:
            print(doc)
            raise e

    logging.info(f'Proc {process_num} done')


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=False, default='default_run')
    parser.add_argument('--reduction', required=False, default='first')
    parser.add_argument('--gpus', required=False, type=int, default=8)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parser.parse_args()

    target_words = get_target_words()

    db = DbConnection(args.run + '_sentences')
    logger.info('Counting sentences')
    total_sents = db.count_sentences()
    logger.info(f'Found {total_sents} sentences')
    db.cur.close()
    db.con.close()

    devices = args.gpus
    batch_size = total_sents // devices
    process_metadata = [
        (idx, idx * batch_size, ((idx + 1) * batch_size) if idx != devices - 1 else total_sents + 1)
        for idx in range(devices)
    ]
    logging.info(process_metadata)

    q = Queue(10000)
    instruction_q = Queue(100)

    processes = [
        Process(target=embedding_executor, args=(q, instruction_q, target_words,
                                                 num, range(start, stop), args.reduction, args.run))
        for num, start, stop in process_metadata
    ]

    for proc in processes:
        proc.start()

    pbar = tqdm(total=total_sents, desc='processing sentences')
    processed_words = 0

    queue_has_filled = False

    writing_db = DbConnection(args.run + '_words')
    write_buffer = WriteBuffer('word', writing_db.save_words)
    lemma_counter = Counter()

    while processed_words < MAX_TOTAL:
        if not queue_has_filled and q.full():
            queue_has_filled = True
            print('------Queue filled up------')

        word = q.get(timeout=600)

        lemma_counter[word.lemma] += 1
        if lemma_counter[word.lemma] >= MAX_PER_LEMMA:
            instruction_q.put(word.lemma)

        write_buffer.add(word)
        pbar.update(1)
        processed_words += 1

    pbar.close()
    logging.info('Cleaning up')
    instruction_q.put(ABORT_INSTRUCTION)

    for proc in processes:
        proc.join()

    logging.info('Done')


if __name__ == '__main__':
    main()
