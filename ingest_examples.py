import argparse
import logging
from pathlib import Path

import spacy
from datasets import tqdm
from generationary.converters.utils import read_contexts_with_line, combine_contexts_and_trgs, read_trgs
from data.db import DbConnection, WriteBuffer, Example, example_splits
from nlp.embedding import EmbeddingExtractor
import re

special_token_re = re.compile(r'<.*?>\s')


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()

    parser.add_argument('--contexts', nargs='+', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=Path, required=True)
    parser.add_argument('--tag', default='define')
    parser.add_argument('--run', required=False, default='default_run')
    parser.add_argument('--split', required=True, type=str, choices=list(example_splits.keys()))
    args = parser.parse_args()

    contexts_and_lines = (c for path in args.contexts for c in read_contexts_with_line(path, tag=args.tag))
    contexts = []
    lines = []
    for context, line in tqdm(contexts_and_lines, 'reading contexts'):
        contexts.append(context)
        lines.append(line)

    if args.split != 'test':
        data = None
        for path in args.targets:
            data = read_trgs(path, data)
        contexts = list(tqdm(combine_contexts_and_trgs(contexts, data, tag=args.tag), 'adding target data'))
    else:
        print('skip adding target data for test')

    db = DbConnection(args.run+'_examples')
    write_buffer = WriteBuffer('example', db.save_examples)
    spacy.require_gpu()
    extractor = EmbeddingExtractor()

    partial_save = {
        'did_not_find_word': 0,
        'spaces': 0
    }

    total = 0
    split_id = example_splits[args.split]

    for context, line in tqdm(zip(contexts, lines), 'processing contexts', total=len(contexts)):
        total += 1
        input_form = context.meta['lemma']
        target_line = context.line_trg() if args.split != 'test' else None

        sentence = special_token_re.sub('', context.line_src())
        if '_' in input_form:
            partial_save['spaces'] += 1
            write_buffer.add(Example(input_form, None, None, None, sentence, None, line, target_line, split_id))
            continue

        doc = extractor.nlp(special_token_re.sub('', context.line_src()))

        word_gen = (Example(input_form, token.text, token.lemma_.lower(), token.pos, sentence, embedding, line,
                            target_line, split_id)
                    for token, embedding in extractor.get_word_embeddings(doc))

        output_example = next((x for x in word_gen if x.form == input_form.lower()), None)
        if output_example is None:
            partial_save['did_not_find_word'] += 1
            write_buffer.add(Example(input_form, None, None, None, sentence, None, line, target_line, split_id))
            continue

        write_buffer.add(output_example)

    write_buffer.flush()
    logger.info("Finished writing")
    total_skipped = sum(x for x in partial_save.values())
    logger.info(f"Found {total} total examples, partial saves for {total_skipped} of those")
    print(str({x: y/total for x, y in partial_save.items()}))


if __name__ == '__main__':
    main()
