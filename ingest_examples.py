import argparse
import logging

import spacy
from datasets import tqdm
from generationary.converters.utils import read_contexts
from data.db import DbConnection, WriteBuffer, Example
from nlp.embedding import EmbeddingExtractor
import re

special_token_re = re.compile(r'<.*?>\s')


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run', required=False, default='default_run')

    args = parser.parse_args()

    db = DbConnection(args.run+'_sentences')
    write_buffer = WriteBuffer('sentence', db.save_examples)
    spacy.require_gpu()
    extractor = EmbeddingExtractor()

    skipped = {
        'did_not_find_word': 0,
        'spaces': 0
    }

    total = 0

    for context in tqdm(read_contexts(args.input)):
        total += 1
        input_form = context.meta['lemma']
        if '_' in input_form:
            skipped['spaces'] += 1
            continue

        sentence = special_token_re.sub('', context.line_src())
        doc = extractor.nlp(special_token_re.sub('', context.line_src()))

        word_gen = (Example(input_form, token.text, token.lemma_.lower(), token.pos, sentence, embedding)
                    for token, embedding in extractor.get_word_embeddings(doc))

        output_word = next((x for x in word_gen if x.form == input_form.lower()), None)
        if output_word is None:
            skipped['did_not_find_word'] += 1
            continue

        write_buffer.add(output_word)

    write_buffer.flush()
    logger.info("Finished writing")
    total_skipped = sum(x for x in skipped.values())
    logger.info(f"Found {total} total examples, skipped {total_skipped} of those")
    print(str({x: y/total for x, y in skipped.items()}))


if __name__ == '__main__':
    main()
