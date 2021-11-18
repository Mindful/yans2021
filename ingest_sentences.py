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
    parser.add_argument('--example_db', required=False)

    args = parser.parse_args()

    if args.example_db is not None:
        example_db_con = DbConnection(args.example_db)
        cur = example_db_con.con.execute("SELECT input_form FROM examples")
        target_forms = {x[0] for x in cur}
        logger.info(f"found {len(target_forms)} target forms")
        saved = 0
        skipped = 0

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
            if args.example_db:
                tokens = set(sentence.split()) #TODO: iffy method of splitting, but probably how input forms are done too
                if len(tokens & target_forms) > 0:
                    seen_sents.add(sentence)
                    saved += 1
                else:
                    skipped += 1
            else:
                seen_sents.add(sentence)

            write_buffer.add(sentence)

    write_buffer.flush()
    if args.example_db:
        print('Saved', saved, 'skipped', skipped)


if __name__ == '__main__':
    main()
