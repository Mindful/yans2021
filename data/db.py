import sqlite3
from collections import namedtuple
from typing import List, Iterable
from itertools import count
import logging
from tqdm import tqdm
import numpy as np
import io

DB_NAME = 'data.db'

word_attributes = [
    ('form', 'TEXT NOT NULL'),
    ('lemma', 'TEXT'),
    ('pos', 'INT NOT NULL'),
    ('sentence', 'INT NOT NULL'),
    ('embedding', 'ARRAY NOT NULL')
]
Word = namedtuple('Word', [name for name, type_ in word_attributes])
WORD_TABLE_SCHEMA = '(' + ', '.join(f'{name} {type_}' for name, type_ in word_attributes) + ')'
SENTENCE_TABLE_SCHEMA = '(id INTEGER NOT NULL PRIMARY KEY, sent TEXT NOT NULL)'


# https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
def adapt_array(arr) -> sqlite3.Binary:
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text) -> np.ndarray:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT/BLOB when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT/BLOB to np.array when selecting
sqlite3.register_converter("ARRAY", convert_array)

logger = logging.getLogger()


class DbConnection:
    def __init__(self, run_name: str, db_name: str = DB_NAME, word_cache_size: int = 10000):
        logger.info(f'Create database connection for run name {run_name}')
        self.run_name = run_name
        self.db_name = db_name
        self.con = None
        self.cur = None
        self.word_cache_size = word_cache_size
        self.word_cache = []

        self.words_table = f'{self.run_name}_words'
        self.sents_table = f'{self.run_name}_sents'

        self.sent_counter = count()
        self.is_done = False

    def connect_for_saving(self):
        con = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        logger.info(f'Creating tables {self.words_table} and {self.sents_table}')
        cur.execute(f'CREATE TABLE {self.words_table}{WORD_TABLE_SCHEMA}')
        cur.execute(f'CREATE TABLE {self.sents_table}{SENTENCE_TABLE_SCHEMA}')
        con.commit()
        self.con = con
        self.cur = cur

    def connect_for_reading(self):
        con = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        self.con = con
        self.cur = cur

    def read_words(self, include_sentences: bool = False, use_tqdm: bool = False) -> Iterable[Word]:
        if self.con is None:
            raise ConnectionError('Must initialize DB connection before reading')

        word_total = self.cur.execute(f'SELECT COUNT(*) FROM {self.words_table}').fetchone()[0] if use_tqdm else None
        lemma_index = next(idx for idx, tpl in enumerate(word_attributes) if tpl[0] == 'lemma')
        form_index = next(idx for idx, tpl in enumerate(word_attributes) if tpl[0] == 'form')

        def build_word(args: List) -> Word:
            args = list(args)
            if args[lemma_index] is None:
                args[lemma_index] = args[form_index]

            return Word(*args)

        if include_sentences:
            word_cursor = self.cur.execute(f'''
                select {self.words_table}.*, {self.sents_table}.sent from {self.words_table}
                join {self.sents_table} on {self.words_table}.sentence = {self.sents_table}.id
                ''')

            sent_idx = [description[0] for description in self.cur.description].index('sentence')

            for row in tqdm(word_cursor, disable=not use_tqdm, total=word_total, desc='reading words with sentences'):
                word_data = list(row[:len(word_attributes)])
                word_data[sent_idx] = row[-1]  # the last element is the sentence text, use that to replace sentence ID
                yield build_word(word_data)

        else:
            word_cursor = self.cur.execute(f'SELECT * from {self.words_table}')
            for row in tqdm(word_cursor, disable=not use_tqdm, total=word_total, desc='reading words'):
                yield build_word(row)

    def add_words(self, words: Iterable[Word]):
        self.word_cache.extend(words)
        if len(self.word_cache) > self.word_cache_size:
            self._save_cache()

    def done(self):
        if len(self.word_cache) > 0:
            self._save_cache()

        self.cur.close()
        self.con.close()
        self.is_done = True

    def _save_cache(self):
        words_to_save = self.word_cache[:self.word_cache_size]
        logger.info(f'Saving {len(words_to_save)} words')
        self._save_words(words_to_save)
        self.word_cache = self.word_cache[self.word_cache_size:]

    def _save_words(self, words: List[Word]) -> None:
        if self.con is None:
            raise ConnectionError('Must initialize DB connection before inserting')

        sents = {sent: next(self.sent_counter) for sent in set(word.sentence for word in words)}

        def rebuild_word(word: Word) -> Word:
            return Word(**{**word._asdict(),
                           **{'sentence': sents[word.sentence],
                              'lemma': word.lemma if word.lemma != word.form else None}
                           })

        words = [rebuild_word(word) for word in words]

        self.cur.executemany(f'INSERT INTO {self.words_table} values ({",".join("?" for x in word_attributes)})', words)
        self.cur.executemany(f'INSERT INTO {self.sents_table} values (?, ?)', ((ident, text) for text, ident in sents.items()))
        self.con.commit()

    def __del__(self):
        if not self.is_done:
            raise RuntimeError('Database connection destroyed before done() was called')












