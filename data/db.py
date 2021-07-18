import sqlite3
from collections import namedtuple
from typing import List, Iterable, Tuple
from itertools import count
import logging
from tqdm import tqdm
import numpy as np
import io

word_attributes = [
    ('form', 'TEXT NOT NULL'),
    ('lemma', 'TEXT NOT NULL'),
    ('pos', 'INT NOT NULL'),
    ('sentence', 'INT NOT NULL'),
    ('embedding', 'ARRAY NOT NULL')
]
Word = namedtuple('Word', [name for name, type_ in word_attributes])
WORD_TABLE_SCHEMA = '(' + ', '.join(f'{name} {type_}' for name, type_ in word_attributes) + ')'
SENTENCE_TABLE_SCHEMA = '(id INTEGER PRIMARY KEY, sent TEXT NOT NULL UNIQUE)'


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
    def __init__(self, db_name: str, buffer_size: int = 100000):
        self.db_name = db_name + '.db'
        self.buffer_size = buffer_size
        self.word_buffer = []
        self.sentence_buffer = []

        self.sent_counter = count()
        self.is_done = False

        con = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        self.con = con
        self.cur = cur

        self.cur.execute(f'CREATE TABLE IF NOT EXISTS words{WORD_TABLE_SCHEMA}')
        self.cur.execute(f'CREATE TABLE IF NOT EXISTS sentences{SENTENCE_TABLE_SCHEMA}')
        self.con.commit()

    def read_sentences(self, use_tqdm: bool = False) -> Iterable[Tuple[int, str]]:
        sentences_total = self.cur.execute(f'SELECT COUNT(*) FROM sentences').fetchone()[0] if use_tqdm else None
        for row in tqdm(self.cur.execute('SELECT * from sentences'), disable=not use_tqdm,
                        total=sentences_total, desc='reading sentences'):
            yield row

    def read_words(self, include_sentences: bool = False, use_tqdm: bool = False) -> Iterable[Word]:
        word_total = self.cur.execute(f'SELECT COUNT(*) FROM words').fetchone()[0] if use_tqdm else None
        lemma_index = next(idx for idx, tpl in enumerate(word_attributes) if tpl[0] == 'lemma')
        form_index = next(idx for idx, tpl in enumerate(word_attributes) if tpl[0] == 'form')

        def build_word(args: List) -> Word:
            args = list(args)
            if args[lemma_index] is None:
                args[lemma_index] = args[form_index]

            return Word(*args)

        if include_sentences:
            word_cursor = self.cur.execute(f'''select words.*, sentences.sent from words
                                               join sentences on words.sentence = sentences.id''')

            sent_idx = [description[0] for description in self.cur.description].index('sentence')

            for row in tqdm(word_cursor, disable=not use_tqdm, total=word_total, desc='reading words with sentences'):
                word_data = list(row[:len(word_attributes)])
                word_data[sent_idx] = row[-1]  # the last element is the sentence text, use that to replace sentence ID
                yield build_word(word_data)

        else:
            word_cursor = self.cur.execute(f'SELECT * from words')
            for row in tqdm(word_cursor, disable=not use_tqdm, total=word_total, desc='reading words'):
                yield build_word(row)

    def done(self):
        if len(self.word_buffer) > 0:
            self._save_word_buffer()

        if len(self.sentence_buffer) > 0:
            self._save_sentence_buffer()

        self.cur.close()
        self.con.close()
        self.is_done = True

    def add_sentence(self, sentence: str):
        self.sentence_buffer.append(sentence)
        if len(self.sentence_buffer) > self.buffer_size:
            self._save_sentence_buffer()

    def _save_sentence_buffer(self):
        sents_to_save = self.sentence_buffer[:self.buffer_size]
        logger.info(f'Saving {len(sents_to_save)} sentences')
        self._save_sentences(sents_to_save)
        self.sentence_buffer = self.sentence_buffer[self.buffer_size:]

    def _save_sentences(self, sents: List[str]):
        self.cur.executemany(f'INSERT OR IGNORE INTO sentences (sent) VALUES (?)', ((x,) for x in sents))
        self.con.commit()

    def add_words(self, words: Iterable[Word]):
        self.word_buffer.extend(words)
        if len(self.word_buffer) > self.buffer_size:
            self._save_word_buffer()

    def _save_word_buffer(self):
        words_to_save = self.word_buffer[:self.buffer_size]
        logger.info(f'Saving {len(words_to_save)} words')
        self._save_words(words_to_save)
        self.word_buffer = self.word_buffer[self.buffer_size:]

    def _save_words(self, words: List[Word]) -> None:
        self.cur.executemany(f'INSERT INTO words values ({",".join("?" for x in word_attributes)})', words)
        self.con.commit()

    def __del__(self):
        if not self.is_done:
            raise RuntimeError('Database connection destroyed before done() was called')












