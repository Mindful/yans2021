import sqlite3
from collections import namedtuple
from typing import List, Iterable, Tuple, Callable
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
WORD_TABLE_SCHEMA = '(id INTEGER PRIMARY KEY, ' + ', '.join(f'{name} {type_}' for name, type_ in word_attributes) + \
                    ', FOREIGN KEY (sentence) REFERENCES sentences (id) )'
SENTENCE_TABLE_SCHEMA = '(id INTEGER PRIMARY KEY, sent TEXT NOT NULL)'


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


class WriteBuffer:
    def __init__(self, name: str, save_function: Callable, buffer_size: int = 2000000):
        self.name = name
        self.save_function = save_function
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.buffer_size:
            self.flush()

    def add_many(self, items: Iterable):
        self.buffer.extend(items)
        if len(self.buffer) > self.buffer_size:
            self.flush()

    def flush(self):
        items_to_save = self.buffer[:self.buffer_size]
        logger.info(f'Saving {len(items_to_save)} {self.name}s')
        self.save_function(items_to_save)
        self.buffer = self.buffer[self.buffer_size:]

    def __del__(self):
        if len(self.buffer) > 0:
            raise RuntimeError(f'f{self.name} write buffer destroyed with {len(self.buffer)} items remaining in memory')


class DbConnection:
    def __init__(self, db_name: str):
        self.db_name = db_name + '.db'
        con = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        self.con = con

        self.con.execute('PRAGMA synchronous = 0')
        self.con.execute('PRAGMA journal_mode = OFF')
        self.con.execute('PRAGMA cache_size = 1000000')
        self.con.execute('PRAGMA locking_mode = EXCLUSIVE')
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
            args = list(args)[1:]
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

    def save_sentences(self, sents: List[str]):
        self.cur.executemany(f'INSERT OR IGNORE INTO sentences (sent) VALUES (?)', ((x,) for x in sents))
        self.con.commit()

    def save_words(self, words: List[Word]) -> None:
        self.cur.executemany(f'INSERT INTO words ({",".join(name for name, type_ in word_attributes)})'
                             f' values ({",".join("?" for x in word_attributes)})', words)
        self.con.commit()










