import sqlite3
from collections import namedtuple
from typing import List, Iterable, Tuple, Callable, Optional
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
WORD_TABLE_SCHEMA = '(id INTEGER PRIMARY KEY, ' + ', '.join(f'{name} {type_}' for name, type_ in word_attributes) + ')'

word_cluster_attributes = [
    ('key', 'TEXT NOT NULL'),
    ('cluster_centers', 'ARRAY NOT NULL'),
    ('labels', 'ARRAY NOT NULL')
]
WordCluster = namedtuple('WordCluster', [name for name, type_ in word_cluster_attributes])
CLUSTER_TABLE_SCHEMA = '(id INTEGER PRIMARY KEY, ' + ', '.join(f'{name} {type_}' for name, type_
                                                               in word_cluster_attributes) + ')'


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
    def __init__(self, name: str, save_function: Callable, buffer_size: int = 500000):
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
        logger.info('Done saving')

    def __del__(self):
        if len(self.buffer) > 0:
            raise RuntimeError(f'f{self.name} write buffer destroyed with {len(self.buffer)} items remaining in memory')


class DbConnection:
    def __init__(self, db_name: str):
        self.db_name = db_name + '.db'
        con = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES,  timeout=600)
        cur = con.cursor()
        self.con = con

        self.con.execute('PRAGMA synchronous = 0')
        self.con.execute('PRAGMA journal_mode = OFF')
        self.con.execute('PRAGMA cache_size = 1000000')
        self.cur = cur
        self.cur.execute(f'CREATE TABLE IF NOT EXISTS words{WORD_TABLE_SCHEMA}')
        self.cur.execute(f'CREATE TABLE IF NOT EXISTS sentences{SENTENCE_TABLE_SCHEMA}')
        self.cur.execute(f'CREATE TABLE IF NOT EXISTS clusters{CLUSTER_TABLE_SCHEMA}')
        self.con.commit()

    def count_sentences(self) -> int:
        return self.cur.execute(f'SELECT COUNT(*) FROM sentences').fetchone()[0]

    def read_sentences(self, use_tqdm: bool = False, bound: Optional[range] = None) -> Iterable[Tuple[int, str]]:
        if bound is None:
            sentences_total = self.count_sentences() if use_tqdm else None
            where_clause = ''
        else:
            where_clause = f' where sentences.id >= {bound.start} and sentences.id < {bound.stop}'
            sentences_total = len(bound)

        for row in tqdm(self.cur.execute('SELECT * from sentences' + where_clause), disable=not use_tqdm,
                        total=sentences_total, desc='reading sentences'):
            yield row

    def count_words(self, where_clause: Optional[str] = None) -> int:
        where_clause = '' if where_clause is None else where_clause
        return self.cur.execute(f'SELECT COUNT(*) FROM words ' + where_clause).fetchone()[0]

    def count_clusters(self, where_clause: Optional[str] = None) -> int:
        where_clause = '' if where_clause is None else where_clause
        return self.cur.execute(f'SELECT COUNT(*) FROM clusters ' + where_clause).fetchone()[0]

    def read_clusters(self, use_tqdm: bool = False, where_clause: Optional[str] = None) -> Iterable[WordCluster]:
        cluster_total = self.count_clusters(where_clause) if use_tqdm else None
        where_clause = '' if where_clause is None else where_clause

        for row in tqdm(self.cur.execute('SELECT * from clusters ' + where_clause), disable=not use_tqdm,
                        total=cluster_total, desc='reading clusters'):
            yield WordCluster(*row[1:])  # skip the first element, which is the ID

    def read_words(self, include_sentences: bool = False, use_tqdm: bool = False,
                   where_clause: Optional[str] = None) -> Iterable[Word]:

        word_total = self.count_words(where_clause) if use_tqdm else None
        where_clause = '' if where_clause is None else where_clause

        def build_word(args: List) -> Word:
            #args = list(args)[1:] #TODO: uncomment when words have ids again
            return Word(*args)

        if include_sentences:
            word_cursor = self.cur.execute(f'''select words.*, sentences.sent from words
                                               join sentences on words.sentence = sentences.id ''' + where_clause)

            sent_idx = [description[0] for description in self.cur.description].index('sentence')

            for row in tqdm(word_cursor, disable=not use_tqdm, total=word_total, desc='reading words with sentences'):
                word_data = list(row[:len(word_attributes)])
                word_data[sent_idx] = row[-1]  # the last element is the sentence text, use that to replace sentence ID
                yield build_word(word_data)

        else:
            sql = f'SELECT * from words ' + where_clause
            print(sql)
            word_cursor = self.cur.execute(sql)
            for row in tqdm(word_cursor, disable=not use_tqdm, total=word_total, desc='reading words'):
                yield build_word(row)

    def save_sentences(self, sents: List[str]):
        self.cur.executemany(f'INSERT OR IGNORE INTO sentences (sent) VALUES (?)', ((x,) for x in sents))
        self.con.commit()

    def save_clusters(self, clusters: List[WordCluster]):
        self.cur.executemany(f'INSERT INTO clusters ({",".join(name for name, type_ in word_cluster_attributes)})'
                             f' values ({",".join("?" for x in word_cluster_attributes)})', clusters)
        self.con.commit()

    def save_words(self, words: List[Word]) -> None:
        self.cur.executemany(f'INSERT INTO words ({",".join(name for name, type_ in word_attributes)})'
                             f' values ({",".join("?" for x in word_attributes)})', words)
        self.con.commit()










