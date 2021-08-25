import pickle
import sqlite3
from collections import namedtuple
from typing import List, Iterable, Tuple, Callable, Optional
import logging

from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import io

word_attributes = [
    ('id', 'INTEGER PRIMARY KEY'),
    ('form', 'TEXT NOT NULL'),
    ('lemma', 'TEXT NOT NULL'),
    ('pos', 'INT NOT NULL'),
    ('sentence', 'INT NOT NULL'),
    ('embedding', 'ARRAY NOT NULL'),
    ('display_embedding', 'ARRAY')
]
Word = namedtuple('Word', [name for name, type_ in word_attributes])
WORD_TABLE_SCHEMA = '( ' + ', '.join(f'{name} {type_}' for name, type_ in word_attributes) + ')'

word_cluster_attributes = [
    ('id', 'INTEGER PRIMARY KEY'),
    ('lemma', 'TEXT NOT NULL'),
    ('pos', 'INT NOT NULL'),
    ('cluster_centers', 'ARRAY NOT NULL'),
    ('pca', 'PCA NOT NULL'),
    ('tree', 'TEXT NOT NULL'),
]

ClusterWord = namedtuple('CluserWord', [name for name, type_ in word_attributes]+['cluster_label'])


class WordCluster:
    def __init__(self, id: int, lemma: str, pos: int, cluster_centers: np.ndarray, pca: PCA, tree: str,
                 words: List[ClusterWord]):
        self.id = id
        self.lemma = lemma
        self.pos = pos
        self.cluster_centers = cluster_centers
        self.pca = pca
        self.tree = tree
        self.words = words

    @property
    def labels(self):
        return sorted(set(word.cluster_label for word in self.words))


CLUSTER_TABLE_SCHEMA = '( ' + ', '.join(f'{name} {type_}' for name, type_
                                        in
                                        word_cluster_attributes) + ', UNIQUE(lemma, pos, tree))'

JUNCTION_TABLE_SCHEMA = '(cluster_id INTEGER, word_id INTEGER, label INTEGER)'
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


def adapt_pca(pca: PCA) -> sqlite3.Binary:
    return sqlite3.Binary(pickle.dumps(pca, pickle.HIGHEST_PROTOCOL))


def convert_pca(data) -> PCA:
    return pickle.loads(data)


# Converts np.array to TEXT/BLOB when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT/BLOB to np.array when selecting
sqlite3.register_converter("ARRAY", convert_array)

sqlite3.register_adapter(PCA, adapt_pca)
sqlite3.register_converter("PCA", convert_pca)

logger = logging.getLogger()


class WriteBuffer:
    def __init__(self, name: str, save_function: Callable, buffer_size: int = 100000):
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
        con = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES, timeout=600)
        cur = con.cursor()
        self.con = con

        self.con.execute('PRAGMA synchronous = 0')
        self.con.execute('PRAGMA journal_mode = OFF')
        self.cur = cur
        self.cur.execute(f'CREATE TABLE IF NOT EXISTS words{WORD_TABLE_SCHEMA}')
        self.cur.execute(f'CREATE TABLE IF NOT EXISTS sentences{SENTENCE_TABLE_SCHEMA}')
        self.cur.execute(f'CREATE TABLE IF NOT EXISTS clusters{CLUSTER_TABLE_SCHEMA}')
        self.cur.execute(f'CREATE TABLE IF NOT EXISTS junction{JUNCTION_TABLE_SCHEMA}')
        self.cur.execute(f'CREATE INDEX IF NOT EXISTS junction_index on junction(cluster_id)')
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

    def save_cluster(self, cluster: WordCluster):
        self.cur.execute(f'INSERT OR REPLACE INTO clusters '
                         f'({",".join(name for name, _ in word_cluster_attributes if name != "id")})'
                         f' values ({",".join("?" for name, _ in word_cluster_attributes if name != "id")})',
                         tuple(val for key, val in cluster.__dict__.items() if key != "words" and key != "id"))

        junction_entries = [
            (self.cur.lastrowid, word.id, word.cluster_label) for word in cluster.words
        ]
        self.cur.executemany('INSERT INTO junction VALUES (?, ?, ?)', junction_entries)

        self.con.commit()

    def get_cluster(self, lemma: str, pos: int, tree: str) -> Optional[WordCluster]:
        # this comes from user input so we can't use string formatting without risking SQL injection
        # consequently, we can't use read_clusters or read_words
        cluster_cursor = self.cur.execute('SELECT * from clusters where lemma = ? and pos =? and tree =?',
                                          (lemma, pos, tree))
        try:
            cluster = WordCluster(*next(cluster_cursor), words=None)
        except StopIteration:
            return None

        word_cursor = self.cur.execute('''select words.id, words.form, words.lemma, words.pos, sentences.sent, 
        words.embedding, words.display_embedding, junction.label
        from junction 
        join words on junction.word_id = words.id 
        join sentences on words.sentence = sentences.id 
        where cluster_id=?''', (cluster.id,))

        cluster.words = [ClusterWord(*x) for x in word_cursor]

        return cluster

    def read_words(self, use_tqdm: bool = False, where_clause: Optional[str] = None) -> Iterable[Word]:

        word_total = self.count_words(where_clause) if use_tqdm else None
        where_clause = '' if where_clause is None else where_clause

        sql = f'SELECT * from words ' + where_clause
        word_cursor = self.cur.execute(sql)
        for row in tqdm(word_cursor, disable=not use_tqdm, total=word_total, desc='reading words'):
            yield Word(*row)

    def save_sentences(self, sents: List[str]):
        self.cur.executemany(f'INSERT OR IGNORE INTO sentences (sent) VALUES (?)', ((x,) for x in sents))
        self.con.commit()

    def save_words(self, words: List[Word]) -> None:
        self.cur.executemany(f'INSERT INTO words ({",".join(name for name, type_ in word_attributes if name != "id")})'
                             f' values ({",".join("?" for _ in word_attributes)})', words)
        self.con.commit()

    def add_display_embedding_to_words(self, display_embedding_data: List[Tuple[int, np.ndarray]]):
        sql_data = [(embed, word_id) for word_id, embed in display_embedding_data]
        self.cur.executemany('UPDATE words SET display_embedding = ? WHERE id = ?', sql_data)
        self.con.commit()
