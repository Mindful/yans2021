from itertools import takewhile, repeat

from jsonlines import jsonlines
from tqdm import tqdm


# https://stackoverflow.com/a/27518377/4243650
def fast_linecount(filename) -> int:
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum(buf.count(b'\n') for buf in bufgen)


class RawFileReader:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'r')
        self.total_lines = fast_linecount(self.filename)

    def __iter__(self):
        for line in tqdm(self.file, desc=f'reading {self.filename}', total=self.total_lines):
            yield line.strip()

    def __del__(self):
        self.file.close()


class JsonFileReader:
    def __init__(self, filename: str, text_key: str = 'text'):
        self.filename = filename
        self.file = jsonlines.open(filename, 'r')
        self.total_lines = fast_linecount(self.filename)
        self.text_key = text_key

    def __iter__(self):
        for line in tqdm(self.file, desc=f'reading {self.filename}', total=self.total_lines):
            if self.text_key in line:
                yield line[self.text_key].strip()
