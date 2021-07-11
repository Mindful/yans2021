from typing import Optional

from tqdm import tqdm
from read_data import fast_linecount


class RawFileReader:

    def __init__(self, filename: str, max_line_length: Optional[int]):
        self.filename = filename
        self.file = open(filename, 'r')
        self.max_len = max_line_length
        self.total_lines = fast_linecount(self.filename)

    def __iter__(self):
        for line in tqdm(self.file, desc=f'reading {self.filename}', total=self.total_lines):
            output_line = line.strip()
            if self.max_len is None or len(output_line) < self.max_len:
                yield output_line

    def __del__(self):
        self.file.close()
