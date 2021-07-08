from tqdm import tqdm
from read_data import fast_linecount


class RawFileReader:

    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'r')

    def __iter__(self):
        for line in tqdm(self.file, desc=f'reading {self.filename}', total=fast_linecount(self.filename)):
            yield line.strip()

    def __del__(self):
        self.file.close()