from data.db import DbConnection

d = DbConnection('default')
d.connect_for_reading()

c = list(d.read_sentences())

a = list(d.read_words())
print('debuggy')


b = list(d.read_words(include_sentences=True))
print('debuggy')
d.done()