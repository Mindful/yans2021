from argparse import ArgumentParser

from data.db import DbConnection


def main():
    parser = ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()

    db = DbConnection(args.run+'_words')

    print('create lemma index')
    db.con.execute('create index if not exists lemma_index on words(lemma);')
    print('create form index')
    db.con.execute('create index if not exists form_index on words(form);')
    print('done')


if __name__ == '__main__':
    main()