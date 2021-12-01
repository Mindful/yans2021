from argparse import ArgumentParser

from data.db import DbConnection


def main():
    parser = ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()

    db = DbConnection(args.run+'_examples')

    print('create split index')
    db.con.execute('create index if not exists split_index on examples(split);')
    print('done')


if __name__ == '__main__':
    main()