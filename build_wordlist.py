import argparse
import os
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    wordset = set()

    for filename in os.listdir(args.input):
        with open(os.path.join(args.input, filename), 'r') as wordfile:
            rows = list(csv.reader(wordfile, delimiter='\t'))
            row_elements = {e for row in rows for e in row}
            for elem in row_elements:
                try:
                    float(elem)
                except ValueError:
                    wordset.add(elem)

    with open(args.output, 'w') as outfile:
        for word in wordset:
            outfile.write(word+'\n')


if __name__ == '__main__':
    main()
