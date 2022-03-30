import string
from argparse import ArgumentParser
from pathlib import Path
import random
from nltk.corpus import words
from itertools import cycle


from constants import sent_separator


def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--preserve_target_word', action='store_true')
    args = parser.parse_args()

    infile_name_parts = args.input.name.split('.')
    outfile = Path(infile_name_parts[0] + '.scrambled.' + infile_name_parts[1])

    wordlist = words.words()
    random.shuffle(wordlist)
    word_iter = cycle(iter(wordlist))

    with args.input.open('r') as input_file:
        with outfile.open('w') as output_file:
            for line in input_file:
                try:
                    start_index = line.index(sent_separator)
                    original_sentence = line[:start_index]
                    target_word = original_sentence[original_sentence.index('<define>') + len('<define>'):original_sentence.index('</define>')].strip()

                    supplemental_data_words = line[start_index:].split()

                    scrambled_words = [
                        next(word_iter) if word[0] != '<' and word not in string.punctuation else word
                        for word in supplemental_data_words
                    ]

                    if args.preserve_target_word:
                        for idx, word in enumerate(supplemental_data_words):
                            if word.lower() == target_word.lower():
                                scrambled_words[idx] = word

                    result = original_sentence + ' '.join(scrambled_words) + '\n'
                    output_file.write(result)

                except ValueError:
                    output_file.write(line)


if __name__ == '__main__':
    main()