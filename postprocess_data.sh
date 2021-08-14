sentences_db="$1_sentences.db"
words_db="$1_words.db"

if [ -f "$sentences_db" ]; then
    echo "found $sentences_db"
else
  echo "could not find $sentences_db"
  exit
fi

if [ -f "$words_db" ]; then
    echo "found $words_db"
else
  echo "could not find $words_db"
  exit
fi

new_db="$1.db"
echo "Copying words to $new_db"
cp "$words_db" "$new_db"

sql_string="attach database '$sentences_db' as sents;
insert into sentences select * from sents.sentences;"
echo "$sql_string"
time -p sqlite3 "$new_db" "$sql_string"

sql_string="create index lemma_index on words(lemma);"
echo "$sql_string"
time -p sqlite3 "$new_db" "$sql_string"