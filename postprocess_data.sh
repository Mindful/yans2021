word_db="$1_words.db"
sentences_db="$1_sentences.db"


if [ -f "$word_db" ]; then
    echo "found $word_db"
else
  echo "could not find $word_db"
  exit
fi

if [ -f "$sentences_db" ]; then
    echo "found $sentences_db"
else
  echo "could not find $sentences_db"
  exit
fi

sql_string="$word_db
attach database '$sentences_db' as sents;
insert into sentences select * from sents.sentences;
create index lemma_index on words(lemma);
"
echo sql_string

time -p sqlite3 "$sql_string"
echo "done"