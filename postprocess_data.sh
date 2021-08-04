sentences_db="$1_sentences.db"

if [ -f "$sentences_db" ]; then
    echo "found $sentences_db"
else
  echo "could not find $sentences_db"
  exit
fi

word_dbs="$1_words*.db"


sql_string="attach database '$sentences_db' as sents;"

word_db_counter=1
for f in $word_dbs
do
  echo "Found $f"
  sql_string+="
attach database '$f' as words$word_db_counter;
insert into words(form,lemma,pos,sentence,embedding) select form, lemma, pos, sentence, embedding from words$word_db_counter.words;"
  word_db_counter=$((word_db_counter+1))
done

sql_string+="
create index lemma_index on words(lemma);
"
echo "$sql_string"

time -p sqlite3 "$sentences_db" "$sql_string"
echo "done"