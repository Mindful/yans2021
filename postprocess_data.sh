sentences_db="$1_sentences.db"

if [ -f "$sentences_db" ]; then
    echo "found $sentences_db"
else
  echo "could not find $sentences_db"
  exit
fi

word_dbs="$1_words*.db"

for f in $word_dbs
do
  sql_string="attach database '$sentences_db' as sents;
attach database '$f' as worddb;
insert into words(form,lemma,pos,sentence,embedding,display_embedding)
select form, lemma, pos, sentence, embedding, display_embedding from worddb.words;"
echo "$sql_string"
time -p sqlite3 "$sentences_db" "$sql_string"
done

sql_string="create index lemma_index on words(lemma);"
echo "$sql_string"
time -p sqlite3 "$sentences_db" "$sql_string"
echo "done"