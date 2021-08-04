# Examplify

```shell
python ingest_sentences.py --input wiki --run fw
python embed_words.py --run fw
./postprocess_data.sh fw
```

```shell
python cluster_by_key.py --run fw --key play
python search_embeddings.py --run fw --input "I like to [play] the violin."
```