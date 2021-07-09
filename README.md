# Examplify

```
python embed_words.py --input ~/data/few_eng_sentences.txt --output embed_data.pkl
python cluster_words.py --input embed_data.pkl --output cluster_data.pkl
python search_embeddings.py --data cluster_data.pkl --input "Don't you like [chicken]?"
```