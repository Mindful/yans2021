# 文脈化埋め込みを用いた言語学習者のための語義別例文検索システム

NLP若手の回（YANS)で発表させていただいた研究です。

[![Poster image](poster.jpg)](poster.pdf)

## データのダウンロード
```shell
wget https://dumps.wikimedia.org/other/cirrussearch/current/enwiki-20210802-cirrussearch-content.json.gz
```


## 事前処理
```shell
python ingest_sentences.py --input wiki --run yans
python embed_words.py --run yans
./postprocess_data.sh yans
```

## ウェブUI
```shell
RUN=yans uvicorn web:app
```


# For definition generation

```shell
CUDA_VISIBLE_DEVICES="2" python ingest_examples.py --contexts ../chang_seen_valid_head.txt --targets ../generationary_emnlp/data/corpora/orig/chang.definitions.txt
python ingest_sentences.py --input ../eng_corpus_data/enwiki_head.json --example_db default_run_examples
python embed_words.py --run default_run --example_db default_run_examples
python generationary_paired_data_creator.py --run_name default_run --output ../test
```

Full run
```shell
python ingest_examples.py --contexts ../generationary_emnlp/data/corpora/orig/chang_seen_train.contexts.txt --targets ../generationary_emnlp/data/corpora/orig/chang.definitions.txt --run cs --split train
python ingest_examples.py --contexts ../generationary_emnlp/data/corpora/orig/chang_seen_valid.contexts.txt --targets ../generationary_emnlp/data/corpora/orig/chang.definitions.txt --run cs --split eval
python ingest_examples.py --contexts ../generationary_emnlp/data/corpora/orig/chang_seen_test.contexts.txt  --targets ../generationary_emnlp/data/corpora/orig/chang.definitions.txt --run cs --split test
python index_examples.py --run cs
python ingest_sentences.py --input ../eng_corpus_data/enwiki-20210726-cirrussearch-content.json --example_db cs_examples --run cs #stopped after 300k articles
python embed_words.py --run cs --example_db cs_examples
python index_words.py --run cs
python generationary_paired_data_creator.py --run_name cs --output cs_train --split train
python generationary_paired_data_creator.py --run_name cs --output cs_eval --split eval
python generationary_paired_data_creator.py --run_name cs --output cs_test--split test
```

Speed up data generation:
```shell
python generationary_paired_data_creator.py --run_name cs --output cs_18 --fraction 1/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_28 --fraction 2/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_38 --fraction 3/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_48 --fraction 4/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_58 --fraction 5/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_68 --fraction 6/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_78 --fraction 7/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_88 --fraction 8/8 --split eval &
```


Scipy memory leak: https://github.com/scipy/scipy/issues/14382