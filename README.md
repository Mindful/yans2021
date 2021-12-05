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

cat cs_eval*.raw.en > cs_eval.raw.en
cat cs_eval*.raw.gloss > cs_eval.raw.gloss
cat cs_train*.raw.en > cs_train.raw.en
cat cs_train*.raw.gloss > cs_train.raw.gloss

# from updated generationary scripts
./preproc-supplemented.sh ../../cs_suppdata_1/cs_train
./preproc-supplemented.sh ../../cs_suppdata_1/cs_eval


./softlink.sh train cs_train
./softlink.sh valid cs_eval

conda activate generationary
./bart-run-kanagawa.sh CHA_S_SUPP


python generate_glosses_from_src_trg.py \
    --src ../data/corpora/preprocessed/cs_test.raw.en \
    --trg ../data/corpora/preprocessed/cs_test.raw.gloss \
    --checkpoint ../data/experiments/CHA_S_SUPP/checkpoint_best.pt \
    --beam 10 --min-len 6 --len-penalty 3.0 \
    --rerank \
    --generated-out ../data/CHANG_S_SUPP.pred.txt \
    --gold-out ../data/CHANG_S_SUPP.gold.txt \
```

Speed up data generation:
```shell
python generationary_paired_data_creator.py --run_name cs --output cs_eval_18 --fraction 1/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_eval_28 --fraction 2/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_eval_38 --fraction 3/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_eval_48 --fraction 4/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_eval_58 --fraction 5/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_eval_68 --fraction 6/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_eval_78 --fraction 7/8 --split eval &
python generationary_paired_data_creator.py --run_name cs --output cs_eval_88 --fraction 8/8 --split eval &
```


Scipy memory leak: https://github.com/scipy/scipy/issues/14382