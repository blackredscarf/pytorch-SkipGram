Using pytorch to implement word2vec algorithm Skip-gram, and refer paper [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546v1).

## dependency
- pytorch 0.4+
- tensorflow

## prepare data 
run the command:
```
mkdir data
python data.py
```
Default use the [text8.zip](http://mattmahoney.net/dc/text8.zip). The script will serialize the data to the local.

If you want to use your own data, use function `read_own_data(filename)` in `data.py`

## train
```
mkdir out
python main.py
```
the model and `vector.txt` will save in the `out` folder


## evaluate
Refer repository [eval-word-vectors](https://github.com/mfaruqui/eval-word-vectors).
Like this:
```
eval/wordsim.py vector.txt eval/data/EN-MTurk-287.txt
```
```
eval/wordsim.py vector.txt eval/data/EN-MC-30.txt
```






