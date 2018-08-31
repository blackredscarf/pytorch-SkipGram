Using pytorch to implement word2vec algorithm Skip-gram Negative Sampling (SGNS), and refer paper [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546v1).

## Dependency
- python 3.6
- pytorch 0.4+

## Usage
Run `main.py`.

Initialize the dataset and model.

```python
# init dataset and model
word2vec = Word2Vec(data_path='text8',
                    vocabulary_size=50000,
                    embedding_size=300)

# the index of the whole corpus
print(word2vec.data[:10])

# word_count like this [['word', word_count], ...]
# the index of list correspond index of word
print(word2vec.word_count[:10])

# index to word
print(word2vec.index2word[34])

# word to index
print(word2vec.word2index['hello'])
```


Train and get the vector.

```python
# train model
word2vec.train(train_steps=200000,
               skip_window=1,
               num_skips=2,
               num_neg=20,
               output_dir='out/run-1')

# save vector txt file to output_dir
word2vec.save_vector_txt(path_dir='out/run-1')

# get vector list
vector = word2vec.get_list_vector()
print(vector[123])
print(vector[word2vec.word2index['hello']])

# get top k similar word
sim_list = word2vec.most_similar('one', top_k=8)
print(sim_list)

# load pre-train model
word2vec.load_model('out/run-1/model_step200000.pt')
```


## Evaluate
Refer repository [eval-word-vectors](https://github.com/mfaruqui/eval-word-vectors).
Like this:
```
eval/wordsim.py vector.txt eval/data/EN-MTurk-287.txt
```
```
eval/wordsim.py vector.txt eval/data/EN-MC-30.txt
```






