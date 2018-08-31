from word2vec import Word2Vec

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
# word2vec.load_model('out/run-1/model_step200000.pt')
