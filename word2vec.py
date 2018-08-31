import os
import random
import torch
from torch.optim import SGD
from data_utils import read_own_data, build_dataset, DataPipeline
from model import SkipGramNeg
from vector_handle import nearest

class Word2Vec:
    def __init__(self, data_path, vocabulary_size, embedding_size, learning_rate=1.0):

        self.corpus = read_own_data(data_path)

        self.data, self.word_count, self.word2index, self.index2word = build_dataset(self.corpus,
                                                                    vocabulary_size)
        self.vocabs = list(set(self.data))

        self.model: SkipGramNeg = SkipGramNeg(vocabulary_size, embedding_size).cuda()
        self.model_optim = SGD(self.model.parameters(), lr=learning_rate)


    def train(self, train_steps, skip_window=1, num_skips=2, num_neg=20, batch_size=128, data_offest=0, vali_size=3, output_dir='out'):
        self.outputdir = os.mkdir(output_dir)

        avg_loss = 0
        pipeline = DataPipeline(self.data, self.vocabs ,self.word_count, data_offest)
        vali_examples = random.sample(self.vocabs, vali_size)

        for step in range(train_steps):
            batch_inputs, batch_labels = pipeline.generate_batch(batch_size, num_skips, skip_window)
            batch_neg = pipeline.get_neg_data(batch_size, num_neg, batch_inputs)

            batch_inputs = torch.tensor(batch_inputs, dtype=torch.long).cuda()
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
            batch_neg = torch.tensor(batch_neg, dtype=torch.long).cuda()

            loss = self.model(batch_inputs, batch_labels, batch_neg)
            self.model_optim.zero_grad()
            loss.backward()
            self.model_optim.step()

            avg_loss += loss.item()

            if step % 2000 == 0 and step > 0:
                avg_loss /= 2000
                print('Average loss at step ', step, ': ', avg_loss)
                avg_loss = 0

            if step % 10000 == 0 and vali_size > 0:
                nearest(self.model, vali_examples, vali_size, self.index2word, top_k=8)

            # checkpoint
            if step % 100000 == 0 and step > 0:
                torch.save(self.model.state_dict(), self.outputdir + '/model_step%d.pt' % step)

        # save model at last
        torch.save(self.model.state_dict(), self.outputdir + '/model_step%d.pt' % train_steps)

    def save_model(self, out_path):
        torch.save(self.model.state_dict(), out_path + '/model.pt')

    def get_list_vector(self):
        sd = self.model.state_dict()
        return sd['input_emb.weight'].tolist()

    def save_vector_txt(self, path_dir):
        embeddings = self.get_list_vector()
        fo = open(path_dir + '/vector.txt', 'w')
        for idx in range(len(embeddings)):
            word = self.index2word[idx]
            embed = embeddings[idx]
            embed_list = [str(i) for i in embed]
            line_str = ' '.join(embed_list)
            fo.write(word + ' ' + line_str + '\n')
        fo.close()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def vector(self, index):
        self.model.predict(index)

    def most_similar(self, word, top_k=8):
        index = self.word2index[word]
        index = torch.tensor(index, dtype=torch.long).cuda().unsqueeze(0)
        emb = self.model.predict(index)
        sim = torch.mm(emb, self.model.input_emb.weight.transpose(0, 1))
        nearest = (-sim[0]).sort()[1][1: top_k + 1]
        top_list = []
        for k in range(top_k):
            close_word = self.index2word[nearest[k].item()]
            top_list.append(close_word)
        return top_list


