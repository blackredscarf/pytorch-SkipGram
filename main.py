import random
import torch
from torch.optim import SGD
from data import read_fromfile, DataPipeline
from model import SkipGramNeg
from utils import get_output_folder
from vector_handle import nearest

print("read data...")
data, count, dictionary, reverse_dictionary = read_fromfile()
print("corpus size", len(data))

vocabulary_size = 50000
batch_size = 128
learning_rate = 1.0
embedding_size = 300
skip_window = 1
num_skips = 2
num_neg = 20
num_steps = 200000
vali_size = 3
data_offest = 0
avg_loss = 0
outputdir = get_output_folder('out', 'sgd')

model = SkipGramNeg(vocabulary_size, embedding_size).cuda()
model_optim = SGD(model.parameters(), lr=learning_rate)

# retrain
# model.load_state_dict(torch.load("out/sgd-run2/model_step300000"))
# data_offest = 600000

pipeline = DataPipeline(data, count, data_offest)
vali_examples = random.sample(data, vali_size)

print("train start...")
for step in range(num_steps):
    batch_inputs, batch_labels = pipeline.generate_batch(batch_size, num_skips, skip_window)
    batch_neg = pipeline.get_neg_data(batch_size, num_neg, batch_inputs)

    batch_inputs = torch.tensor(batch_inputs, dtype=torch.long).cuda()
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
    batch_neg = torch.tensor(batch_neg, dtype=torch.long).cuda()

    loss = model(batch_inputs, batch_labels, batch_neg)
    model_optim.zero_grad()
    loss.backward()
    model_optim.step()

    avg_loss += loss.item()

    if step % 2000 == 0 and step > 0:
        avg_loss /= 2000
        print('Average loss at step ', step, ': ', avg_loss)
        average_loss = 0

    if step % 10000 == 0:
        nearest(model, vali_examples, vali_size, reverse_dictionary, top_k=8)

    if step % 100000 == 0 and step > 0:
        torch.save(model.state_dict(), outputdir + '/model_step%d' % step)

torch.save(model.state_dict(), outputdir + '/model_step%d' % num_steps)


