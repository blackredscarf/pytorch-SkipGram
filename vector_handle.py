import torch


def model_to_vector(model, emb_layer_name='input_emb'):
    """
    get the wordvec weight
    :param model:
    :param emb_layer_name:
    :return:
    """
    sd = model.state_dict()
    return sd[emb_layer_name + '.weight'].cpu().numpy().tolist()

def save_embedding(file_name, embeddings, id2word):
    """
    wordvec save to text file
    :param file_name:
    :param embeddings:
    :param id2word:
    :return:
    """
    fo = open(file_name, 'w')
    for idx in range(len(embeddings)):
        word = id2word[idx]
        embed = embeddings[idx]
        embed_list = [str(i) for i in embed]
        line_str = ' '.join(embed_list)
        fo.write(word + ' ' + line_str + '\n')

    fo.close()

def nearest(model, vali_examples, vali_size, id2word_dict, top_k=8):
    """
    find the nearest word of vali_examples
    :param model: model
    :param vali_examples: []
    :param vali_size: int
    :param id2word_dict: {}
    :param top_k: int
    :return:
    """
    vali_examples = torch.tensor(vali_examples, dtype=torch.long).cuda()
    vali_emb = model.predict(vali_examples)
    # sim: [batch_size, vocab_size]
    sim = torch.mm(vali_emb, model.input_emb.weight.transpose(0, 1))
    for i in range(vali_size):
        vali_word = id2word_dict[vali_examples[i].item()]
        nearest = (-sim[i, :]).sort()[1][1: top_k + 1]
        log_str = 'Nearest to %s:' % vali_word
        for k in range(top_k):
            close_word = id2word_dict[nearest[k].item()]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)