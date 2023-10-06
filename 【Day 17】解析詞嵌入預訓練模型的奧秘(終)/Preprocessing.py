from collections import Counter
from torchtext.vocab import vocab
import gensim.downloader as api
import torch

def load_data(file_path):
    sentences, labels = [], []
    sentence, label = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                if sentence and label:
                    sentences.append(sentence)
                    labels.append(label)
                sentence, label = [], []
            else:
                parts = line.strip().split()

                sentence.append(parts[0])
                label.append(parts[-1])
    
    return sentences, labels
    
def torchText(all_sentences, all_labels, specials = ('<PAD>', '<UNK>')):
    token_counter, label_counter = Counter(), Counter()
    for sentence, labels in zip(all_sentences, all_labels):
        token_counter.update(sentence)
        label_counter.update(labels)
        
    token_voc = vocab(token_counter, specials=specials)
    token_voc.set_default_index(token_voc.get_stoi()['<UNK>'])
    
    label_voc = vocab(label_counter)
    
    return token_voc, label_voc
    
def tokens2nums(sentences, labels, token_voc, label_voc):

    token_nums, label_nums = [], []
    for word, label in zip(sentences, labels):
        token_num = token_voc.lookup_indices(word)
        label_num = label_voc.lookup_indices(label)

        token_nums.append(torch.tensor(token_num))
        label_nums.append(torch.tensor(label_num))

    return token_nums, label_nums
    
    
def pre_trained_model(model_name, all_sentences, all_labels, specials = ('<PAD>', '<UNK>')):
    model = api.load(model_name)
    token_voc, label_voc = torchText(all_sentences, all_labels, specials = specials)
    unk_idx = token_voc.get_stoi()['<UNK>']
    pretrained_voc, word2vec_voc = [], {}
    for word in token_voc.get_stoi():
        idx = model.key_to_index.get(word, unk_idx)
        if idx != 1:
            pretrained_voc.append(model[idx])
            word2vec_voc.update({word:1})
            
   
    word2vec_voc = vocab(word2vec_voc, specials=specials)
    word2vec_voc.set_default_index(word2vec_voc.get_stoi()['<UNK>'])
    
    pretrained_emb = torch.tensor(pretrained_voc)
    pretrained_emb = torch.cat((torch.zeros(len(specials), pretrained_emb.shape[1]), pretrained_emb))
    
    return word2vec_voc, label_voc, pretrained_emb
    