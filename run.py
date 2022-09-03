from torchtext.datasets import Multi30k
from torchtext.data import Field,BucketIterator
import torch.optim as optim
from model import Decoder, Encoder, Seq2Seq
import torch.nn as nn
import spacy
import torch

from utils import load_checkpoint, save_data, translate_sentence

if __name__ == '__main__':
    spacy_ger = spacy.load("de_core_news_sm")
    spacy_eng = spacy.load('en_core_web_sm')

    # 'hello this is me' -> ['hello','this','is','me']
    def tokenizer_eng(text):
        return [tok.text for tok in spacy_eng.tokenizer(text)]
    def tokenizer_ger(text):
            return [tok.text for tok in spacy_ger.tokenizer(text)]


    # Field used to define how the preprocessing is done
    german = Field(tokenize=tokenizer_ger,lower=True,init_token= '<sos>',eos_token='<eos>')
    english = Field(tokenize=tokenizer_eng,lower=True,init_token= '<sos>',eos_token='<eos>')

    #building vocab
    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(german, english)
    )
    german.build_vocab(train_data, max_size=10000, min_freq=2)
    english.build_vocab(train_data, max_size=10000, min_freq=2)
    # Traininig hyperparameters
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 64
    # model hyperparamters
    load_model = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size_encoder = len(german.vocab)
    input_size_decoder = len(english.vocab)
    output_size = len(english.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    train_iterator,valid_iterator,test_iterator = BucketIterator.splits(
        (train_data,valid_data,test_data),
        batch_size=batch_size,
        sort_within_batch =True,
        sort_key = lambda x:len(x.src),
        device=device
    )
    encoder_net = Encoder(input_size_encoder,encoder_embedding_size,hidden_size,num_layers,enc_dropout).to(device)
    decoder_net = Decoder(input_size_decoder,decoder_embedding_size,hidden_size,output_size,num_layers,dec_dropout).to(device)
    model = Seq2Seq(encoder_net,decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = english.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)        # to ignore padding in loss
    load_checkpoint(torch.load('models/0.pth.tar'),model,optimizer)

    # Two people go to the store to buy very cold ice cream
    sen = 'Zwei Leute gehen zum Laden, um sehr kaltes Eis zu kaufen'
    translated_sentence = translate_sentence(model,sen,german,english,device,max_length=50)
    print(translated_sentence)
    save_data(5,translated_sentence)