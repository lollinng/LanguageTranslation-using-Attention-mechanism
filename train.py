from model import Decoder, Encoder, Seq2Seq
from utils import load_checkpoint, save_data, translate_sentence,blue,save_checkpoint


import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field,BucketIterator
 
from torch.utils.tensorboard import SummaryWriter

def train():

    # Building vocab
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




    # Training

    # Traininig hyperparameters
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 32
    # model hyperparamters
    load_model = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size_encoder = len(german.vocab)
    input_size_decoder = len(english.vocab)
    output_size = len(english.vocab)
    target_vocab_size = len(english.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 1
    enc_dropout = 0.5
    dec_dropout = 0.5
    # Tensorboard
    writer = SummaryWriter(f'runs/loss_plot')
    step = 0

    # here we using iterator to sort the acc to lenof input words so that a batch_size can have minimum diff in word_len 
    # which will result in less use of <pad> seq word hence saving our computation time
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

    if load_model:
        load_checkpoint(torch.load('models/my_checkpoint.pth.tar'),model,optimizer)

    # Two people go to the store to buy very cold ice cream
    sen = 'Zwei Leute gehen zum Laden, um sehr kaltes Eis zu kaufen'

    # 29 done
    for epoch in range(29,41):
        print(f'Epoch [{epoch}/{num_epochs}]')
        checkpoint = {'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)

        # test our transaltion over epochs
        if epoch%5==0:
            save_checkpoint(checkpoint,filename=f'models/{epoch}.pth.tar')
            model.eval()
            translated_sentence = translate_sentence(model,sen,german,english,device,max_length=50)
            save_data(epoch,translated_sentence)
            model.train()

        # src - source
        for batch_idx,batch in enumerate(train_iterator):
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)
        
            output = model(inp_data,target,target_vocab_size,device)
            # output shape: (trg_len,batch_size,output_dim)
            # for loss function we convert output to (trg_len*batch_size,output_dim) 
                # output[1:] skips 1st index of trg_len coz of <sos> and makes it (trg_len-1,batch_size,output_dim)
                # .reshape(-1,output.shape[2]) makes it (trg_len*batch_size,output_dim) 
            # for loss function we convert target to ()
            output = output[1:].reshape(-1,output.shape[2])
            target = target[1:].reshape(-1)


            optimizer.zero_grad()
            loss = criterion(output,target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)    # clipping lstm gradiants to asve from graidna exploding
            optimizer.step() # update optimizer values

            #ternsorboard
            writer.add_scalar('Training Loss',loss,global_step=step)
            step+=1


    score = blue(test_data,model,german,english,device)
    print(f'Blue score {score*100:.2f}')        

if __name__ == '__main__':
    train()