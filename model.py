import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,p):
        super(Encoder,self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,bidirectional=True)

        # 2 hidden size,cell state of bidrectional network(forward & backward) converted to 1 hidden size using nn
        self.fc_hidden = nn.Linear(hidden_size*2,hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2,hidden_size)

    def forward(self,x):
        # x shape : (seq_length,N)    where N is batchsize and seq_length is the number of words
        embedding = self.dropout(self.embedding(x))
        # embeddin shape: (seq_length,N,embedding_size) where embedding_size is the total len of embedding array

        # passing hidden and cell state of encoder to decoder
        encoder_states,(hidden,cell) = self.rnn(embedding)

        hidden_forward = hidden[0:1]
        hidden_backward = hidden[1:2]
        hidden = self.fc_hidden(torch.cat((hidden_forward,hidden_backward),dim=2))

        cell_forward = cell[0:1]
        cell_backward = cell[1:2]
        cell = self.fc_cell(torch.cat((cell_forward,cell_backward),dim=2))

        # we dont need the ouput of the encoder
        return encoder_states,hidden,cell

class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,p):
        """
        input_size=ouput_size -  size of the english embedded vocabulary
        hidden_size of encode and decoder are same
        embedding_size - is converting embedding to a size
        p - dropout prob
        """
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU() 


        # input for the lstm will be encoder_states(hidden_size*2) from encoder + embeding_size or word
        lstm_input_size = hidden_size*2+embedding_size
        self.rnn = nn.LSTM(lstm_input_size,hidden_size,num_layers)

        # input = encoder_hiddent(hiden_size*2) + decoder_hidden
        self.energy = nn.Linear(hidden_size*3,1)

        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x,encoder_states,hidden,cell):
        """
        shape of x is : (N) but we want (1,N) signifying we have n batches of single word at a time for decoder
                        in encoder we have len of seq but while predicting we only have 1 previous word as input
        """
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embeddin shape: (1,N,embedding_size) 
        
        # repeat our decoder hidden state till we match the encoder hidden states for concatining them to find energy
        # hidden - [x,y,z] -> [[x,y,z],[x,y,z]]
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length,1,1)

        # adding encoder states with repeated decoder hidden state
        attention_hidden_state = torch.cat((h_reshaped,encoder_states),dim=2)
        energy = self.relu(self.energy(attention_hidden_state))
        attention = self.softmax(energy)  # shape - (seq_length,N,1)

        # Encoder sates : 3d tensor
        # torch.dmm to multiply 3d tensors
        attention = attention.permute(1,2,0)  # (seq,N,1) to (n,1,seq)
        encoder_states = encoder_states.permute(1,0,2)  # (seq,n,hidden*2) to (n,seq,hidden*2)
        context_vector = torch.bmm(attention,encoder_states).permute(1,0,2) # (n,1,hidden*2) to (1,n,hidden*2)

        rnn_input = torch.cat((context_vector,embedding),dim=2) # adding context to embeddings for predictions

        outputs , (hidden,cell) = self.rnn(rnn_input,(hidden,cell))
        # shape of ouuput = (1,N,hidden_size)
        
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        # shape of predictions: (1,N,lenght_of_vocab) --> (N,lenght_of_vocab)
        # so that its easier to add all the docoder output together 
      
        return predictions,hidden,cell

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,source,target,target_vocab_size,device,teacher_force_ratio=0.5,):
        """
        source  - (target_len,N) 
        target - correct translation
        teacher_force_ratio : while training second or later words in decoder ,  we input from the last cell word 50% time and the 
                              correct input 50% of time so that it can learn to predict next word optimally not 
                              worring of following in incorrect loop due to wrong predictions by the last cell
        """
        batch_size = source.shape[1]
        target_len = target.shape[0]
        outputs = torch.zeros(target_len,batch_size,target_vocab_size).to(device)
        encoder_states,hidden,cell = self.encoder(source)

        # Grab start token
        x = target[0]

        # running decoder repeatedly to get the required sequence
        for t in range(1,target_len):
            output,hidden,cell = self.decoder(x,encoder_states,hidden,cell)
            # output - (N,taget_vocab_size) where N is batch_size and taget_vocab_size is the ouput len of array
            outputs[t] = output
            best_guess = output.argmax(1)   # 1 to get best value of taget_vocab_size or the text index
            
            # using teacher force to either give it predicted last value or the real last word value in 50% chance
            # target[t] is a true value whereas best_guess is the predicted value
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

