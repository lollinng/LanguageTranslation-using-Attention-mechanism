# Language Translation using Attention mechanism

Bidirectional LSTM used with attention layers to translate German sentences to English sentences

# Contents

***[Objective](https://github.com/lollinng/LanguageTranslation-using-Attention-mechanism#objective)***

***[Concepts](https://github.com/lollinng/LanguageTranslation-using-Attention-mechanism#concepts)***

***[Overview](https://github.com/lollinng/LanguageTranslation-using-Attention-mechanism#overview)***

***[Usage](https://github.com/lollinng/LanguageTranslation-using-Attention-mechanism#Usage)***

# Objective

**Learn implementation of attention layers on seqtoseq model in language translation problem(converting German to English)**

We will be implementing the *[Attention on seq to seq model](https://arxiv.org/abs/1409.0473)* paper. This is by no means the current state-of-the-art,
but it gave a breakthrough and an idea to develop further paper like *[Attention is all you need](https://arxiv.org/abs/1706.03762)* and other transformer
based implementation for language translation.

The model takes a germen sentences and convert it into an embedding as input for the encoder , the encoder consisting of bidirectional lstm outputs the cell, hidden and encoder states .

The hidden and encoded states helps in creating a context vector using attention weights which when inputted to the
lstm outputs an English word for every iteration.

---

# Concepts

- **Encoder-Decoder architecture**. This type of architecture was first introduced in paper *[Seq to Seq](https://arxiv.org/abs/1409.3215)* . This arrangement of blocks helps us to get context vector from input and convert context vector into a sequence of output depending upon your business logic
- **LSTM**. LSTM or Long Short Term Memory is a type of rnn which is used to generate sequence of outputs by taking either sequence input or a vector value . *[LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)* was first introduced in 1997 and is still considered on of the best model for sequential data.
- **Attention**. Attention is the phenomenon in which we create a context vector by referring to the different words from the encoded_states and decide how much imp./attention to give to each word context from encoded_states . The context vector is later added to prev word output of decoder to provide rich input to decoder LSTM.

# Overview

In this section, I will present an overview of this model. If you're already familiar with it, you can skip straight to the [Usage](https://github.com/lollinng/LanguageTranslation-using-Attention-mechanism/#Usage) section.

### Encoder

The Encoder's task is to **Convert the Sequence of German words into encoded states to allow decoder to create English words through it**.

The Encoder consists of bidirectional LSTM which gets German word_embeddings as input ( created using vocabulary and a dropout layer) . I prefer Biderectional LSTM over normal LSTM to better understand the context of word and its position in these sentences.

bidirectional LSTM's outputs - `hidden_state and cell_state` which are sent through neural networks to create only one hidden_state and cell_state instead of two created by bidirectional lstm.

The `encoded states` are also created by the bi directional lstm which helps us in creating attention weights as well as well as influencing input to the decoder’s lstm.

![https://user-images.githubusercontent.com/55660103/188798998-cca48642-511c-4cf7-bbd9-9e89f57b27c8.png](https://user-images.githubusercontent.com/55660103/188798998-cca48642-511c-4cf7-bbd9-9e89f57b27c8.png)

### Decoder

The Decoder's task is to **Convert the  encoded states from encoder to an English translated sentence**.

### Attention - Attention weights are created by inputting prev. hidden_state and encoder’s encoded_state to a neural network.

***This attention weights multiplied by encoded states help us choose or give attention/importance to certain features in encoded states creating a context vector***

The context vector added with the prev. word embedding help us to create a newly translated word for current index using the lstm.

The lstm then updates the prev. word embeddings , hidden_state and cell_state but the encoded_state remains same for the next iteration .

And hence whole process repeat itself until we get EOS token signaling end of the sentence.
  
# Usage

### 1. Clone the repositories

```
git clone <https://github.com/lollinng/LanguageTranslation-using-Attention-mechanism.git>
cd LanguageTranslation-using-Attention-mechanism

```

### 2. Dataset

```
# After you run train.py python file will download the 'Multi30k' dataset you dont have worry

```

### 3. Hyperparamters

```
# check for hyper parameters at train.py
# check load_model paramter if u want to run a checkpoint

```

### 4. Train the model

```
python train.py

```

### 5. Test the model

```
python run.py

```
