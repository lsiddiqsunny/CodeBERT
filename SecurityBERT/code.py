# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import random


# %%
dataFile = pd.read_csv("./SecurityBERT/benchmark_261_I_Dataset.csv")
dataFile = dataFile.sample(frac=1).reset_index(drop=True)
print(dataFile.head())


# %%
trueValues = dataFile['isTrueVulnerable'].tolist()


# %%
import torch
from transformers import AutoTokenizer, AutoModel


# %%
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")


# %%
keywords = [
            "MD2",
            "MD4",
            "MD5",
            "RIPEMD",
            "RIPEMD",
            "sha-1",
            "sha1",
            "whirlpool",
            "base64",
            "b64",
            "btoa",
            "atob",
            "b2a_hex",
            "a2b_hex"
        ]
foundInFile = False


# %%
def TextAnalyzer(filepath):
    global keywords
    fileInInspection = open(filepath, 'r', encoding='utf-8-sig',errors='ignore')
    lines = fileInInspection.readlines()
    fileInInspection.close()
    vulnerableLines = ''
    for line in lines:
        if line.strip().startswith("#") or line.strip().startswith("//") or line.strip().startswith("/*")  :
            continue
        if len(line.strip())==0:
            continue
        # print(line)
        for keyword in keywords:
            if keyword.lower() in line.lower():
                vulnerableLines+=line
                
    fileInInspection.close()
    return vulnerableLines


# %%
tokenList = []
chunksize = 512
vocab_size = 0
for ind in dataFile.index:
    filename = dataFile['FileName'][ind]
    filenamePart = filename.split('/')
    filename = './../IoT-Security-Improper-Authentication/' +  '/'.join([str(elem) for elem in filenamePart[1:]])
    vulenrableCode = TextAnalyzer(filename)
    code_tokens=tokenizer.tokenize(vulenrableCode)
    tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    # print(tokens)
    tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
    if (len(tokens_ids)>512):
        tokens_ids = [tokens_ids[0]] + random.sample(tokens_ids[1:len(tokens_ids)-1], 510) +[tokens_ids[len(tokens_ids)-1]]
    # get required padding length
    pad_len = chunksize - len(tokens_ids)
    if pad_len > 0:
        tokens_ids = tokens_ids + ([0] * pad_len)
    vocab_size = max(tokens_ids)
    # context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
    # print(context_embeddings)
    tokenList.append(tokens_ids)
    # input_id_chunks.append(torch.tensor(tokens_ids))
    # print(tokens_ids)
        # context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
        # print(context_embeddings) 
    # break
print(len(tokenList))


# %%
train_x=tokenList[0:500]
train_y=trueValues[0:500]
valid_x=tokenList[500:600]
valid_y=trueValues[500:600]
test_x=tokenList[600:]
test_y=trueValues[600:]
print(len(train_y), len(valid_y), len(test_y))


# %%
train_on_gpu = torch.cuda.is_available()


# %%
from torch.utils.data import DataLoader, TensorDataset

#create Tensor Dataset
train_data=TensorDataset(torch.LongTensor(train_x), torch.IntTensor(train_y))
valid_data=TensorDataset(torch.LongTensor(valid_x), torch.IntTensor(valid_y))
test_data=TensorDataset(torch.LongTensor(test_x), torch.IntTensor(test_y))

#dataloader
batch_size=10
train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader=DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=True)


# %%
import torch.nn as nn
 
class SentimentalLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):    
        """
        Initialize the model by setting up the layers
        """
        super().__init__()
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        
        #Embedding and LSTM layers
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        #dropout layer
        self.dropout=nn.Dropout(0.3)
        
        #Linear and sigmoid layer
        self.fc1=nn.Linear(hidden_dim, 64)
        self.fc2=nn.Linear(64, 16)
        self.fc3=nn.Linear(16,output_size)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size=x.size()
        
        #Embadding and LSTM output
        embedd=self.embedding(x)
        lstm_out, hidden=self.lstm(embedd, hidden)
        
        #stack up the lstm output
        lstm_out=lstm_out.contiguous().view(-1, self.hidden_dim)
        
        #dropout and fully connected layers
        out=self.dropout(lstm_out)
        out=self.fc1(out)
        out=self.dropout(out)
        out=self.fc2(out)
        out=self.dropout(out)
        out=self.fc3(out)
        sig_out=self.sigmoid(out)
        
        sig_out=sig_out.view(batch_size, -1)
        sig_out=sig_out[:, -1]
        
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize Hidden STATE"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


# %%
output_size = 1
embedding_dim = 512
hidden_dim = 256
n_layers = 2

net = SentimentalLSTM(vocab_size+1, output_size, embedding_dim, hidden_dim, n_layers)
print(net)


# %%
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

# training params

epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 10
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs=inputs.cuda()
            labels=labels.cuda()
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])
                if(train_on_gpu):
                    inputs=inputs.cuda()
                    labels=labels.cuda()

                # inputs, labels = inputs.cuda(), labels.cuda()  
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# %%
test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


# %%



