# %matplotlib inline

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
import os
import torch
# import torchnlp.nn as nlpnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation

cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda:1') if cuda else 'cpu'

# load all that we need

dataset = np.load('../dataset/wiki.train.npy')
fixtures_pred = np.load('../fixtures/dev_fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/dev_fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/test_fixtures/prediction.npz')  # test
fixtures_gen_test = np.load('../fixtures/test_fixtures/generation.npy')  # test
vocab = np.load('../dataset/vocab.npy')

# data loader

class LanguageModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset, batch_size, shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset        

    def __iter__(self):
        # concatenate your articles and build into batches
        
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            data = self.dataset[indices].copy()
        else:
            data = self.dataset
        data = np.concatenate(data, axis=0)
        data_input = data[:-1]
        data_target = data[1:]
        
        N = len(data)
        bs = self.batch_size
        # discard data that does not fit
        data_input = data_input[:bs * (N // bs)]
        data_target = data_target[:bs * (N // bs)]
        # reshape and transpose
        data_input = data_input.reshape(bs, N // bs).T
        data_target = data_target.reshape(bs, N // bs).T
        
        i = 0
        while i < (N // bs):
            base_length = 70
            s = 5
            rand = np.random.random()
            if rand > 0.95:
                L = int(np.random.normal(base_length / 2, s))
            else:
                L = int(np.random.normal(base_length, s))
            
            input_batch = data_input[i: min(i+L, N // bs)]
            target_batch = data_target[i: min(i+L, N // bs)]
            
            input_batch = torch.tensor(input_batch, dtype=torch.long).view(-1, bs) # L x BS            
            target_batch = torch.tensor(target_batch, dtype=torch.long).view(-1, bs) # L x BS
            
            i += L
            yield (input_batch, target_batch)

# model

class LanguageModel(nn.Module):
    """
        TODO: Define your model here
    """
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = 400
        self.hidden_size = 400 # 1150
        self.nlayers = 3
        
        # initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        nn.init.uniform_(self.embedding.weight.data, -0.1, 0.1)
        
        lb, ub = -1/np.sqrt(self.hidden_size), 1/np.sqrt(self.hidden_size)
        # initialize LSTM layers
        self.rnn = nn.LSTM(input_size=self.embed_size, 
                          hidden_size=self.hidden_size,
                          num_layers=self.nlayers)        
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.uniform_(param, lb, ub)
        
        # initialize linear layer
        self.scoring = nn.Linear(self.embed_size, vocab_size)
        self.scoring.weight = self.embedding.weight # wright tying

    def forward(self, seq_batch): # L x BS
        
        bs = seq_batch.size(1)
        
        embed = self.embedding(seq_batch) # L x BS x E
        hidden = None

        output_lstm, hidden = self.rnn(embed, hidden) # L x BS x H                
        
        output_lstm_flatten = output_lstm.view(-1, self.embed_size) # (L x BS) x H
        output_flatten = self.scoring(output_lstm_flatten) # (L x BS) x V
        return output_flatten.view(-1, bs, self.vocab_size) # L X BS x V
        
# model trainer

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
            patience=3)
        self.criterion = nn.CrossEntropyLoss()
                

    def train(self):
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(inputs, targets)
            if batch_num % 100 == 0:
                print('batch {}  loss = {}'.format(batch_num, epoch_loss / (batch_num + 1)))
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """ 
            TODO: Define code for training a single batch of inputs
        
        """
        inputs = Variable(inputs).to(DEVICE)
        targets = Variable(targets).to(DEVICE)
    
        outputs = self.model(inputs)
        loss = self.criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))
        self.optimizer.zero_grad()
        loss.backward()        
        self.optimizer.step()
        lpw = loss.item()         
        
        return lpw
    
    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)
        nll = test_prediction(predictions, fixtures_pred['out'])
        
        generated_logits = TestLanguageModel.generation(fixtures_gen, 20, self.model) # predictions for 20 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 20, self.model) # predictions for 20 words

        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   NLL: %.4f'
                      % (self.epochs, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-test-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


class TestLanguageModel:
    def prediction(inp, model):
        """
            TODO: write prediction code here
            
            :param inp:
            :return: a np.ndarray of logits
        """
        inputs = Variable(torch.tensor(inp.T, dtype=torch.long)).to(DEVICE) # L x BS        
                
        logits = model(inputs) # L x BS x V
        return logits[-1].cpu().detach().numpy() # BS x V

        
    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """        
        inputs = Variable(torch.tensor(inp.T, dtype=torch.long)).to(DEVICE) # L x BS
                            
        generated_words = []
        embed = model.embedding(inputs) # L x BS x E
        hidden = None
        output_lstm, hidden = model.rnn(embed, hidden) # L x BS x H        
        output = output_lstm[-1] # BS x H
        scores = model.scoring(output) # BS x V
        _,current_word = torch.max(scores, dim=1, keepdim=True) # BS x 1
        generated_words.append(current_word)
        if forward > 1:                      
            for i in range(forward-1):                
                embed = model.embedding(current_word.unsqueeze(0)).squeeze(2) # 1 x BS x E                
                output_lstm, hidden = model.rnn(embed, hidden) # L x BS x H                
                output = output_lstm[0] # BS x H
                scores = model.scoring(output) # BS x V
                _,current_word = torch.max(scores, dim=1, keepdim=True) # BS x 1
                generated_words.append(current_word)        
        return torch.cat(generated_words, dim=1).cpu().detach().numpy() # BS x forward


# TODO: define other hyperparameters here

NUM_EPOCHS = 20
BATCH_SIZE = 80


run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)


model = LanguageModel(len(vocab)).to(DEVICE)
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)


best_nll = 1e30  # set to super large value at first
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    trainer.scheduler.step(nll)
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch " + 
              str(epoch)+" with NLL: " + str(best_nll))
        trainer.save()
    

# Don't change these
# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation NLL')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()


# see generated output
print (trainer.generated[-1]) # get last generated output

