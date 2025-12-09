import pandas as pd

df_b = pd.read_csv('balanced.csv')



import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from sklearn.preprocessing import LabelEncoder

from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# clean data
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    text = text.lower()
    return text

df_b['lyrics'] = df_b['text'].apply(clean_text)

# Tokenizer
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=30000,
)

tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(df_b['lyrics'], trainer=trainer)
vocab = tokenizer.get_vocab()

print(f"Vocab size: {len(vocab)}")
print(f"Total tokens in corpus: {sum(len(text.split()) for text in df_b['lyrics'])}")

# encode to token IDs
id_seq = [
    torch.tensor(x.ids, dtype=torch.int64)
    for x in tokenizer.encode_batch(df_b['lyrics'])
]

# pad sequences
padded_seq = pad_sequence(id_seq, batch_first=True)
padded_seq = padded_seq[:, :100]
print(f"Padded sequences shape: {padded_seq.shape}")

# pncode emotion labels
label_encoder = LabelEncoder()
df_b['emotion_encoded'] = label_encoder.fit_transform(df_b['emotion'])

# show label mapping
print("\nEmotion label mapping:")
for i, emotion in enumerate(label_encoder.classes_):
    count = (df_b['emotion'] == emotion).sum()
    print(f"  {emotion}: {i} (count: {count})")

# convert to tensor
targets = torch.tensor(df_b['emotion_encoded'].values, dtype=torch.int64)
print(f"Targets shape: {targets.shape}")
print(f"Targets dtype: {targets.dtype}")

# create dataset object
dataset = torch.utils.data.TensorDataset(padded_seq, targets)
print(f"Dataset created with {len(dataset)} samples")


torch.manual_seed(0)
(train_dataset, val_dataset) = random_split(dataset, [0.7, 0.3])
len(train_dataset), len(val_dataset)





# CUSTOM LSTM MODEL

vocab_size = len(vocab)
num_classes = len(df_b['emotion'].unique())

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics import Accuracy, F1Score
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from torchmetrics.classification import BinaryAccuracy

# LIGHTNING MODULE
class MyLightning(LightningModule):
    def __init__(self, learning_rate=0.001, weight_decay=0.01):
        super().__init__()
        # self.save_hyperparameters()

        #torchmetrics accuracy
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        #F1 Scores
        # self.train_f1 = F1Score(task="binary")
        # self.val_f1 = F1Score(task="binary")

    def training_step(self, batch, batch_idx):
        x, target = batch
        y = self.forward(x)
        loss = nn.CrossEntropyLoss()(y, target)

        # get predictions
        preds = torch.argmax(y, dim = 1)
        self.accuracy(preds, target)

        self.log("train_loss", loss, prog_bar=True)
        self.log("accuracy", self.accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):

        x, target = batch

        # forward pass
        y = self.forward(x)

        # make sure target is a tensor
        if isinstance(target, list):
            target = torch.tensor(target, device=y.device)

        # check if we have proper shape for binary classification
        if y.shape[1] > 1:
            preds = torch.argmax(y, dim=1)
        else:

            preds = (y > 0.5).float()

        # accuracy
        self.accuracy(preds, target)
        self.log("val_accuracy", self.accuracy, prog_bar=True)


class MyLSTM(MyLightning):
    def __init__(self, vocab_size=len(vocab), dim_emb=200, dim_hidden=256,
                 num_classes=num_classes, dropout=0.5, learning_rate=0.001, filename="metrics.txt"):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate

        #Embedding
        self.embedding = nn.Embedding(vocab_size, dim_emb, padding_idx=0)

        def init_weights(self):
          initrange = 0.1
          nn.init.uniform_(self.encoder.weight, -initrange, initrange)
          nn.init.zeros_(self.decoder.bias)
          nn.init.uniform_(self.decoder.weight, -initrange, initrange)

        # LSTM with layer normalization
        self.lstm = nn.LSTM(
            input_size=dim_emb,
            hidden_size=dim_hidden,
            num_layers=2,
            batch_first=True,
            #bidirectional LSTM
            bidirectional=True,
            dropout=dropout
        )
        self.output = nn.Linear(dim_hidden, num_classes)

        #attention
        self.attention = nn.Linear(dim_hidden * 2, 1, bias=False)

        #classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, batch_of_seqs):
      emb = self.embedding(batch_of_seqs)
      _, (state, _) = self.lstm(emb)
      # state: (num_layers, batch, dim_state)
      output = self.output(state[-1])
      return output

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid),
        )


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from lightning import seed_everything
import shutil, os, time


batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




def train(*, name:str, model:LightningModule, epochs: int):
  seed_everything(0)

  logger = CSVLogger(save_dir="logs/", name=name)
  trainer = Trainer(
      max_epochs=epochs,
      logger=logger,
  )


  try:
    shutil.rmtree(f"./logs/{name}/")
    os.mkdirs(f"./logs/{name}")
  except:
    pass

  start = time.time()
  trainer.fit(model, train_dataloader, val_dataloader)\


  duration = time.time() - start
  print(f"Training time: {duration:0.2f} seconds.")






import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import pickle

class EmotionClassifierFromCheckpoint:
    def __init__(self, checkpoint_path, vocab, class_names=None, device=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # loading the model from checkpoint file
        self.model = MyLSTM.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # freeze, since model is already trained
        self.model.freeze()

        # Extract hyperparameters from the model
        self.hparams = self.model.hparams if hasattr(self.model, 'hparams') else {}

        self.vocab = vocab
        self.class_names = class_names


    def process_text(self, text, max_length=100):
        # tokenize
        tokens = text.lower().split()

        # get the ID for the unknown token and padding token
        # Assuming 0 is a safe default if [UNK]/[PAD] are somehow missing, or check vocab directly
        unk_token_id = self.vocab.get("[UNK]", 0)
        pad_token_id = self.vocab.get("[PAD]", 0)


        token_indices = []

        #unk
        for token in tokens:
            token_id = self.vocab.get(token, unk_token_id)
            token_indices.append(token_id)

        token_indices = token_indices + [pad_token_id] * (max_length - len(token_indices))

        return torch.tensor([token_indices], dtype=torch.long).to(self.device)



    def predict(self, text, return_probabilities=False):
        # predict emotions for input lyrics
        input_tensor = self.process_text(text)
        outputs = self.model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        if return_probabilities:
            prob_dict = {
                self.class_names[i]: float(probabilities[0][i])
                for i in range(len(self.class_names))
            }

            output = str(str(round(prob_dict['anger'],2))+"% likely to be \"anger\"\n"+
            str(round(prob_dict['joy']*100,2))+"% likely to be \"joy\"\n"+
            str(round(prob_dict['sadness']*100,2))+"% likely to be \"sadness\"")

            print(str(round(prob_dict['anger']*100,2))+"% likely to be \"anger\"")
            print(str(round(prob_dict['joy']*100,2))+"% likely to be \"joy\"")
            print(str(round(prob_dict['sadness']*100,2))+"% likely to be \"sadness\"")

            return output



def getOutput():


    classifier = EmotionClassifierFromCheckpoint(
        checkpoint_path = "epoch=9-step=3290.ckpt",
        vocab = vocab,
        class_names = ['anger', 'joy', 'sadness']
    )


    with open('Lyrics.txt', 'r') as file:
        lyrics = file.read().rstrip()

    lyrics = clean_text(lyrics)
    # print(lyrics)
    # print(type(vocab))

    output = classifier.predict(lyrics, return_probabilities=True)
    return output
    # if output_dict:
    #     output_lines = []
    #     for emotion, prob in output_dict.items():
    #         percentage = prob * 100  # Convert to percentage
    #         output_lines.append(f"{percentage:.1f}% likely to be \"{emotion}\"")
            
    #     return "\n".join(output_lines)
    # else:
    #     return "No prediction available"