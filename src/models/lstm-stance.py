import lightning as L
from torch import optim, nn

class SardiStanceLSTM(L.LightningModule):
  def __init__(self, 
               n_features, 
               hidden_size, 
               vocab_size,
               num_classes,
               num_layers,
               dropout,
               learning_rate,
               criterion):
    super(SardiStanceLSTM, self).__init__()
    self.save_hyperparameters()
    self.n_features = n_features
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.criterion = criterion
    self.learning_rate = learning_rate
    self.num_classes = num_classes
    self.vocab_size = vocab_size

    self._load_model()

  def _load_model(self):
    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
    self.lstm = nn.LSTM(input_size=self.hidden_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        batch_first=True)
    self.linear = nn.Linear(self.hidden_size, self.num_classes)

  def forward(self, x):
    embedded = self.embedding(x)
    lstm_out, _ = self.lstm(embedded)
    lstm_out = lstm_out[:, -1, :]
    logits = self.linear(lstm_out)
    return logits
  
  def training_step(self, batch, batch_idx):
    input_ids, attention_mask, target = batch
    logits = self.forward(input_ids, batch_idx)
    loss = self.criterion(logits, target)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids, attention_mask, target = batch
    outputs = self.forward(input_ids, batch_idx)
    loss = self.criterion(outputs, target)
    self.log("val_loss", loss)
    return loss
    
  def test_step(self, batch, batch_idx):
    input_ids, attention_mask, target = batch
    outputs = self.forward(input_ids, batch_idx)
    loss = self.criterion(outputs, target)
    self.log("test_loss", loss)
    return loss

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer