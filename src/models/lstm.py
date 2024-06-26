import lightning as L
from torch import optim, nn
from ..config.loaders import load_config, ModelConfig

class SardiStanceLSTM(L.LightningModule):
  def __init__(self, config: ModelConfig):
    super(SardiStanceLSTM, self).__init__()
    self.save_hyperparameters()

    self.n_features = config.n_features
    self.hidden_size = config.hidden_size
    self.num_layers = config.num_layers
    self.dropout = config.dropout
    self.criterion = config.criterion
    self.learning_rate = config.learning_rate
    self.num_classes = config.num_classes
    self.vocab_size = config.vocab_size

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