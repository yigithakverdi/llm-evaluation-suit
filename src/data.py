## Default imports
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from src.config.schemas import DataConfig

class SardiStanceDataset(Dataset):
    def __init__(self, dataset, max_length, tokenizer):
      self.dataset = dataset
      self.max_length = max_length
      self.tokenizer = tokenizer
      self.target_encoding = {
          "AGAINST" : 0,
          "FAVOR" : 1,
          "NONE" : 2
      }

    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, index):
      sample = self.dataset.iloc[index]
      
      text = sample["text"]
      label = sample["label"]
      target = sample["choices"][sample["label"]]

      encoding = self.tokenizer(
          text,
          max_length=self.max_length,
          padding="max_length",
          truncation=True,
          return_tensors="pt"
      )

      target_encode = self.target_encoding[target] 

      input_ids = encoding["input_ids"].squeeze(0)
      attention_mask = encoding["attention_mask"].squeeze(0)
      
      return input_ids, attention_mask, target_encode

class SardiStanceDataModule(LightningDataModule):
    def __init__(self, config: DataConfig, tokenizer):
      super().__init__()
      self.config = config
      self.max_length = config.max_length
      self.batch_size = config.batch_size
      self.num_workers = config.num_workers
      
      # self.tokenizer = tokenizer
      from transformers import AutoTokenizer
      self.tokenizer = AutoTokenizer
      
    def _load_dataset(self):
      """      
      Load the dataset from the path
      path information is stored in the config
      """
      self.path_train = self.config.train_path
      self.path_test = self.config.test_path
      self.dataset = {
          "train" : pd.read_json(self.path_train, lines=True),
          "test": pd.read_json(self.path_test, lines=True)
          }

    def setup(self):
      """
      Default setup function of PyTorch Lightning, sets up the dataset
      """
      self._load_dataset()      
      self.train_dataset = SardiStanceDataset(
            self.dataset["train"],
            self.max_length,
            self.tokenizer,
            
        )

      self.test_dataset = SardiStanceDataset(
            self.dataset["test"],
            self.max_length,
            self.tokenizer,            
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def debug_dataloader(self, dataloader, num_batches=1):
      print("Single sample from torch.utils.Dataset initialization")
      print(self.train_dataset[0])
      print("Per batch debugging:")
      for batch_index, (input_ids, attention_mask, target_encode) in enumerate(dataloader):
        if num_batches == batch_index:
          break
        print(f"  Batch {batch_index + 1}:")
        print("   Input IDs:", input_ids)
        print("   Attention Mask:", attention_mask)
        print("   Target Encoding:", target_encode)
        print()
  