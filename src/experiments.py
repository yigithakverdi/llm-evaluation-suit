import transformers
import datetime as dt
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from .data import SardiStanceDataModule
from src.models.lstm import SardiStanceLSTM
from src.config.schemas import DataConfig

class BaseExperiment():
    def __init__(self):
        self.dataloader = None
        self.trainer = None
        # Instantiate DataLoaders, Trainers, etc. here

    def run(self):
        # Implement the logic for running the experiment here
        pass

class BaselineSardiStanceExperiment(BaseExperiment):
    pass

class LSTMSardiStanceExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()
    
    def run(self):
        chekcpoint = "bert-base-uncased"
        tokenizer = transformers.AutoTokenizer.from_pretrained(chekcpoint)
        name_of_the_run = str(dt.datetime.now())[:-7]
        wandb_logger = WandbLogger(log_model="all", project="llm-evaluation-suit", name=name_of_the_run)

        dm = SardiStanceDataModule(
            DataConfig,
            max_length=128,
            tokenizer=tokenizer,
            batch_size=1,
            num_workers=8,
        )
        

        trainer = L.Trainer(
            max_epochs=p['max_epochs'],
            accelerator="auto",
            logger=wandb_logger
        )

        model = SardiStanceLSTM()

        wandb_logger.watch(model, log="all")
        trainer.fit(model, dm)