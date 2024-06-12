import os

from pydantic import BaseModel, Field, validator, HttpUrl, model_validator, constr
from typing import List, Optional, Union
from enum import Enum

class LossFunction(str, Enum):
    """Supported loss functions."""
    MSE = "mse"
    MAE = "mae"
    CROSS_ENTROPY = "cross_entropy"

class Optimizer(str, Enum):
    """Supported optimizers."""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"

class ModelConfig(BaseModel):
    """Configuration for the machine learning model."""
    n_features: int = Field(..., ge=1, description="Number of input features")
    hidden_size: int = Field(..., ge=1, description="Hidden size of each layer")
    num_layers: int = Field(..., ge=1, description="Number of layers in the model")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout probability")
    criterion: LossFunction = Field(LossFunction.MSE, description="Loss function to use")
    optimizer: Optimizer = Field(Optimizer.ADAM, description="Optimizer to use")
    learning_rate: float = Field(1e-3, gt=0.0, description="Learning rate")
    num_classes: int = Field(..., ge=2, description="Number of output classes")
    vocab_size: int = Field(..., ge=1, description="Vocabulary size (for text models)")
    embedding_dim: Optional[int] = Field(None, ge=1, description="Embedding dimension (for text models)")

    # Additional Validation: 
    # @validator("embedding_dim")
    # def validate_embedding_dim(cls, value, values):
    #     if values["vocab_size"] > 1 and not value:
    #         raise ValueError("embedding_dim must be specified if vocab_size is greater than 1")
    #     return value
    
    # # Additional Validation:
    # @model_validator(mode='after')
    # def check_criterion_compatibility(cls, values):
    #     if values["criterion"] == LossFunction.CROSS_ENTROPY and values["num_classes"] <= 2:
    #         raise ValueError("Cross-entropy loss requires more than 2 classes")
    #     return values

class DataConfig(BaseModel):
    train_path: str = Field(..., description="Path to training data")
    test_path: Optional[str] = Field(None, description="Path to test data (optional)")
    max_length: int = Field(128, ge=1, description="Maximum sequence length for text models")
    batch_size: int = Field(32, ge=1, description="Batch size for training and validation")
    num_workers: int = Field(0, ge=0, description="Number of data loader workers")
    # ... other data parameters

    # @validator("train_path", "val_path", "test_path", pre=True)
    # def validate_data_path(cls, value):
    #     # Check if the path exists
    #     if not os.path.exists(value):
    #         raise ValueError(f"Data path not found: {value}")
    #     return value
    
class GlobalConfig(BaseModel):
    # Add your Global Configs here
    logging_dir: str
    cache_dir: str
    random_seed: int = 42
    
class ExperimentConfig(BaseModel):
    # Add your Experiment Configs here
    experiment_name: str
    tracking_uri: HttpUrl
    num_epochs: int = 10

