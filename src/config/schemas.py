import os
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Optional, Union

class ModelConfig(BaseModel):
    # ... (model parameters)
    pass

class DataConfig(BaseModel):
    train_path: str = Field(..., description="Path to training data")
    val_path: str = Field(..., description="Path to validation data")
    test_path: Optional[str] = Field(None, description="Path to test data (optional)")
    batch_size: int = Field(32, ge=1, description="Batch size for training and validation")
    # ... other data parameters

    @validator("train_path", "val_path", "test_path", pre=True)
    def validate_data_path(cls, value):
        # Check if the path exists
        if not os.path.exists(value):
            raise ValueError(f"Data path not found: {value}")
        return value
    
    
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

