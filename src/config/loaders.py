import yaml
import os

from pydantic import ValidationError
from .schemas import ModelConfig, DataConfig, GlobalConfig, ExperimentConfig



def include_constructor(loader, node):
    # Extract the filename from the node
    filename = os.path.join(os.path.dirname(loader.name), node.value)
    
    with open(filename, 'r') as f:
        return yaml.safe_load(f)



def load_config(config_file="default.yaml", overrides_file=None, env_prefix="APP_"):
    """
    Loads configuration from YAML files, with optional overrides and environment variable support.

    Args:
        config_file (str): Path to the main configuration file (default: "default.yaml").
        overrides_file (str, optional): Path to a file with overrides.
        env_prefix (str, optional): Prefix for environment variables (default: "APP_").

    Returns:
        tuple: A tuple containing (ModelConfig, DataConfig, GlobalConfig, ExperimentConfig) instances.
    """    
    yaml.SafeLoader.add_constructor('!include', include_constructor)    
    with open(config_file, "r") as f:
        yaml.SafeLoader.name = config_file
        config_data = yaml.safe_load(f)
        print(config_data)

    if overrides_file:
        with open(overrides_file, "r") as f:
            overrides_data = yaml.safe_load(f)
        config_data.update(overrides_data)

    # Load environment variables with the specified prefix
    for key, value in config_data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():                
                env_var = f"{env_prefix}{key.upper()}_{subkey.upper()}"
                if env_var in os.environ:
                    print("Envirnment variable found")
                    print(f"Overriding {key}.{subkey}: {subvalue} with {env_var}")
                    config_data[key][subkey] = os.environ[env_var]

    try:
        return (
            ModelConfig(**config_data["model"]['LSTM']),
            DataConfig(**config_data["data"]['SardiStance']),
            # GlobalConfig(**config_data["global"]),
            # ExperimentConfig(**config_data["experiment"]),
        )
    except ValidationError as e:
        raise ValueError("Invalid configuration: " + str(e))


# def validate_config(config):
#     """
#     Performs additional validation on the loaded configuration.

#     Args:
#         config: The loaded configuration (tuple of ModelConfig, DataConfig, GlobalConfig, ExperimentConfig).

#     Raises:
#         ValueError: If any validation fails.
#     """
#     model_config, data_config, global_config, experiment_config = config

#     # Example validation:
#     if model_config.num_layers < 1:
#         raise ValueError("Model must have at least one layer.")

    # ... other custom validation rules
