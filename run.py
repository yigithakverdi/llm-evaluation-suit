from src.experiments import LSTMSardiStanceExperiment
from src.config.loaders import load_config

def main():
    load_config()
    
    lstm_experiment = LSTMSardiStanceExperiment()
    lstm_experiment.run()

if __name__ == "__main__":
    main()
