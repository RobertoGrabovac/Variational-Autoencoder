import importlib
import argparse
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Launch the training session for a VAE using a custom configuration file.')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='<config_folder_path>/<vae_type>.yaml',
                        help='Specify the YAML configuration file that defines the model and training parameters.',
                        default='configs/mse_vae.yaml')
    
    return parser.parse_args()

def load_model_config(filename):
    try:
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError as fnf_error:
        print(f"Config file not found: {fnf_error}")
        sys.exit(1)
    except yaml.YAMLError as yaml_error:
        print(f"Error parsing YAML file: {yaml_error}")
        sys.exit(1)

def load_model_class(model_name: str):
    if model_name == "MSEVAE":
        module = importlib.import_module("models.mse_vae")
        return getattr(module, "MSEVAE")
    elif model_name == "MSSIMVAE":
        module = importlib.import_module("models.mssim_vae")
        return getattr(module, "MSSIMVAE")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def initialize_models():
    vae_models = {
        "MSEVAE": load_model_class("MSEVAE"),
        "MSSIMVAE": load_model_class("MSSIMVAE")
    }
    return vae_models

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params