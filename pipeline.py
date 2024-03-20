import tomli
import shutil
import os
import argparse
from train import train
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
import torch

def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main(raw_config,parent_dir,real_data_path,num_samples_to_generate):
    print("Ricardo")
    print(raw_config)
    
    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda:1')
    
    timer = zero.Timer()
    timer.run()
    #save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    
    train(
        **raw_config['train']['main'],
        **raw_config['diffusion_params'],
        parent_dir=parent_dir,
        real_data_path=real_data_path,
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        change_val=False
    )

    sample(
        num_samples=num_samples_to_generate,
        batch_size=raw_config['sample']['batch_size'],
        disbalance=raw_config['sample'].get('disbalance', None),
        **raw_config['diffusion_params'],
        parent_dir=parent_dir,
        real_data_path=real_data_path,
        model_path=os.path.join(parent_dir, 'model.pt'),
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        seed=raw_config['sample'].get('seed', 0),
        change_val=False
    )

    # save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
   

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    
    main()