import json
import argparse
import os
import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from dataset import get_data_dict, VideoFeatureDataset
from utils import load_config_file, set_random_seed, get_convert_matrix
from model import ASDiffusionModel
import torch.nn.functional as F
import numpy as np
from debug import d_start, d_print

# General Settings
platform_used = "google cloud" # ["google_cloud", "NCI"]

# Functions
def load_config(config_file):
    configs = json.load(open(config_file))

    if 'result_dir' not in configs:
       configs['result_dir'] = 'result'

    return configs

def eval_and_save_the_result():
    model = ASDiffusionModel(configs['encoder_params'], configs['decoder_params'], configs['diffusion_params'], configs['causal'], num_targets, guidance_matrices, device)
    d_print("eval_and_save_the_result", f"The device used is {device}")
    model = model.to(device)

    state_dict_path = os.path.join(configs["evalonly_params"]["pretrain_folder"], configs["evalonly_params"]["pretrain_naming"])
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()

    whole_dataset = ConcatDataset(train_test_dataset, test_test_dataset)
    
    if configs['set_sampling_seed']:
        seed = video_idx
    else:
        seed = None
    
    with torch.no_grad():
        for video_idx in tqdm(range(len(whole_dataset))):
            feature, label, video = whole_dataset[video_idx]
            if configs['ant_range'] > 0:
                feature = [i[:,:,:-configs['ant_range']] for i in feature]
            ant_range = configs['ant_range']
            ant = torch.tensor([configs['ant_range']], device=device).long()

            # begin: eval
            if configs["evalonly_params"]["eval_mode"] in ['encoder-agg', 'decoder-agg', 'ensemble-agg']:
                
                if configs["evalonly_params"]["eval_mode"] == 'encoder-agg':
                    output = [model.encoder(feature[i].to(device), ant) 
                           for i in range(len(feature))] # output is a list of tuples
                    output = [F.sigmoid(i).cpu() for i in output]
                if configs["evalonly_params"]["eval_mode"] == 'decoder-agg':
                    output = [model.ddim_sample(feature[i].to(device), ant_range, seed) 
                               for i in range(len(feature))] # output is a list of tuples
                    output = [i.cpu() for i in output]
                if configs["evalonly_params"]["eval_mode"] == 'ensemble-agg':
                    output_encoder = [model.encoder(feature[i].to(device), ant) 
                           for i in range(len(feature))]
                    output_encoder = [F.sigmoid(i).cpu() for i in output_encoder]
                    output_decoder = [model.ddim_sample(feature[i].to(device), ant_range, seed) 
                               for i in range(len(feature))] 
                    output_decoder = [i.cpu() for i in output_decoder]
                    output = [(output_encoder[i] + output_decoder[i]) / 2 # TO DO: maybe change combination weights
                        for i in range(len(feature))]

                assert(output[0].shape[0] == 1)
                agg_output = np.zeros(label.shape) # C x T
                for offset in range(configs['sample_rate']):
                    agg_output[:,offset::configs['sample_rate']] = output[offset].squeeze(0).numpy()
                output = agg_output

            if configs["evalonly_params"]["eval_mode"] in ['encoder-noagg', 'decoder-noagg', 'ensemble-noagg']: # temporal aug must be true

                if configs["evalonly_params"]["eval_mode"] == 'encoder-noagg':
                    output = model.encoder(feature[len(feature)//2].to(device), ant) 
                    output = F.sigmoid(output).cpu()
                if configs["evalonly_params"]["eval_mode"] == 'decoder-noagg':
                    output = model.ddim_sample(feature[len(feature)//2].to(device), ant_range, seed) 
                    output = output.cpu() 
                if configs["evalonly_params"]["eval_mode"] == 'ensemble-noagg':
                    output_encoder = model.encoder(feature[len(feature)//2].to(device), ant) 
                    output_encoder = F.sigmoid(output_encoder).cpu()
                    output_decoder = model.ddim_sample(feature[len(feature)//2].to(device), ant_range, seed) 
                    output_decoder = output_decoder.cpu() 
                    output = (output_encoder + output_decoder) / 2 # TO DO: maybe change combination weights

                # TO DO
                assert(output.shape[0] == 1) # 1xCxT
                output = F.interpolate(output, size=label.shape[1])
                output = output.squeeze(0).numpy()
                
            assert(output.max() <= 1 and output.min() >= 0)
            # end: eval

            d_print("eval_and_save_the_result", f"output's shape is {output.shape}")

if __name__ == '__main__':
    d_start()
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str, help="config")
    parser.add_argument('device', type=int, help="-1 if cpu, >=0 if gpu")
    args = parser.parse_args()

    configs = load_config(args.config)
    d_print("main", f"the config is\n{json.dumps(configs, indent=4)}")

    # specially for usage in NCI, set the cuda device to visible
    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    mapping_file = os.path.join(configs['root_data_dir'], configs['dataset_name'], 'label_mapping.txt')

    train_data_dict = get_data_dict(
        root_data_dir=configs['root_data_dir'], 
        dataset_name=configs['dataset_name'], 
        feature_subdir=configs['feature_subdir'],
        mapping_file=mapping_file,
        target_components=configs['target_components'],
        video_list=configs['train_video_list'], 
        sample_rate=configs['sample_rate'], 
        temporal_aug=configs['temporal_aug'],
        ant_range=configs['ant_range'],
    )

    train_train_dataset = VideoFeatureDataset(train_data_dict, configs['target_components'], mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, configs['target_components'], mode='test')

    if configs['test_video_list']:
        test_data_dict = get_data_dict(
            root_data_dir=configs['root_data_dir'], 
            dataset_name=configs['dataset_name'], 
            feature_subdir=configs['feature_subdir'],
            mapping_file=mapping_file,
            target_components=configs['target_components'],
            video_list=configs['test_video_list'], 
            sample_rate=configs['sample_rate'], 
            temporal_aug=configs['temporal_aug'],
            ant_range=configs['ant_range']
        )
        test_test_dataset = VideoFeatureDataset(test_data_dict, configs['target_components'], mode='test')
    else:
        test_test_dataset = None

    if configs['val_video_list']:
        val_data_dict = get_data_dict(
            root_data_dir=configs['root_data_dir'], 
            dataset_name=configs['dataset_name'], 
            feature_subdir=configs['feature_subdir'],
            mapping_file=mapping_file,
            target_components=configs['target_components'],
            video_list=configs['val_video_list'], 
            sample_rate=configs['sample_rate'], 
            temporal_aug=configs['temporal_aug'],
            ant_range=configs['ant_range'],
        )
        val_test_dataset = VideoFeatureDataset(val_data_dict, configs['target_components'], mode='test')
    else:
        val_test_dataset = None

    num_targets = train_train_dataset.num_targets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_print("main", f"the device used is {device}")
    
    guidance_matrices = {}
    guidance_matrices['i_ivt'] = get_convert_matrix(mapping_file, 'i', 'ivt')
    guidance_matrices['v_ivt'] = get_convert_matrix(mapping_file, 'v', 'ivt')
    guidance_matrices['t_ivt'] = get_convert_matrix(mapping_file, 't', 'ivt')
    guidance_matrices['i_ivt'] = torch.from_numpy(guidance_matrices['i_ivt']).to(device).float().unsqueeze(0)
    guidance_matrices['v_ivt'] = torch.from_numpy(guidance_matrices['v_ivt']).to(device).float().unsqueeze(0)
    guidance_matrices['t_ivt'] = torch.from_numpy(guidance_matrices['t_ivt']).to(device).float().unsqueeze(0)
    
    eval_and_save_the_result()
    
    