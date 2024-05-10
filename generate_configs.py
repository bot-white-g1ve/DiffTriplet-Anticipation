import os
import json
import copy
from debug import d_print

# General Setting
platform_used = "google cloud" # ["google_cloud", "NCI"]
if platform_used == "google cloud":
    run_inside_folder = True
debug = True

# Detailed Setting
dataset_used = "CholecT45"
feature_used = 'feature-RDV-4x4'
options = {
    "ant_range": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 75, 100, 200]
}
config_name = 'RDV-T45One5'
repeat_num = 1

# Default Setting
split_num = 5 # how many splits for cross-validation

splits_T45_cv = {"k1": ["VID79","VID02","VID51","VID06","VID25","VID14","VID66","VID23","VID50"],
                  "k2": ["VID80","VID32","VID05","VID15","VID40","VID47","VID26","VID48","VID70"],
                  "k3": ["VID31","VID57","VID36","VID18","VID52","VID68","VID10","VID08","VID73"],
                  "k4": ["VID42","VID29","VID60","VID27","VID65","VID75","VID22","VID49","VID12"],
                  "k5": ["VID78","VID43","VID62","VID35","VID74","VID01","VID56","VID04","VID13"]
                }

splits_T50_cv = {"k1": ["VID79","VID02","VID51","VID06","VID25","VID14","VID66","VID23","VID50","VID111"],
                  "k2": ["VID80","VID32","VID05","VID15","VID40","VID47","VID26","VID48","VID70","VID96"],
                  "k3": ["VID31","VID57","VID36","VID18","VID52","VID68","VID10","VID08","VID73","VID103"],
                  "k4": ["VID42","VID29","VID60","VID27","VID65","VID75","VID22","VID49","VID12","VID110"],
                  "k5": ["VID78","VID43","VID62","VID35","VID74","VID01","VID56","VID04","VID13","VID92"]
                }

if debug == True:
    splits_T45_cv = {
        "k1": ["VID79"],
        "k2": ["VID80"],
        "k3": ["VID31"],
        "k4": ["VID42"],
        "k5": ["VID78"]
    }

train_videos_T45_cv = {
    'k1': splits_T45_cv['k2'] + splits_T45_cv['k3'] + splits_T45_cv['k4'] + splits_T45_cv['k5'],
    'k2': splits_T45_cv['k3'] + splits_T45_cv['k4'] + splits_T45_cv['k5'] + splits_T45_cv['k1'],
    'k3': splits_T45_cv['k4'] + splits_T45_cv['k5'] + splits_T45_cv['k1'] + splits_T45_cv['k2'],
    'k4': splits_T45_cv['k5'] + splits_T45_cv['k1'] + splits_T45_cv['k2'] + splits_T45_cv['k3'],
    'k5': splits_T45_cv['k1'] + splits_T45_cv['k2'] + splits_T45_cv['k3'] + splits_T45_cv['k4'],
}

test_videos_T45_cv = {
    'k1': splits_T45_cv['k1'],
    'k2': splits_T45_cv['k2'],
    'k3': splits_T45_cv['k3'],
    'k4': splits_T45_cv['k4'],
    'k5': splits_T45_cv['k5'],
}

train_videos_T50_cv = {
    'k1': splits_T50_cv['k2'] + splits_T50_cv['k3'] + splits_T50_cv['k4'] + splits_T50_cv['k5'],
    'k2': splits_T50_cv['k3'] + splits_T50_cv['k4'] + splits_T50_cv['k5'] + splits_T50_cv['k1'],
    'k3': splits_T50_cv['k4'] + splits_T50_cv['k5'] + splits_T50_cv['k1'] + splits_T50_cv['k2'],
    'k4': splits_T50_cv['k5'] + splits_T50_cv['k1'] + splits_T50_cv['k2'] + splits_T50_cv['k3'],
    'k5': splits_T50_cv['k1'] + splits_T50_cv['k2'] + splits_T50_cv['k3'] + splits_T50_cv['k4'],
}

test_videos_T50_cv = {
    'k1': splits_T50_cv['k1'],
    'k2': splits_T50_cv['k2'],
    'k3': splits_T50_cv['k3'],
    'k4': splits_T50_cv['k4'],
    'k5': splits_T50_cv['k5'],
}

train_videos_challenge = {
  'k1': ['VID01', 'VID10', 'VID22', 'VID29', 'VID42', 'VID50', 'VID60', 'VID73', 'VID05', 'VID02', 'VID12', 'VID23', 'VID31', 
  'VID43', 'VID51', 'VID62', 'VID75', 'VID18', 'VID04', 'VID13', 'VID25', 'VID32', 'VID47', 'VID52', 'VID66', 'VID78', 'VID36', 
  'VID06', 'VID14', 'VID26', 'VID35', 'VID48', 'VID56', 'VID68', 'VID79', 'VID65', 'VID08', 'VID15', 'VID27', 'VID40', 'VID49', 
  'VID57', 'VID70', 'VID80', 'VID74'],
}

test_videos_challenge = {
  'k1': ['VID92', 'VID96', 'VID103', 'VID110', 'VID111'],
}

default_params = {
   "ant_range": 0,

   "naming": "default",
   "root_data_dir":"first_layer",
   "result_dir": "first_layer/results",
   "dataset_name":"",
   "feature_subdir": "",
   "train_video_list": None,
   "test_video_list": None,
   "val_video_list": None,

   "target_components": ['ivt', 'i', 'v', 't'],

   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":4,
      "num_f_maps":128,
      "input_dim":2500,
      "kernel_size":5,
      "ant_emb_dim":512,
      "normal_dropout_rate":0.1,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         1, 2, 3
      ]
   },
   "decoder_params":{
      "num_layers":4,  
      "num_f_maps":256, 
      "time_emb_dim":512,
      "ant_emb_dim":512,
      "kernel_size":11,
      "dropout_rate":0.2
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":8,
      "ddim_sampling_eta":1.0,
      "snr_scale":1.0,
      "cond_types":[
         "full", "full", "zero"
      ],
      "xt_mask_groups":None,   #useless
      "xt_mask_reverse":False, #useless
      "guidance_scale":1.0,
      "detach_decoder":False,
      "cross_att_decoder":False
   },

   "loss_weights":{
      "encoder_bce_loss":0.5,
      "decoder_bce_loss":0.5,
   },

   "causal":True,
   "sample_rate":1,
   "temporal_aug":True,

   "batch_size":1,
   "learning_rate":0.00005,
   "weight_decay":1e-5,
   "num_epochs":1201,
   "log_freq":100,
   "class_weighting":[],
   "set_sampling_seed":True,

   "log_train_results":True,
   "log_APs": ["i", "v", "t", "iv", "it", "ivt"],
   "evaluation_protocol": "Non-Challenge",

   "evalonly_params": {
       "pretrain_naming": "",
       "epochs": [],
       "mode": "decoder-agg",
   },
}

feature_dim_dict = {
  'feature-RDV-2500': 2500,
  'feature-RDV-4x5': 2000,
  'feature-RDV-4x4': 1600, 
  'feature-RDV-2x2': 400,
  'feature-RDV-6x8': 4800,
  'feature-RDV-3x3': 900,
  'feature-SelfDistillSwin': 1024,
}

# Functions
def generate_cv_config(params_template, default_feature_prefix, options, naming_prefix, repeat_num, split_num, pretrain_prefix=None, pretrain_suffix=None, pretrain_epochs=None):
    for key, values in options.items():
        for iv, value in enumerate(values):
            for repeat_id in range(repeat_num):
                for split_id in range(1, split_num+1):

                    params = copy.deepcopy(params_template)

                    if type(value) == list:
                        params['naming'] = f'{naming_prefix}-S{split_id}-{repeat_id}-{key}-{"_".join([str(i) for i in value])}'
                        if type(value[0]) == list:
                            params['naming'] = f'{naming_prefix}-S{split_id}-{repeat_id}-{key}-{iv}'
                    else:
                        params['naming'] = f'{naming_prefix}-S{split_id}-{repeat_id}-{key}-{value}'
                    
                    if pretrain_prefix and pretrain_suffix and pretrain_epochs:
                        params['evalonly_params']['pretrain_naming'] = f'{pretrain_prefix}-S{split_id}-{repeat_id}{pretrain_suffix}'
                        params['evalonly_params']['epochs'] = pretrain_epochs

                    #################
                    
                    if params['dataset_name'] == 'CholecT45':
                        params['train_video_list'] = train_videos_T45_cv[f'k{split_id}']
                        params['test_video_list'] = test_videos_T45_cv[f'k{split_id}']
                        params['val_video_list'] = []
                        
                    if params['dataset_name'] == 'CholecT50':
                        params['train_video_list'] = train_videos_T50_cv[f'k{split_id}']
                        params['test_video_list'] = test_videos_T50_cv[f'k{split_id}']
                        params['val_video_list'] = []

                    if params['dataset_name'] == 'Challenge':
                        params['train_video_list'] = train_videos_challenge[f'k{split_id}']
                        params['test_video_list'] = test_videos_challenge[f'k{split_id}']
                        params['val_video_list'] = []
                                            
                    params['feature_subdir'] = f'{default_feature_prefix}-k{split_id}'
                    params['encoder_params']['input_dim'] = feature_dim_dict[default_feature_prefix]

                    params['num_epochs'] = 1201
                    params['log_train_results'] = False # to be turned on later
                    
                    ################

                    assert (key != 'dataset_name')
                    # assert (key != 'feature_prefix')    #!!!!!!!!!!!!!

                    if key == 'baseline':
                        pass
                    elif key.startswith('encoder_'):
                        sub_key = key[8:]
                        assert(sub_key in params['encoder_params'].keys())
                        params['encoder_params'][sub_key] = value
                        # if sub_key == 'num_layers':
                        #     temp = [value-1, value-2, value-3]
                        #     params['encoder_params']['feature_layer_indices'] = [l for l in temp if l > 0]
                    elif key.startswith('decoder_'):
                        sub_key = key[8:]
                        assert(sub_key in params['decoder_params'].keys())
                        params['decoder_params'][sub_key] = value

                    elif key.startswith('diffusion_'):
                        sub_key = key[10:]
                        assert(sub_key in params['diffusion_params'].keys())
                        params['diffusion_params'][sub_key] = value

                    elif key.startswith('lossw_'):
                        sub_key = key[6:]
                        assert(sub_key in params['loss_weights'].keys())
                        params['loss_weights'][sub_key] = value

                    elif key.startswith('evalonly_'):
                        sub_key = key[9:]
                        assert(sub_key in params['evalonly_params'].keys())
                        params['evalonly_params'][sub_key] = value

                    elif key == 'feature_prefix':
                        params['feature_subdir'] = f'{value}-k{split_id}'
                        params['encoder_params']['input_dim'] = feature_dim_dict[value]

                    else:
                        assert(key in params.keys())
                        params[key] = value

                        if key == 'target_components':
                            if value != ['ivt', 'i', 'v', 't']:
                                params['diffusion_params']['guidance_scale'] = 0
                            if 'i' not in value:
                                params['log_APs'].remove('i')
                            if 'v' not in value:
                                params['log_APs'].remove('v')
                            if 't' not in value:
                                params['log_APs'].remove('t')

                    if not os.path.exists('configs'):
                        os.makedirs('configs')

                    file_name = os.path.join('configs', f'{params["naming"]}.json')

                    with open(file_name, 'w') as outfile:
                        json.dump(params, outfile, ensure_ascii=False, indent=4)

def feature_subdir_from_NCI_to_google_cloud(dataset_used, feature_used):
    second_dash_index = feature_used.find('-', feature_used.find('-') + 1)
    feature_used = feature_used[:second_dash_index + 1] + dataset_used + '-' + feature_used[second_dash_index + 1:]
    return feature_used
                        
if __name__ == "__main__":
    if platform_used == "google cloud":
        if run_inside_folder == True:
            default_params["root_data_dir"] = "../first_layer"
            default_params["result_dir"] = "../first_layer/results"
        elif run_inside_folder == False:
            default_params["root_data_dir"] = "first_layer"
            default_params["result_dir"] = "first_layer/results"
        feature_used = feature_subdir_from_NCI_to_google_cloud(dataset_used, feature_used)
        feature_dim_dict = {
          'feature-RDV-CholecT45-2500': 2500,
          'feature-RDV-CholecT45-4x5': 2000,
          'feature-RDV-CholecT45-4x4': 1600, 
          'feature-RDV-CholecT45-2x2': 400,
          'feature-RDV-CholecT45-6x8': 4800,
          'feature-RDV-CholecT45-3x3': 900,
          'feature-RDV-CholecT50-2500': 2500,
          'feature-RDV-CholecT50-4x5': 2000,
          'feature-RDV-CholecT50-4x4': 1600, 
          'feature-RDV-CholecT50-2x2': 400,
          'feature-RDV-CholecT50-6x8': 4800,
          'feature-RDV-CholecT50-3x3': 900,
          'feature-SelfDistillSwin': 1024,
        }
    if dataset_used == "CholecT45":
        default_params_T45 = copy.deepcopy(default_params)
        default_params_T45['dataset_name'] = 'CholecT45'
        default_params_T45['weight_decay'] = 5e-5
        params_template = default_params_T45
    elif dataset_used == "CholecT50":
        default_params_T50 = copy.deepcopy(default_params)
        default_params_T50['dataset_name'] = 'CholecT50'
        params_template = default_params_T50
        
    generate_cv_config(
      params_template=params_template, 
      default_feature_prefix=feature_used,
      options=options, 
      naming_prefix=config_name, 
      repeat_num=repeat_num, 
      split_num=split_num, 
    )
        