import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils import convert_label_format, split_tensor_ivt
from scipy.ndimage import gaussian_filter1d

def get_data_dict(root_data_dir, dataset_name, feature_subdir, mapping_file,
    target_components, video_list, sample_rate=4, temporal_aug=True, ant_range=0):

    feature_dir = os.path.join(root_data_dir, dataset_name, feature_subdir)

    if dataset_name in ['CholecT50', 'Challenge']:
        label_dir = os.path.join(root_data_dir, dataset_name, 'labels')
    elif dataset_name == 'CholecT45':
        label_dir = os.path.join(root_data_dir, dataset_name, 'triplet')
    else:
        raise Exception('Invalid Dataset')

    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file = os.path.join(feature_dir, '{}.npy'.format(video))

        feature = np.load(feature_file, allow_pickle=True)

        if len(feature.shape) == 3:
            feature = np.swapaxes(feature, 0, 1)  
        elif len(feature.shape) == 2: # F,T
            feature = np.swapaxes(feature, 0, 1) # T,F
            feature = np.expand_dims(feature, 0) # 1,T,F
        else:
            raise Exception('Invalid Feature.')
                    
        if dataset_name in ['CholecT50', 'Challenge']:

            event_file = os.path.join(label_dir, '{}.json'.format(video))

            event = json.load(open(event_file))
            # dict_keys(['annotations', 'fps', 'licenses', 'info', 'categories', 'num_frames', 'video'])
            
            assert(event['fps'] == 1)
            frame_num = event['num_frames']

            event = [event['annotations'][str(i)] for i in range(frame_num)]

            event_seq_ivt = []
            for i in range(frame_num):
                event_seq_ivt.append([j[0] for j in event[i]])
                # -1 is background

            event_seq_ivt = convert_label_format(event_seq_ivt, num_classes=100, list_to_tensor=True)
            event_seq_ivt = event_seq_ivt.float()

        if dataset_name == 'CholecT45':

            event_file = os.path.join(label_dir, '{}.txt'.format(video))
            tensor_list = []
    
            with open(event_file, 'r') as file:
                for line in file:
                    vector = line.strip().split(',')
                    int_vector = [int(x) for x in vector[1:]]
                    tensor_list.append(int_vector)
            
            tensor = torch.tensor(tensor_list)
            event_seq_ivt = tensor.float()

        assert(feature.shape[1] == event_seq_ivt.shape[0]) # 1,T,F / T,C

        # if ant_range > 0:
        #     feature = feature[:,:-ant_range,:]
        #     event_seq_ivt = event_seq_ivt[ant_range:,:]

        # assert(feature.shape[1] == event_seq_ivt.shape[0]) # 1,T,F / T,C

        event_seq_i, event_seq_v, event_seq_t, event_seq_iv, event_seq_it = split_tensor_ivt(event_seq_ivt, mapping_file)

        event_seq_dict = {
            'i': event_seq_i,
            'v': event_seq_v,
            't': event_seq_t,
            'iv': event_seq_iv,
            'it': event_seq_it,
            'ivt': event_seq_ivt,
        }

        event_seq_raw = [event_seq_dict[i] for i in target_components]
        event_seq_raw = torch.cat(event_seq_raw, 1)

        if temporal_aug:
            
            feature = [
                feature[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext = [
                event_seq_raw[offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
                        
        else:
            feature = [feature[:,::sample_rate,:]]  
            event_seq_ext = [event_seq_raw[::sample_rate,:]]

        data_dict[video]['feature'] = [torch.from_numpy(i).float() for i in feature]
        data_dict[video]['event_seq_raw'] = event_seq_raw
        data_dict[video]['event_seq_ext'] = event_seq_ext
        
    return data_dict



def restore_full_sequence(x, full_len, left_offset, right_offset, sample_rate):
        
    frame_ticks = np.arange(left_offset, full_len-right_offset, sample_rate)
    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1]+1, 1)

    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    assert(len(frame_ticks) == len(x)) # Rethink this
    
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]
    out[frame_ticks[0]:frame_ticks[-1]+1] = interp_func(full_ticks)
    out[frame_ticks[-1]+1:] = x[-1]

    return out




class VideoFeatureDataset(Dataset):
    def __init__(self, data_dict, target_components, mode):
        super(VideoFeatureDataset, self).__init__()
        
        assert(mode in ['train', 'test'])

        self.data_dict = data_dict
        self.target_components = target_components
        self.mode = mode

        self.num_classes = {
            'i': 6,
            'v': 10,
            't': 15,
            'iv': 26,
            'it': 59,
            'ivt': 100,
        }

        self.num_targets = sum([self.num_classes[i] for i in self.target_components])
        self.video_list = [i for i in self.data_dict.keys()]
        
    def get_class_weights(self, class_weighting):
        
        # TO DO, Maybe related to the calculation of ivtmetrics, check later

        full_event_seq = torch.cat([self.data_dict[v]['event_seq_raw'] for v in self.video_list], dim=0)

        t_counts = full_event_seq.shape[0]
        p_counts = full_event_seq.sum(0)
        n_counts = t_counts - p_counts

        # p_weights = t_counts / (p_counts * 2 + 10)
        # n_weights = t_counts / (n_counts * 2 + 10)

        class_weights = n_counts / (p_counts + 10) # pos weights

        c_i = 0
        for component in self.target_components:
            n_i = c_i + self.num_classes[component]
            if component in class_weighting:
                pass
            else:
                class_weights[c_i:n_i] = 1
            c_i = n_i

        return class_weights

    def split_components(self, input_tensor): # Input: T x C
        output = {}

        c_i = 0
        for component in self.target_components:
            n_i = c_i + self.num_classes[component]
            output[component] = input_tensor[:,c_i:n_i]
            c_i = n_i

        return output

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]

        if self.mode == 'train':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_ext']

            temporal_aug_num = len(feature)
            temporal_rid = random.randint(0, temporal_aug_num - 1) # a<=x<=b
            feature = feature[temporal_rid]
            label = label[temporal_rid]

            spatial_aug_num = feature.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1) # a<=x<=b
            feature = feature[spatial_rid]
            
            feature = feature.T   # F x T
            label = label.T       # C x T

        if self.mode == 'test':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_raw']

            feature = [torch.swapaxes(i, 1, 2) for i in feature]  # [10 x F x T]
            label = label.T.unsqueeze(0)   # 1 X C X T'  

        return feature, label, video

    
