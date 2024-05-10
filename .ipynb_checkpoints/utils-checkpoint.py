import os
import json
import random
import torch
import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter

def load_config_file(config_file):

    all_params = json.load(open(config_file))

    if 'result_dir' not in all_params:
        all_params['result_dir'] = 'result'

    return all_params


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ############ Utils for CholecT50 dataset #################


def convert_label_format(input, num_classes, list_to_tensor=True):
  '''
  input: [[triplet_idx], [triplet_idx * n], [...], ...]
  output: tensor([T, C]) in multi-hot
  num_classes: specify how many classes, CholecT50 should be 100
  list_to_tensor: true means input => output, false means output => input

  note: if triplet_idx == -1, then the multi-hot tensor will be full 0
  if want to ues this for i, v or t, just use a different num_classes
  '''
  if list_to_tensor == True:
    output = torch.zeros(len(input), num_classes)
    for t, l in enumerate(input):
      for idx in l:
        assert(idx is not None)
        if idx != -1:
          output[t, idx] = 1
    
    return output
  
  if list_to_tensor == False:
    output=[]

    for l in input:
        indices = l.nonzero(as_tuple=False).view(-1).tolist()
        if not indices:
            indices = [-1]
        output.append(indices)

    return output

def split_tensor_ivt(input, mappingFilePath, num_classes = (6,10,15,26,59)):
    '''
    input: tensor([T, C]) in multi-hot for triplet
    output: [ tensor([T, C]) in multi-hot for i, v, t, iv, it]
    num_classes: respective count of classes for i, v and t, iv, it
    '''
    num_class_i, num_class_v, num_class_t, num_class_iv, num_class_it = num_classes
    list_I = []
    list_V = []
    list_T = []
    list_IV = []
    list_IT = []
    if type(input) == np.ndarray:
        input = torch.from_numpy(input)
    list_IVT = convert_label_format(input, input.shape[1], list_to_tensor=False)
    # list_IVT: [[triplet_idx(int)], [triplet_idx * n], [...], ...]
    # l_IVT: [triplet_idx * n]
    for l in list_IVT:
        l_I = []
        l_V = []
        l_T = []
        l_IV = []
        l_IT = []
        for ivt in l:
            i, v, t, iv, it = split_single_ivt(ivt, mappingFilePath) #int
            l_I.append(i)
            l_V.append(v)
            l_T.append(t)
            l_IV.append(iv)
            l_IT.append(it)
        list_I.append(l_I)
        list_V.append(l_V)
        list_T.append(l_T)
        list_IV.append(l_IV)
        list_IT.append(l_IT)
    tensor_I = convert_label_format(list_I, num_class_i)
    tensor_V = convert_label_format(list_V, num_class_v)
    tensor_T = convert_label_format(list_T, num_class_t)
    tensor_IV = convert_label_format(list_IV, num_class_iv)
    tensor_IT = convert_label_format(list_IT, num_class_it)
    return [tensor_I, tensor_V, tensor_T, tensor_IV, tensor_IT]


def get_convert_matrix(mappingFilePath, comp_in, comp_out):
    
    assert(comp_in in ['i', 'v', 't', 'ivt', 'iv', 'it'])
    assert(comp_out in ['i', 'v', 't', 'ivt', 'iv', 'it'])

    num_classes = {'i':6, 'v':10, 't':15, 'ivt':100, 'iv':26, 'it':59}
    col_ids = {'i':1, 'v':2, 't':3, 'ivt':0, 'iv':4, 'it':5}
    
    mapping = np.loadtxt(mappingFilePath, delimiter=',')

    mapping = remap_iv_it(mapping)

    nc_in = num_classes[comp_in]
    nc_out = num_classes[comp_out]
    
    ci_in = col_ids[comp_in]
    ci_out = col_ids[comp_out]

    matrix = np.zeros((nc_in, nc_out))

    for row in mapping:
        matrix[int(row[ci_in]),int(row[ci_out])] = 1
        
    return matrix

def remap_iv_it(mapping):

    iv_ids = np.unique(mapping[:,4])
    iv_ids.sort()
    iv_mapping = {int(iv_ids[i]): i for i in range(len(iv_ids))}

    it_ids = np.unique(mapping[:,5])
    it_ids.sort()
    it_mapping = {int(it_ids[i]): i for i in range(len(it_ids))}

    for i in range(mapping.shape[0]):
        mapping[i,4] = iv_mapping[int(mapping[i,4])]
        mapping[i,5] = it_mapping[int(mapping[i,5])]
        
    return mapping

############# Mapping Reader for CholecT50 ###################
def split_single_ivt(input, mappingFilePath):
    '''
    input: a triplet_idx
    output: i, v, t, iv, it

    note: if input's type is string, output's also string. Same when input's int
    If the input can't be found on mappingFile, return -1, -1, -1, -1, -1
    '''
    mapping = np.loadtxt(mappingFilePath, delimiter=',')
        
    mapping = remap_iv_it(mapping)
    
    if input == -1:
        return -1, -1, -1, -1, -1
    
    for line in mapping:
        
        parts = [str(int(i)) for i in line]
        
        assert(len(parts) == 6)

        if int(parts[0]) == input:
            i, v, t, iv, it = parts[1], parts[2], parts[3], parts[4], parts[5]
            return int(i), int(v), int(t), int(iv), int(it)
        elif parts[0] == input:
            i, v, t, iv, it = parts[1], parts[2], parts[3], parts[4], parts[5]
            return i, v, t, iv, it

    raise Exception('Invalid Triplet')
    return 

def index_name_converter(input, ivt, root_data_dir): # read from origical data from the dataset
    '''
    input: string, then output index; index, then output string
    ivt: "instrument", "verb", "target", "triplet"
    root_data_dir: the CholecT50 dataset root dir

    note: if not found, return -1 or None
    '''
    assert(ivt in ["instrument", "verb", "target", "triplet"])

    if type(input) == str:
       with open(os.path.join(root_data_dir,"labels","VID01.json")) as mappingFile:
            content = json.load(mappingFile)
            for x, y in content["categories"].items():
              if x == ivt:
                for k,v in y.items():
                    if v == input:
                        return k
                return -1
    elif type(input) == int:
        with open(os.path.join(root_data_dir, "labels", "VID01.json")) as mappingFile:
            content = json.load(mappingFile)
            for x, y in content["categories"].items():
              if x == ivt:
                for k, v in y.items():
                    if int(k) == input:
                        return v
                return None

# ############ Visualization #################

def plot_barcode(class_num, gt=None, pred=None, show=True, save_file=None):

    if class_num <= 10:
        color_map = plt.cm.tab10
    elif class_num > 20:
        color_map = plt.cm.gist_ncar
    else:
        color_map = plt.cm.tab20

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, 
                interpolation='nearest', vmin=0, vmax=class_num-1)

    fig = plt.figure(figsize=(18, 4))

    # a horizontal barcode
    if gt is not None:
        ax1 = fig.add_axes([0, 0.45, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth')
        ax1.imshow(gt.reshape((1, -1)), **barprops)

    if pred is not None:
        ax2 = fig.add_axes([0, 0.15, 1, 0.2], **axprops)
        ax2.set_title('Predicted')
        ax2.imshow(pred.reshape((1, -1)), **barprops)

    if save_file is not None:
        fig.savefig(save_file, dpi=400)
    if show:
        plt.show()

    plt.close(fig)
