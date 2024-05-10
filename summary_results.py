import os
import numpy as np

# Global Setting
platform_used = "google cloud"
prefix_used = "RDV-T45OneSmo5"

# Detailed Setting
if platform_used == "NCI":
    result_dir = '/g/data/zg12/dir_Axel/DA_result'
elif platform_used == "google cloud":
    result_dir = "first_layer/results/"

repeat_ids = [0] # corresponding to "generate_configs.py"
split_ids = [1, 2, 3, 4, 5] # corresponding to "generate_configs.py"
    
# Default Setting
mode = 'decoder-agg'
epochs = [i for i in range(100, 1101, 100)]
window_size = 2

# Functions
def get_best_epoch(results, epochs, window_size):
    # results: (epoch_num, )

    max_value = 0
    max_index = -1
    for o in range(len(epochs)-window_size+1):
        if results[o:o+window_size].mean() > max_value:
            max_value = results[o:o+window_size].mean()
            max_index = o
        
    return epochs[max_index:max_index+window_size], max_value

def read_file_and_get_best_epoch(file_path, epochs, window_size):
    results = np.zeros(len(epochs))
    
    