import os
import numpy as np

# Global Settings


# Functions
def _get_best_epochs(results, epochs, window_size):
    # results: (epoch_num, )

    max_value = 0
    max_index = -1
    for o in range(len(epochs)-window_size+1):
        if results[o:o+window_size].mean() > max_value:
            max_value = results[o:o+window_size].mean()
            max_index = o
        
    return epochs[max_index:max_index+window_size], max_value


def get_best_epochs(result_dir, prefix, suffix, mode, subset, repeat_ids, split_ids, epochs, window_size, result_key='AP_IVT'):

    results = np.zeros((len(repeat_ids), len(split_ids), len(epochs)))

    for r_id, repeat_id in enumerate(repeat_ids):
        for s_id, split_id in enumerate(split_ids):
            naming = f'{prefix}-S{split_id}-{repeat_id}{suffix}'
            for e_id, epoch in enumerate(epochs):
                result_file = os.path.join(result_dir, naming, f'{subset}_results_{mode}_epoch{epoch}.npy')
                try:
                    result = np.load(result_file, allow_pickle=True).item()
                    results[r_id, s_id, e_id] = result[result_key]
                except Exception as exception:
                    results[r_id, s_id, e_id] = 0
                    print(exception, result_file)

    results = results.mean(0).mean(0)
    best_epochs, best_value = _get_best_epochs(results, epochs, window_size)
    return best_epochs, best_value


def get_result(result_dir, prefix, suffix, mode, subset, repeat_ids, split_ids, epochs, result_key='AP_IVT'):

	results = np.zeros((len(repeat_ids), len(split_ids), len(epochs)))

	for r_id, repeat_id in enumerate(repeat_ids):
		for s_id, split_id in enumerate(split_ids):
			naming = f'{prefix}-S{split_id}-{repeat_id}{suffix}'
			for e_id, epoch in enumerate(epochs):
				result_file = os.path.join(result_dir, naming, f'{subset}_results_{mode}_epoch{epoch}.npy')
				try:
					result = np.load(result_file, allow_pickle=True).item()
					results[r_id, s_id, e_id] = result[result_key]
				except Exception as exception:
					results[r_id, s_id, e_id] = np.nan
					print(exception, result_file)

	results = results * 100

	return results.mean(), results.mean(2).std(1).mean(0)


# To add STD summary (STD cross splits?)
""
print('All Same: Select on Test')

result_dir = '/g/data/zg12/dir_Axel/DA_result'
mode = 'decoder-agg'
repeat_ids = [0]
split_ids = [1, 2, 3, 4, 5]
# split_ids = [1]
epochs = [i for i in range(100, 1101, 100)]
# epochs = [400, 500]
window_size = 2

used_prefix='RDV-T45OneSmo5'

naming_paris = [
	[used_prefix, '-ant_range-1'],
	[used_prefix, '-ant_range-2'],
	[used_prefix, '-ant_range-3'],
	[used_prefix, '-ant_range-4'],
	[used_prefix, '-ant_range-5'],
	[used_prefix, '-ant_range-6'],
	[used_prefix, '-ant_range-7'],
	[used_prefix, '-ant_range-8'],
	[used_prefix, '-ant_range-9'],
	[used_prefix, '-ant_range-10'],
	[used_prefix, '-ant_range-12'],
	[used_prefix, '-ant_range-14'],
	[used_prefix, '-ant_range-16'],
	[used_prefix, '-ant_range-18'],
	[used_prefix, '-ant_range-20'],
	[used_prefix, '-ant_range-25'],
	[used_prefix, '-ant_range-30'],
	[used_prefix, '-ant_range-40'],
	[used_prefix, '-ant_range-50'],
	[used_prefix, '-ant_range-75'],
	[used_prefix, '-ant_range-100'],
	[used_prefix, '-ant_range-200'],
]

all_results_aps = {}
all_results_topns = {}

for prefix, suffix in naming_paris:
	selected_epochs, selected_value = get_best_epochs(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, epochs, window_size, result_key='AP_IVT')
	print(selected_epochs)

	AP_I_mean, AP_I_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='AP_I')
	AP_V_mean, AP_V_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='AP_V')
	AP_T_mean, AP_T_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='AP_T')
	AP_IV_mean, AP_IV_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='AP_IV')
	AP_IT_mean, AP_IT_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='AP_IT')
	AP_IVT_mean, AP_IVT_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='AP_IVT')
	
	result_line1 = str(AP_IVT_mean)
	# result_line1 = f'{AP_I_mean:.1f}±{AP_I_std:.1f} & {AP_V_mean:.1f}±{AP_V_std:.1f} & {AP_T_mean:.1f}±{AP_T_std:.1f} & {AP_IV_mean:.1f}±{AP_IV_std:.1f} & {AP_IT_mean:.1f}±{AP_IT_std:.1f} & {AP_IVT_mean:.1f}±{AP_IVT_std:.1f}'
	all_results_aps[''.join([prefix, suffix])] = result_line1

	Top1_mean, Top1_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='Top-1')
	Top5_mean, Top5_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='Top-5')
	Top10_mean, Top10_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='Top-10')
	Top20_mean, Top20_std = get_result(result_dir, prefix, suffix, mode, 'test', repeat_ids, split_ids, selected_epochs, result_key='Top-20')

	result_line2 = ''
	# result_line2 = f'{Top1_mean:.1f}±{Top1_std:.1f} & {Top5_mean:.1f}±{Top5_std:.1f} & {Top10_mean:.1f}±{Top10_std:.1f} & {Top20_mean:.1f}±{Top20_std:.1f}'
	all_results_topns[''.join([prefix, suffix])] = result_line2


for key in all_results_aps.keys():
	print(key)
	print(all_results_aps[key])
	print(all_results_topns[key])
	print('')


