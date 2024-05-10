import numpy as np

npy_path = 'path_to_your_file.npy'

def read_npy(path):
    """
    读取指定路径的 NumPy 文件并打印内容。
    
    参数:
    path (str): NumPy 文件的路径。
    """
    try:
        data = np.load(path)
        print("NumPy 文件内容:")
        print(data)
    except Exception as e:
        print(f"读取文件出错: {e}")

read_npy(npy_path)