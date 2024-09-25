import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import warnings
warnings.filterwarnings("ignore")



ALL_data = dict(
    #
    BMNC = {1: 'BMNC', 2: 'd1', 'N': 30672, 'K': 27, 'V': 2, 'n_input': [1000,25], 'n_hid': [10,256], 'n_output': 64},
# PBMC = {1: 'PBMC', 2: 'd1', 'N': 3762, 'K': 16, 'V': 2, 'n_input': [1000,49], 'n_hid': [10,256], 'n_output': 64},
# SLN111 = {1: 'SLN111', 2: 'd1', 'N': 16828, 'K': 35, 'V': 2, 'n_input': [1000,112], 'n_hid': [10,256], 'n_output': 64},
# SMAGE3 = {1: 'SMAGE3', 2: 'd1', 'N': 2585, 'K': 14, 'V': 2, 'n_input': [2000,2000], 'n_hid': [10,256], 'n_output': 64},
# SMAGE10 = {1: 'SMAGE10', 2: 'd1', 'N': 11020, 'K': 12, 'V': 2, 'n_input': [2000,2000], 'n_hid': [10,256], 'n_output': 64},
# BMNC_generated = {1: 'BMNC_generated', 2: 'd1', 'N': 30672, 'K': 27, 'V': 2, 'n_input': [1000,25], 'n_hid': [10,256], 'n_output': 64},
    )


path = './datasets/'


def load_data(dataset):
    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])
    # 将数据缩放到给定的范围（默认为[0,1]）
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)

    size = len(Y[0])
    view_num = len(X)

    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])

    return X, Y
#
#
#
#
#
#
# import numpy as np
# import torch
# from sklearn.preprocessing import MinMaxScaler
# import scipy.io
# import warnings
# import os
#
# warnings.filterwarnings("ignore")
#
# ALL_data = dict(
#     BMNC_generated={1: 'BMNC_generated', 2: 'd1', 'N': 30672, 'K': 27, 'V': 2, 'n_input': [1000, 25],
#                     'n_hid': [10, 256], 'n_output': 64},
# )
#
# path = './datasets/'
#
#
# def load_data(dataset):
#     file_path = os.path.join(path, dataset[1] + ".mat")
#
#     # 检查文件是否存在
#     if not os.path.exists(file_path):
#         print(f"文件 {file_path} 不存在，请检查路径和文件名。")
#         return None, None
#
#     print(f"文件路径正确，正在加载文件: {file_path}")
#
#     try:
#         data = scipy.io.loadmat(file_path)
#         print("加载的文件内容键:", data.keys())
#     except Exception as e:
#         print(f"无法打开文件: {e}")
#         return None, None
#
#     X = []
#     Y = []
#     Label = data['Y'].flatten()
#
#     # 打印数据维度
#     print("Y (标签) 的维度:", Label.shape)
#
#     # 将数据缩放到给定的范围（默认为[0,1]）
#     mm = MinMaxScaler()
#
#     for key in ['X1', 'X2']:
#         if key in data:
#             diff_view = data[key]
#             diff_view = np.array(diff_view, dtype=np.float32)  # 不转置
#             std_view = mm.fit_transform(diff_view)
#             X.append(std_view)
#             print(f"{key} 的维度:", std_view.shape)
#         else:
#             print(f"键 '{key}' 不存在于加载的文件中，请检查文件内容。")
#             return None, None
#
#     size = Label.shape[0]
#     view_num = len(X)
#
#     index = np.arange(size)
#     np.random.shuffle(index)
#     Label = Label[index]
#
#     for v in range(view_num):
#         X[v] = X[v][index, :]  # 对行进行索引
#
#     for v in range(view_num):
#         X[v] = torch.from_numpy(X[v])
#
#     return X, Label


# 示例用法
# dataset = ALL_data['BMNC_generated']
# X, Y = load_data(dataset)
# if X is not None and Y is not None:
#     print("数据加载成功")
# else:
#     print("数据加载失败")




