import numpy as np

def min_max_norm(data:np.array, verbose=False):
    # x - min / max-min
    # data.shape: N, E. N:Num of Features, E:Embedding size
    # 将min、max扩充到data维度，然后相减，相除. 直接加减乘除都是按位加减乘除。
    index_min = data.min(axis=0) # shape: E
    index_max = data.max(axis=0) # shape: E
    data = data - np.tile(index_min, (data.shape[0], 1)) # x-min
    data = data / np.tile(index_max-index_min, (data.shape[0], 1)) # x-min/max-min
    if verbose:
        return data, index_max-index_min, index_min
    else:
        return data


if __name__ == '__main__':
    a = np.array([[1,2,3],[-1,0,0]])
    print(min_max_norm(a))
    print("")
    