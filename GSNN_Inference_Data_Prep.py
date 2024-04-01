import scipy.io as sio
import numpy as np


def GSNN_Inference_Data_Prep(cmp_lvl,namesR):
    data = sio.loadmat('/content/GSNN_Demo_Data.mat')
    nameID = np.arange(0, len(data['feature_names']))
    np.random.seed(100)


    # create composite features
    jj = data['geo_features'].shape[0]
    if cmp_lvl == 2:
        combs = [(idx1, idx2) for idx1 in range(len(data['geo_features'])) for idx2 in range(idx1 + 1, len(data['geo_features']))]
        data['geo_features'][data['geo_features'].shape[0]+combs.shape[0],0,0] = 0
        for j in range(combs.shape[0]):
            jj = jj+1
            data['geo_features'][jj] = data['geo_features'][combs[j,0]] * data['geo_features'][combs[j,1]]
        for j in range(combs.shape[0]):
            data['feature_names'].append(f"{data['feature_names'][combs[j,0]]} & {data['feature_names'][combs[j,1]]}")
            nameID.append(f"{nameID[combs[j,0]]} {nameID[combs[j,1]]}")

    # truncate outliers
    Fc = []
    mask = np.where(data['geo_exclude']==1)
    F = np.array(data['geo_features'][mask], dtype=float)
    for j in range(F.shape[0]):
        mean_val = np.mean(F[j])
        std_val = np.std(F[j])
        Fc.append(np.clip(F[j], mean_val - 6 * std_val, mean_val + 6 * std_val))

    inds = np.zeros(len(namesR), dtype=int)
    for j1, nameR in enumerate(namesR):
        for j2, feature_name in enumerate(data['feature_names']):
            if feature_name == nameR:
                inds[j1] = j2

    Features = Fc[inds]
    return Features