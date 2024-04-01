import scipy.io as sio
import numpy as np
import math
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from itertools import combinations

def GSNN_Data_Prep(cmp_lvl):
    data = sio.loadmat('/content/GSNN_Demo_Data.mat')
    nameID = np.arange(0, len(data['feature_names']))
    np.random.seed(100)

    # create Pythagorean-tiling mask
    tsz = max(data['geo_exclude'].shape)
    ssz = 13
    tsz = math.ceil(tsz / ssz) * ssz
    indX = np.arange(0, tsz, ssz)
    indX = np.tile(indX, (len(indX), 1))
    M = np.zeros((tsz, tsz))
    for j in range(indX.shape[0]):
        for k in range(j % 2, indX.shape[1], 2):
            M[indX[j][0]:indX[j][0] + ssz - 1 + 3, indX[k][0]:indX[k][0] + ssz - 1 + 3] = 1
    M = M[0:data['geo_exclude'].shape[0], 0:data['geo_exclude'].shape[1]]
    M = M + 1

    # select random samples for validation
    tmp = np.zeros((M.shape[0], M.shape[1]))
    selected_indices = np.random.choice(tmp.size, int(round(0.35 * tmp.size)), replace=False)
    tmp.flat[selected_indices] = 1
    tmp = tmp * (M == 1)
    M[tmp==1] = 3
    se = np.ones((7, 7))
    dilated_geo_target = binary_dilation(data['geo_target'], structure=se)
    mask = (dilated_geo_target - data['geo_target']) == 1
    geo_exclude1 = data['geo_exclude'].copy()
    geo_exclude1[mask] = 0
    M = M * geo_exclude1

    print(f"Training: {round(100 * np.sum(M[:] == 2) / np.sum(M[:] > 0), 4)}%")
    print(f"Validation: {round(100 * np.sum(M[:] == 3) / np.sum(M[:] > 0), 4)}%")
    print(f"Testing: {round(100 * np.sum(M[:] == 1) / np.sum(M[:] > 0), 4)}%")

    # study area visualization
    vis_elev = data['geo_features'][0, :, :]
    vis_elev[data['geo_exclude'] == 0] = np.nan
    vis_elev_resized = np.array(Image.fromarray(vis_elev).resize((int(vis_elev.shape[1] * 0.5), int(vis_elev.shape[0] * 0.5))))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(range(vis_elev_resized.shape[1]), range(vis_elev_resized.shape[0]))
    ax.plot_surface(x, y, vis_elev_resized, cmap='spring', shade=False)
    ax.set_title('Study Area')
    ax.set_axis_off()
    ax.view_init(45, 45)
    plt.show()

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
    F = np.array(data['geo_features'], dtype=float)
    T = np.array(data['geo_target'] != 0, dtype=float)
    for j in range(F.shape[0]):
        mean_val = np.mean(F[j])
        std_val = np.std(F[j])
        Fc.append(np.clip(F[j], mean_val - 6 * std_val, mean_val + 6 * std_val))

    for j in range(F.shape[0]):
        F[j] = Fc[j]

    # data partitioning (Training/Validation/Testing)
    TR = F[:, M == 2]
    VL = F[:, M == 3]
    TST = F[:, M == 1]
    TAR = T[M == 2].T
    TARV = T[M == 3].T
    TART = T[M == 1].T

    # data normalization
    F = TR.copy()
    MN = np.min(F, axis=1)
    F = F - MN[:, np.newaxis]
    MX = np.max(F, axis=1)
    F = F / MX[:, np.newaxis]
    TR = F

    F = VL.copy()
    F = F - MN[:, np.newaxis]
    F = F / MX[:, np.newaxis]
    VL = F

    F = TST.copy()
    F = F - MN[:, np.newaxis]
    F = F / MX[:, np.newaxis]
    TST = F

    return TR,TAR,VL,TARV,TST,TART,data['feature_names'],nameID,MN,MX