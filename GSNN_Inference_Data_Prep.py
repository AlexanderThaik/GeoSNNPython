import scipy.io as sio
import numpy as np


def GSNN_Inference_Data_Prep(cmp_lvl,namesR):
  data = sio.loadmat('/content/GSNN_Demo_Data.mat')
  nameID = np.arange(0, len(data['feature_names']))
  np.random.seed(42)


  # create composite features
  if cmp_lvl == 2:
    combs = [(idx1, idx2) for idx1 in range(len(data['geo_features'])) for idx2 in range(idx1 + 1, len(data['geo_features']))]

    data['geo_features'] = np.concatenate((data['geo_features'], np.zeros((len(combs), 1, 1))), axis=0)

    for idx, (idx1, idx2) in enumerate(combs):
        data['geo_features'][idx + len(data['geo_features']) - len(combs)] = data['geo_features'][idx1] * data['geo_features'][idx2]

        data['feature_names'].append(f"{data['feature_names'][idx1]} & {data['feature_names'][idx2]}")
        nameID.append(f"{nameID[idx1]} {nameID[idx2]}")

  # truncate outliers
  Fc = []
  F = np.array(data['geo_features'], dtype=float)
  for j in range(F.shape[0]):
      mean_val = np.mean(F[j])
      std_val = np.std(F[j])
      Fc.append(np.clip(F[j], mean_val - 6 * std_val, mean_val + 6 * std_val))

  inds = np.zeros(len(namesR), dtype=int)
  for j1, nameR in enumerate(namesR):
      for j2, feature_name in enumerate(data['feature_names']):
          if feature_name == nameR:
              inds[j1] = j2

  Features = Fc[inds, :]
  return Features