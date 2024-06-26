import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def GSNN_TR2(TR,TAR,VL,TARV,NurNum,EpcNum,reps,cmp_level):
    net0 = tf.keras.Sequential([
        tf.keras.layers.Dense(NurNum, activation='relu'),
        tf.keras.layers.Dense(NurNum, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    indp = np.where(TAR == 1)[0]
    indn = np.where(TAR == 0)[0]
    slctnum = round(min(len(indn)/300, len(indp)))
    indpV = np.where(TARV == 1)[0]
    indnV = np.where(TARV == 0)[0]
    slctnumV = round(min(len(indnV)/300, len(indpV)))
    TRo = []
    VLo = []
    TR2indx = []
    j1 = 0
    down_flag = 0
    Net = []
    while j1 < TR.shape[0] and down_flag == 0:
        j1 = j1 + 1
        AUCs = np.zeros((reps, TR.shape[0]))
        for k1 in range(reps):
            slct_pos = np.random.choice(indp, size=round(slctnum), replace=True)
            slct_neg = np.random.choice(indn, size=round(slctnum), replace=True)
            slct = np.concatenate([slct_pos, slct_neg])
            np.random.shuffle(slct)
            r = np.random.rand() * 1000
            for j2 in range(TR.shape[0]):
                np.random.seed(int(j1 + r))
                if j2 not in TR2indx:
                    tr = np.vstack([TRo, TR[j2]])
                    net = tf.keras.models.clone_model(net0)
                    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    net.fit(TR[:, slct], TAR[slct], epochs=EpcNum, verbose=0)
                    Net.append(net)
            np.random.seed(r)
            auc = np.zeros((1,TR.shape[0]));
            slct_pos = np.random.choice(indpV, size=round(slctnumV), replace=True)
            slct_neg = np.random.choice(indnV, size=round(slctnumV), replace=True)
            slctV = np.concatenate([slct_pos, slct_neg])
            np.random.shuffle(slctV)
            r = np.random.rand() * 1000
            for j2 in range(TR.shape[0]):
                np.random.seed(int(j1 + r))
                if j2 not in TR2indx:
                    net = Net[j2]
                    tr = np.vstack([VLo, VL[j2]])
                    auc[j2] = roc_auc_score(TARV[slctV], net(tr[:,slctV]))
            np.random.seed(r)
            AUCs[k1] = auc
        auc = np.max(AUCs)
        imax = np.argmax(auc)
        vmax = AUCs[imax]
        TRo = np.vstack([TRo, TR[imax]])
        VLo = np.vstack([VLo, VL[imax]])
        TR2indx[j1] = imax
        AUC[j1] = vmax
        imax = np.argmax(AUC)
        vmax = AUC[imax]
        minf = 2
        if imax <= j1-5 and j1 >= minf:
            down_flag = 1
    imax = np.argmax(AUC)
    vmax = AUC[imax]
    imax = max(imax,minf)
    TR2indx = TR2indx[:imax]
    return TR2indx