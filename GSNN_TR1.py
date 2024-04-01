import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def GSNN_TR1(TR,TAR,VL,TARV,NurNum,EpcNum,reps,thr):
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
    r = np.random.rand() * 1000
    Net = []
    SLCT2 = []
    for j1 in range(reps):
        np.random.seed(int(j1 + r))
        slct_pos = np.random.choice(indp, size=slctnum, replace=True)
        slct_neg = np.random.choice(indn, size=slctnum, replace=True)
        slct = np.concatenate([slct_pos, slct_neg])
        np.random.shuffle(slct)
        slct2 = np.random.choice(TR.shape[0], size=3, replace=False)
        net = tf.keras.models.clone_model(net0)
        net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        net.fit(TR[slct2][:, slct], TAR[slct], epochs=EpcNum, verbose=0)

        Net.append(net)
        SLCT2.append(slct2)
    np.random.seed(r)
    
    Dif = np.zeros((reps, TR.shape[0]))
    Div = np.zeros((reps, TR.shape[0]))
    for j1 in range(reps):
        slct2 = SLCT2[j1]
        net = Net[j1]
        slct_pos = np.random.choice(indpV, size=round(slctnumV), replace=True)
        slct_neg = np.random.choice(indnV, size=round(slctnumV), replace=True)
        slctV = np.concatenate([slct_pos, slct_neg])
        np.random.shuffle(slctV)
        tr = VL[slct2][:, slctV]
        auco = roc_auc_score(TARV[slctV], net(tr))

        auc = np.zeros((1, len(slct2)))
        for j2 in range(len(slct2)):
            trm = tr.copy()
            trm[j2] = trm[j2,np.random.choice(trm.shape[1])]
            auc[j2] = roc_auc_score(TARV[slctV], net(trm))
        Dif[j1][:slct2] = auco - auc
        Div[j1][:slct2] = 1
        TR1indx = np.where(sum(Dif)/sum(Div)>thr)[0]
        return TR1indx
    