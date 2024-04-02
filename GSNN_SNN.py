import numpy as np
from scipy.fftpack import dct, idct
import tensorflow as tf
import math
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def GSNN_SNN(TR,TAR,VL,TARV,TST,TART,S1,S1V,MN,MX,SNN_iterations):
    G = []
    for j in range(TR.shape[0]):
        mn = np.min(TR[j])
        mx = np.max(TR[j])
        rn = mx - mn
        G.append(np.linspace(mn, mx, num=10001, endpoint=True)[1:-1])

    trL = len(TAR)
    TR = np.concatenate((TR, VL), axis=1)
    TAR = np.concatenate((TAR, TARV), axis=1)
    S1 = np.concatenate((S1, S1V), axis=1)
    indp = np.where(TAR == 1)[0]
    indn = np.where(TAR == 0)[0]
    slctnum = round(min(len(indn) / 300, len(indp)))
    indpV = np.where(TARV == 1)[0]
    indnV = np.where(TARV == 0)[0]
    slctnumV = round(min(len(indnV) / 300, len(indpV)))

    TARo = TAR
    indp = np.where(TAR==1)[0]
    indn = np.where(TAR==0)[0]

    S1 = S1 - np.min(S1)
    taro = np.power(S1, 0.25)
    bl = 0
    df = 0.1
    tar = taro - bl
    tars = np.zeros((9, len(tar)))

    for j4 in range(1, 10):
        tard = np.fft.dct(tar)
        tard[round(len(tard) * (df + 1.5 * (j4 - 5) / 100)):] = 0
        tars[j4-1, :] = np.fft.idct(tard)

    f = np.zeros_like(TR)
    fV = np.zeros_like(VL)
    fT = np.zeros_like(TST)
    fG = np.zeros_like(G)
    Ss = bl * np.ones(TR.shape[1])
    nn = 2
    ne = 2
    net0 = tf.keras.Sequential([
        tf.keras.layers.Dense(nn, activation='radial_basis'),
        tf.keras.layers.Dense(nn, activation='radial_basis'),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    net0.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    net0.fit(TR, TAR, epochs=ne, verbose=0)
    auc2 = []
    net2 = []
    auc = []
    aucV = []
    aucT = []
    fop = []
    fVop = []
    fTop = []
    fGop = []
    blop = []

    net1 = []
    auc1 = []
    for k2 in range(TR.shape[0]):
        net1.append([])
        auc1.append(-np.inf)
    j = 1
    for j in range(SNN_iterations):
        slcto = np.concatenate((np.random.choice(indp, size=round(slctnum), replace=True),
                                np.random.choice(indn, size=round(slctnum), replace=True)))
        slctoV = np.concatenate((np.random.choice(indpV, size=round(slctnumV), replace=True),
                                np.random.choice(indnV, size=round(slctnumV), replace=True))) + trL
        np.random.shuffle(slcto)
        np.random.shuffle(slctoV)
        slct = np.concatenate((slcto, slctoV), axis=1)
        net2 = []
        auc2 = []
        for k1 in range(9):
            r = np.random.rand()*1000
            tr1 = TR[k2][:slct]
            tar1 = tars[k1][:slct]
            net0 = tf.keras.Sequential([
                tf.keras.layers.Dense(nn, activation='radial_basis'),
                tf.keras.layers.Dense(nn, activation='radial_basis'),
                tf.keras.layers.Dense(1, activation='relu')
            ])
            net0.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            net0.fit(tr1, tar1, epochs=ne, verbose=0)
            ss = np.vstack(Ss, TR[k2])
            net1[k2] = net1
            auc2[k2] = roc_auc_score(TARo, sum(ss))

        inds = np.argmax(auc2)
        vs = auc2[inds]
        auc1 = vs
        for j2 in range(TR.shape[0]):
            nets = net2[inds[j2]]
            net1[j2] = nets[j2]
        vs = np.sort(-auc1)
        inds = np.argsort(-auc1)
        eind2 = inds

        j2 = 1
        netb = net1[eind2[j2]]
        f[eind2[j2]] = f[eind2[j2]]+netb[TR[eind2[j2]]]
        fV[eind2[j2]] = fV[eind2[j2]]+netb[VL[eind2[j2]]]
        fT[eind2[j2]] = fT[eind2[j2]]+netb[TST[eind2[j2]]]
        fG[eind2[j2]] = fG[eind2[j2]]+netb[G[eind2[j2]]]

        Ss = np.sum(f) + bl
        auc.append(roc_auc_score(TARo, sum(ss)))
        auctr = auc[-1]
        tar = taro - Ss
        tars = np.zeros((9, len(tar)))
        df = 0.1

        for j4 in range(1, 10):
            tard = np.fft.dct(tar)
            end_index = int(round(len(tard) * (df + 1.5 * (j4 - 5) / 100)))
            tard[end_index:] = 0
            tars[j4 - 1] = np.fft.idct(tard) 

        tara = TAR - Ss
        SsV = np.sum(fV) + bl
        aucV.append(roc_auc_score(TARV, SsV))
        aucval = aucV[-1]
        SsT = np.sum(fT) + bl
        aucT.append(roc_auc_score(TART, SsT))
        auctst = aucT[-1]

        plt.plot(aucT, 'g')
        plt.grid(True)
        plt.title('Training Progress')
        plt.ylabel('AUC')
        plt.xlabel('Iteration')
        plt.show()

        fop[j] = f
        fVop[j] = fV
        fTop[j] = fT
        fGop[j] = fG
        blop[j] = bl
    
    if j > 1:
        if auc[-1] - auc[-2] <= 0.005:
            nn = nn + 1
            ne = ne + 1
            nn = max(min(nn, 15), 2)
            ne = max(min(ne, 15), 1)
        if auc[-1] - auc[-2] > 0.005:
            nn = nn - 1
            ne = ne - 1
            nn = max(min(nn, 15), 2)
            ne = max(min(ne, 15), 1)
        net0 = tf.keras.Sequential([
            tf.keras.layers.Dense(nn, activation='radial_basis'),
            tf.keras.layers.Dense(nn, activation='radial_basis'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
    j = j + 1
    ind = np.argmax(aucV)
    v = aucV[ind]

    f = fop[ind]
    fV = fVop[ind]
    fT = fTop[ind]
    fG = fGop[ind]
    bl = blop[ind]

    f0 = f
    ss = np.zeros(TR.shape[1])
    auc1 = np.zeros(TR.shape[0])
    aucb = np.zeros(TR.shape[0])
    eind = np.zeros(TR.shape[0])
    for j in range(TR.shape[0]):
        for j2 in range(TR.shape[0]):
            auc1[j2] = roc_auc_score(TAR, np.sum(np.vstack((ss, f0[j2])), axis=0))
        indm = np.argmax(auc1)
        vm = auc1[indm]
        aucb[j1] = vm
        eind[j1] = indm
        ss = ss + f0[indm]
        f0[indm] = 0
    indm = np.argmax(aucb)
    vm = aucb[indm]
    eind = eind[:indm]

    f = f[eind, :]
    fV = fV[eind, :]
    fT = fT[eind, :]
    fG = fG[eind, :]
    G = G[eind, :]
    TR = TR[eind, :]
    VL = VL[eind, :]
    TST = TST[eind, :]
    MN = MN[eind]
    MX = MX[eind]
    ranks = eind

    f0 = f.copy()
    GM = np.zeros((TR.shape[0],))
    for j in range(TR.shape[0]):
        GM[j] = np.min(f[j])
        fT[j] -= np.min(f[j])
        fG[j] -= np.min(f[j])
        fV[j] -= np.min(f[j])
        f[j] -= np.min(f[j])

    GM = np.sum(GM) + bl
    GMx = np.max(np.sum(f))

    fT = fT / GMx
    fG = fG / GMx
    fV = fV / GMx
    f = f / GMx

    S1b = np.sum(f)*GMx+GM
    S1bV = np.sum(fV)*GMx+GM
    S1bT = np.sum(fT)*GMx+GM
    AUCtr = roc_auc_score(TAR, S1b)
    AUCval = roc_auc_score(TARV, S1bV)
    AUCtst = roc_auc_score(TART, S1bT)

    MNG = 0
    a = np.zeros((f.shape[0], 30))
    b = np.zeros_like(a)
    w = np.zeros_like(a)
    c = np.zeros((f.shape[0], 1))
    f2 = []
    f2V = []
    f2T = []

    for j in range(eind):
        cnd = 0
        cnt = 5
        net = None
        net = tf.keras.Sequential([
                tf.keras.layers.Dense(cnt, activation='linear'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
        while (cnd == 0) and (cnt <= 30):
            net = tf.keras.Sequential([
                tf.keras.layers.Dense(cnt, activation='radial_basis'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
            net.compile(optimizer='adam', loss='mse')
            net.fit(G[j], fG[j], epochs=100, verbose=0)
            if np.mean(np.abs(f[j] - net.predict(TR[j:j+1]))) < 1e-4:
                cnd = 1
            else:
                cnt += 5
        cnt = min(cnt, 30)

        f2.append(net.predict(TR[j]))
        f2V.append(net.predict(VL[j]))
        f2T.append(net.predict(TST[j]))

        tmp = np.zeros(30)
        tmp[:cnt] = net.layers[0].get_weights()[0][:, 0]
        a[j] = tmp
        tmp[:cnt] = net.layers[0].get_weights()[1]
        b[j] = tmp - a[j] * MN[j] / MX[j] - a[j] * MNG
        a[j] /= MX[j]
        c[j] = net.layers[1].get_weights()[1][0]
        tmp[:cnt] = net.layers[1].get_weights()[0][0, :]
        w[j] = tmp
    
    SNN.a = a
    SNN.b = b
    SNN.w = w
    SNN.c = c

    Go = G * np.tile(MX, (G.shape[1], 1)).T + np.tile(MN, (G.shape[1], 1)).T

    def radbas(x):
        return np.exp(-x ** 2)

    f2G = np.squeeze(np.sum(np.tile(w.T, (1, 1, Go.shape[2])) * \
                            radbas(np.tile(a.T, (1, 1, Go.shape[2])) * \
                                    np.transpose(np.tile(Go, (1, 1, a.shape[1])), (2, 0, 1)) + \
                                    np.tile(b.T, (1, 1, Go.shape[2]))), axis=1)) + \
        np.tile(c, (1, Go.shape[2]))
    
    S2T = np.sum(f2T) * GMx + GM
    S2V = np.sum(f2V) * GMx + GM
    S2 = np.sum(f2) * GMx + GM 
    AUCtr = roc_auc_score(TAR, S2)
    AUCval = roc_auc_score(TARV, S2V)
    AUCtst = roc_auc_score(TART, S2T)

    print('SNN AUC:', round(AUCtst, 4))

    return SNN, ranks