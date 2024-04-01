import numpy as np
from scipy.fftpack import dct, idct
import tensorflow as tf
import math
from sklearn.metrics import roc_auc_score

def GSNN_MST(TR,TAR,VL,TARV,TST,TART):
    TRo = TR.copy()
    TARo = TAR.copy()
    trL = len(TAR)
    vlL = len(TARV)
    tstL = len(TART)
    indp = np.where(TAR == 1)[0]
    indn = np.where(TAR == 0)[0]
    slctnum = round(min(len(indn) / 300, len(indp)))
    indpV = np.where(TARV == 1)[0]
    indnV = np.where(TARV == 0)[0]
    slctnumV = round(min(len(indnV) / 300, len(indpV)))
    TR = np.concatenate((TR, VL), axis=1)
    TAR = np.concatenate((TAR, TARV), axis=1)

    reps = 30
    r = np.random.rand() * 1000
    Net1a = []
    Slct2a = []
    S1a = np.zeros((reps, len(TR)))
    S1Va = np.zeros((reps, len(VL)))
    S1Ta = np.zeros((reps, len(TST)))
    S1oa = np.zeros((reps, len(TRo)))

    for j1 in range(reps):
        np.random.seed(j1 + r)
        slcto = np.concatenate((np.random.choice(indp, size=round(slctnum), replace=True),
                                np.random.choice(indn, size=round(slctnum), replace=True)))
        slctoV = np.concatenate((np.random.choice(indpV, size=round(slctnumV), replace=True),
                                np.random.choice(indnV, size=round(slctnumV), replace=True))) + trL
        np.random.shuffle(slcto)
        np.random.shuffle(slctoV)
        slct = np.concatenate((slcto, slctoV), axis=1)
        slct2 = np.arange(0, TR.shape[0])

        tr = TR[slct2][:, slct]
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(3, activation='linear'),
            tf.keras.layers.Activation('radial_basis'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(tr, TAR[slct], epochs=3, verbose=0)

        Net1a.append(model)
        Slct2a.append(slct2)
        S1a[j1, :] = model.predict(TR[slct2])
        S1Va[j1, :] = model.predict(VL[slct2])
        S1Ta[j1, :] = model.predict(TST[slct2])
        S1oa[j1, :] = model.predict(TRo[slct2])
     
    np.random.seed(r)
    r = np.random.rand() * 1000
    Net1b = []
    Slct2b = []
    S1b = np.zeros((reps, trL + vlL))
    S1Vb = np.zeros((reps, vlL))
    S1Tb = np.zeros((reps, tstL))
    S1ob = np.zeros((reps, trL))

    for j1 in range(reps):
        np.random.seed(j1 + r)
        slcto = np.arange(trL)
        slctoV = np.arange(trL, trL + vlL)
        taro = TAR[slcto] - S1a[j1][:slcto]
        tarod = dct(taro)
        tarod[int(round(len(tarod) * 0.1)):] = 0
        taro = idct(tarod)
        tarov = TAR[slctoV] - S1a[j1][:slctoV]
        tarovd = dct(tarov)
        tarovd[int(round(len(tarovd) * 0.1)):] = 0
        tarov = idct(tarovd)
        TARb = np.concatenate((taro, tarov), axis=1)
        
        slcto = np.concatenate((np.random.choice(indp, size=round(slctnum), replace=True),
                                np.random.choice(indn, size=round(slctnum), replace=True)))
        
        slctoV = np.concatenate((np.random.choice(indpV, size=round(slctnumV), replace=True),
                                np.random.choice(indnV, size=round(slctnumV), replace=True))) + trL
        
        slcto = np.random.choice(slcto, size=len(slcto), replace=False)
        slctoV = np.random.choice(slctoV, size=len(slctoV), replace=False)
        
        slct = np.concatenate((slcto, slctoV), axis=1)
        slct2 = Slct2a[j1]

        tr = np.vstack((TR[slct2][:slct], S1a[j1][:slct]))
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(3, activation='linear'),
            tf.keras.layers.Activation('radial_basis'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(tr, TARb[slct], epochs=3, verbose=0)

        Net1b.append(model)
        Slct2b.append(slct2)
        S1b[j1] = model.predict(np.vstack((TR[slct2], S1a[j1])))
        S1Vb[j1] = model.predict(np.vstack((VL[slct2], S1Va[j1])))
        S1Tb[j1] = model.predict(np.vstack((TST[slct2], S1Ta[j1])))
        S1ob[j1] = model.predict(np.vstack((TRo[slct2], S1oa[j1])))

    np.random.seed(r)
    r = np.random.rand() * 1000
    Net1c = []
    Slct2c = []
    S1c = np.zeros((reps, trL + vlL))
    S1Vc = np.zeros((reps, vlL))
    S1Tc = np.zeros((reps, tstL))
    S1oc = np.zeros((reps, trL))
    for j1 in range(reps):
        np.random.seed(j1 + r)
        slcto = np.arange(trL)
        slctoV = np.arange(trL, trL + vlL)
        taro = TAR[slcto] - S1a[j1][:slcto] - S1b[j1][:slcto]
        tarod = dct(taro)
        tarod[int(round(len(tarod) * 0.1)):] = 0
        taro = idct(tarod)
        tarov = TAR[slctoV] - S1a[j1][:slctoV] - S1b[j1][:slctoV]
        tarovd = dct(tarov)
        tarovd[int(round(len(tarovd) * 0.1)):] = 0
        tarov = idct(tarovd)
        TARc = np.concatenate((taro, tarov), axis=1)
        slcto = np.concatenate((np.random.choice(indp, size=round(slctnum), replace=True),
                                np.random.choice(indn, size=round(slctnum), replace=True)))
        
        slctoV = np.concatenate((np.random.choice(indpV, size=round(slctnumV), replace=True),
                                np.random.choice(indnV, size=round(slctnumV), replace=True))) + trL
        
        slcto = np.random.choice(slcto, size=len(slcto), replace=False)
        slctoV = np.random.choice(slctoV, size=len(slctoV), replace=False)
        
        slct = np.concatenate((slcto, slctoV), axis=1)
        slct2 = Slct2a[j1]
        tr = np.vstack([TR[slct2][:slct], S1a[j1][:slct], S1b[j1][:slct]])
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(3, activation='linear'),
            tf.keras.layers.Activation('radial_basis'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(tr, TARc[slct], epochs=3, verbose=0)

        Net1c.append(model)
        Slct2c.append(slct2)

        S1c[j1] = model(np.vstack([TR[slct2], S1a[j1], S1b[j1]]))
        S1Vc[j1] = model(np.vstack([VL[slct2], S1Va[j1], S1Vb[j1]]))
        S1Tc[j1] = model(np.vstack([TST[slct2], S1Ta[j1], S1Tb[j1]]))
        S1oc[j1] = model(np.vstack([TRo[slct2], S1oa[j1], S1ob[j1]]))
    np.random.seed(r)
    S1 = np.vstack([S1a, S1b, S1c])
    S1V = np.vstack([S1Va, S1Vb, S1Vc])
    S1T = np.vstack([S1Ta, S1Tb, S1Tc])
    S1o = np.vstack([S1oa, S1ob, S1oc])

    res = np.mean(S1oa + S1ob + S1oc)
    resV = np.mean(S1Va + S1Vb + S1Vc)
    resT = np.mean(S1Ta + S1Tb + S1Tc)
    auctr = []
    aucval = []
    auctst = []

    auctr.append(roc_auc_score(TARo, res))
    aucval.append(roc_auc_score(TARV, resV))
    auctst.append(roc_auc_score(TART, resT))

    reps = 90
    Net2 = []
    S2 = np.zeros((reps, trL + vlL))
    S2V = np.zeros((reps, vlL))
    S2T = np.zeros((reps, tstL))
    S2o = np.zeros((reps, trL))
    for j1 in range(reps):
        slctos = [np.random.choice(indp, size=round(slctnum), replace=True) + 
                    np.random.choice(indn, size=round(slctnum), replace=True) 
                    for _ in range(reps)]
        slctosV = [np.random.choice(indpV, size=round(slctnumV), replace=True) + 
                    np.random.choice(indnV, size=round(slctnumV), replace=True) + 
                    len(slctos[j]) 
                    for j in range(reps)]
    r = np.random.rand() * 1000
    for j1 in range(reps):      
        np.random.seed(j1+r)     
        slct2 = np.random.choice(S1.shape[0], 50, replace=False)
        slcto = slctos[j1]
        slctoV = slctosV[j1]
        np.random.shuffle(slcto)
        np.random.shuffle(slctoV)
        slct = np.concat(slcto, slctoV)
        tr = S1[slct2][:slct]
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(3, activation='linear'),
            tf.keras.layers.Activation('radial_basis'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(tr, TARc[slct], epochs=3, verbose=0)     
        Net2.append(model)
        S2[j1] = model(S1[slct2])
        S2V[j1] = model(S1V[slct2])
        S2T[j1] = model(S1T[slct2])
        S2o[j1] = model(S1o[slct2])
    np.random.seed(r)
    res = np.mean(S2o)
    resV = np.mean(S2V)
    resT = np.mean(S2T)
    auctr.append(roc_auc_score(TARo, res))
    aucval.append(roc_auc_score(TARV, resV))
    auctst.append(roc_auc_score(TART, resT))

    Net3 = []
    S3 = np.zeros((reps, trL + vlL))
    S3V = np.zeros((reps, vlL))
    S3T = np.zeros((reps, tstL))
    S3o = np.zeros((reps, trL))
    for j1 in range(reps):  
        slctos = [np.random.choice(indp, size=round(slctnum), replace=True) + 
                    np.random.choice(indn, size=round(slctnum), replace=True) 
                    for _ in range(reps)]
        slctosV = [np.random.choice(indpV, size=round(slctnumV), replace=True) + 
                    np.random.choice(indnV, size=round(slctnumV), replace=True) + 
                    len(slctos[j]) 
                    for j in range(reps)]
    r = np.random.rand() * 1000
    for j1 in range(reps):      
        np.random.seed(j1+r)     
        slct2 = np.random.choice(S2.shape[0], 60, replace=False)
        slcto = slctos[j1]
        slctoV = slctosV[j1]
        np.random.shuffle(slcto)
        np.random.shuffle(slctoV)
        slct = np.concat((slcto, slctoV), axis=1)
        tr = S2[slct2][:slct]
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation='linear'),
            tf.keras.layers.Activation('radial_basis'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(tr, TAR[slct], epochs=5, verbose=0)     
        Net3.append(model)
        S3[j1] = model(S2[slct2])
        S3V[j1] = model(S2V[slct2])
        S3T[j1] = model(S2T[slct2])
        S3o[j1] = model(S2o[slct2])
    np.random.seed(r)
    res = np.mean(S3o)
    resV = np.mean(S3V)
    resT = np.mean(S3T)
    auctr.append(roc_auc_score(TARo, res))
    aucval.append(roc_auc_score(TARV, resV))
    auctst.append(roc_auc_score(TART, resT))
    SS = []
    SSV = []
    SST = []
    SSo = []
    SS.append(S3)
    SSV.append(S3V)
    SST.append(S3T)
    SSo.append(S3o)

    Net4 = []
    S4 = np.zeros((reps, trL + vlL))
    S4V = np.zeros((reps, vlL))
    S4T = np.zeros((reps, tstL))
    S4o = np.zeros((reps, trL))
    for j1 in range(reps):  
        slctos = [np.random.choice(indp, size=round(slctnum), replace=True) + 
                    np.random.choice(indn, size=round(slctnum), replace=True) 
                    for _ in range(reps)]
        slctosV = [np.random.choice(indpV, size=round(slctnumV), replace=True) + 
                    np.random.choice(indnV, size=round(slctnumV), replace=True) + 
                    len(slctos[j]) 
                    for j in range(reps)]
    for j1 in range(reps):      
        np.random.seed(j1+r)     
        slct2 = np.random.choice(S3.shape[0], 60, replace=False)
        slcto = slctos[j1]
        slctoV = slctosV[j1]
        np.random.shuffle(slcto)
        np.random.shuffle(slctoV)
        slct = np.concat((slcto, slctoV), axis=1)
        tr = S3[slct2][:slct]
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation='linear'),
            tf.keras.layers.Activation('radial_basis'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(tr, TAR[slct], epochs=5, verbose=0)     
        Net4.append(model)
        S4[j1] = model(S3[slct2])
        S4V[j1] = model(S3V[slct2])
        S4T[j1] = model(S3T[slct2])
        S4o[j1] = model(S3o[slct2])
    np.random.seed(r)
    res = np.mean(S4o)
    resV = np.mean(S4V)
    resT = np.mean(S4T)
    auctr.append(roc_auc_score(TARo, res))
    aucval.append(roc_auc_score(TARV, resV))
    auctst.append(roc_auc_score(TART, resT))
    SS.append(S4)
    SSV.append(S4V)
    SST.append(S4T)
    SSo.append(S4o)

    Net5 = []
    S5 = np.zeros((reps, trL + vlL))
    S5V = np.zeros((reps, vlL))
    S5T = np.zeros((reps, tstL))
    S5o = np.zeros((reps, trL))
    for j1 in range(reps):
        slctos = [np.random.choice(indp, size=round(slctnum), replace=True) + 
                    np.random.choice(indn, size=round(slctnum), replace=True) 
                    for _ in range(reps)]
        slctosV = [np.random.choice(indpV, size=round(slctnumV), replace=True) + 
                    np.random.choice(indnV, size=round(slctnumV), replace=True) + 
                    len(slctos[j]) 
                    for j in range(reps)]
    for j1 in range(reps):      
        np.random.seed(j1+r)     
        slct2 = np.random.choice(S4.shape[0], 60, replace=False)
        slcto = slctos[j1]
        slctoV = slctosV[j1]
        np.random.shuffle(slcto)
        np.random.shuffle(slctoV)
        slct = np.concat((slcto, slctoV), axis=1)
        tr = S4[slct2][:slct]
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation='linear'),
            tf.keras.layers.Activation('radial_basis'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(tr, TAR[slct], epochs=5, verbose=0)     
        Net5.append(model)
        S5[j1] = model(S4[slct2])
        S5V[j1] = model(S4V[slct2])
        S5T[j1] = model(S4T[slct2])
        S5o[j1] = model(S4o[slct2])
    np.random.seed(r)
    res = np.mean(S5o)
    resV = np.mean(S5V)
    resT = np.mean(S5T)
    auctr.append(roc_auc_score(TARo, res))
    aucval.append(roc_auc_score(TARV, resV))
    auctst.append(roc_auc_score(TART, resT))
    SS.append(S5)
    SSV.append(S5V)
    SST.append(S5T)
    SSo.append(S5o)

    ev = aucval[2:] / auctr[2:]
    ev = aucval[2:] - (np.array(ev < 0.98, dtype=float) * 0.025)
    indm = np.argmax(ev)
    vm = ev[indm]

    res = np.mean(SSo[indm])
    resV = np.mean(SSV[indm])
    resT = np.mean(SST[indm])
    auctr = roc_auc_score(TARo, res)
    aucval = roc_auc_score(TARV, resV)
    auctst = roc_auc_score(TART, resT)
    print('Teacher AUC:', round(auctst, 4))
    return res, resV, resT