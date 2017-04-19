import numpy as np

def poolMfccs(mfccs, vectorSize):
    pooledMfccs = np.zeros([len(mfccs), vectorSize, 13])
    for mfcc, i in zip(mfccs, range(len(mfccs))):
        if vectorSize > len(mfcc):
            pooledMfccs[i] = np.pad(mfcc, ((0, vectorSize - len(mfcc)), (0,0)),
                          "constant", constant_values= 0)
        
        windowLength = 2 * len(mfcc) // (vectorSize + 2)
        stride = max(windowLength // 2, 1)
        for j in range(vectorSize):
            window = mfcc[j*stride : j*stride + windowLength]
            if window.size != 0:
                pooledMfccs[i,j] = np.mean(window, axis= 0)
        
    return pooledMfccs
