import numpy as np
import pickle
from python_speech_features import mfcc

SAMPLE_RATE = 44100

def createMfccDataset():
    train_set = loadDataSet("train_set.pkl")
    test_set = loadDataSet("test_set.pkl")
    
    print("[*] Converting wav files to mfcc features")
    train_set[0] = list(map(lambda x: mfcc(x, samplerate= SAMPLE_RATE), train_set[0]))
    test_set[0] = list(map(lambda x: mfcc(x, samplerate= SAMPLE_RATE), test_set[0]))

    print("[*] Saving mfcc features to disk for later use")
    with open("mfcc_train_set.pkl", "wb") as f:
      pickle.dump(train_set, f)
    with open("mfcc_test_set.pkl", "wb") as f:
      pickle.dump(test_set, f)

    return train_set, test_set
      
def loadDataSet(path):
    with open(path, "rb") as f:
        trainSet = pickle.load(f)

    data = []
    labels = []
    for wav, label in trainSet:
        data.append(wav)
        labels.append(label)

    return [data, labels]


def poolMfccs(mfccs, vectorSize):
    pooledMfccs = np.zeros([len(mfccs), vectorSize, 13])
    for mfcc, i in zip(mfccs, range(len(mfccs))):
        if vectorSize == len(mfcc):
            pooledMfccs[i] = mfcc
        elif vectorSize > len(mfcc):
            pooledMfccs[i] = np.pad(mfcc, ((0, vectorSize - len(mfcc)), (0,0)),
                          "constant", constant_values= 0)
        else:
            windowLength = 2 * len(mfcc) // (vectorSize + 2)
            stride = max(windowLength // 2, 1)
            for j in range(vectorSize):
                window = mfcc[j*stride : j*stride + windowLength]
                if window.size != 0:
                    pooledMfccs[i,j] = np.mean(window, axis= 0)
        
    return pooledMfccs

def padBatch(mfccs):
    outputLength = min(max([x.shape[0] for x in mfccs]), 200)
    paddedMfccs = poolMfccs(mfccs, outputLength)
    return paddedMfccs, outputLength
