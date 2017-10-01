from functools import reduce
import numpy as np
import pickle
from python_speech_features import mfcc
from sklearn.decomposition import PCA

SAMPLE_RATE = 44100
MFCC_SIZE = 13
WHITENING_EPSILON = 0.1

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
    mfcc_size = mfccs[0].shape[-1]
    pooledMfccs = np.zeros([len(mfccs), vectorSize, mfcc_size])
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
    outputLength = min(max([x.shape[0] for x in mfccs]), 250)
    paddedMfccs = poolMfccs(mfccs, outputLength)
    return paddedMfccs, outputLength

def nestedListMeanAndVarNormalize(dataset):
    # Calculate averages
    mfcc_size = dataset[0][0].shape[-1]
    totals = np.zeros(mfcc_size)
    numPoints = 0
    for example in dataset:
        totals += reduce(lambda x,y: x+y, map(lambda z: np.sum(z, axis= 0), example))
        numPoints += reduce(lambda x,y: x+y, map(lambda x: x.shape[0], example))

    # Mean Normalization
    averages = totals / numPoints
    distanceFromMean = np.zeros(mfcc_size)
    for example in dataset:
        example = list(map(lambda x: x - averages, example))
        distanceFromMean += reduce(lambda x,y: x+y,
                                   map(lambda z: np.sum(np.square(z), axis= 0), example))

    # Variance normalization
    variances = distanceFromMean / numPoints
    for example in dataset:
            example = list(map(lambda x: x / variances, example))

    return dataset



def meanAndVarNormalize(dataset, labels= False):
    if labels:
        print("--[|] Note: Normalizing with labels in dataset")
        datasetWithoutLabels = [example[0] for example in dataset]
        datasetWithoutLabels = nestedListMeanAndVarNormalize(datasetWithoutLabels)
        return [[data, example[1]] for data, example in zip(datasetWithoutLabels, dataset)]

    # Check if argument is a nested list
    if isinstance(dataset[0], list):
        print("--[|] Note: Normalizing grouped dataset")
        return nestedListMeanAndVarNormalize(dataset)

    mfcc_size = dataset[0][0].shape[-1]
    # Calculate averages
    totals = np.zeros(mfcc_size)
    numPoints = 0
    for example in dataset:
        totals = totals + np.sum(example, axis= 0)
        numPoints += example.shape[0]

    # Mean Normalization
    averages = totals / numPoints
    distanceFromMean = np.zeros(mfcc_size)
    for example in dataset:
        example = example - averages
        distanceFromMean += np.sum(np.square(example), axis= 0)

    # Variance normalization
    variances = distanceFromMean / numPoints
    for example in dataset:
        example = example / variances
        
    return dataset

# Don't use!
def zcaWhiten(dataset):
    exampleLengths = [example.shape[0] for example in dataset]
    dataMatrix = np.concatenate(dataset)
    u, s, v = np.linalg.svd(dataset)
    xPCA = np.matmul(u.T, x0)
    print("PCA done")
    delta = np.diag(np.reciprocal(s + WHITENING_EPSILON))
    xPCAWhite = np.matmul(np.matmul(delta, u.T), xPCA)
    xZCAWhite = np.matmul(u, xPCAWhite)
    return np.split(xZCAWhite, exampleLengths, axis= 0)

#print("[*] Loading val set")
#val_set = loadDataSet("validation_set.pkl")
#val_set[0] = list(map(lambda x: mfcc(x, SAMPLE_RATE), val_set[0]))
#print("[*] Saving val set")
#with open("mfcc_val_set.pkl", "wb") as f:
#pickle.dump(val_set, f)

