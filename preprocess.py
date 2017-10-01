from functools import reduce
import numpy as np

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

def splitIntoPatches(dataset):
    """
    Given a dataset of examples grouped by speaker
    of the form [[[examples], label], ...] return a new dataset
    of the form [[example, label], ...] which is not grouped by speaker

    dataset:  dataset with a many to one relationship between examples and labels
    shuffle:  If true, the newly created dataset is shuffled before being returned

    return:   A dataset with a one-to-one relationship between examples and labels,
              which may or may not be shuffled depending on the shuffle argument
    """

    patches = []
    for examplesLabelPair in dataset:
        ungroupedExamples = list(map(lambda x: [x, examplesLabelPair[1]], examplesLabelPair[0]))
        patches += ungroupedExamples

    return patches

def splitDataAndLabels(dataset):
   """
   Convert a data set of the form [[data, label], ...]
   to a set of the form [[data], [labels]]
   """
   data = []
   labels = []
   for dataLabelPair in dataset:
       data.append(dataLabelPair[0])
       labels.append(dataLabelPair[1])
   return [data, labels]
