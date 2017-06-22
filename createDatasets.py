import sys
import os
import pickle
import random

import scipy.io.wavfile as wav
from python_speech_features import mfcc

DATA_DIR = "data/"
MALE_HEALTHY_DIR = DATA_DIR + "male healthy"
MALE_PATIENT_DIR = DATA_DIR + "male patient"
FEMALE_HEALTHY_DIR = DATA_DIR + "female healthy"
FEMALE_PATIENT_DIR = DATA_DIR + "female patient"
HEALTHY_LABEL = 0
PATIENT_LABEL = 1
SAMPLE_RATE = 44100

def loadAndGroupWavFilesWithLabels(directory, label):
    """
    Loads all wav files associated with each speaker,
    groups them, and associates a label

    Automatically filters out files that aren't wav files and those that are
    too long to be single mandarin characters
    
    Returns: A list of lists of the form [[[wavFiles], label], ...]
    """
    # get directories containing speakers' mandarin character pronunciations
    speakerDirNames = [os.path.join(directory, dirName) for dirName in os.listdir(directory)
                                                            if dirName[0] != '.']
    examples = []
    discardCount = 0
    for dirName in speakerDirNames:
        # get file names and filter out the ones that aren't wav files
        filenames = [os.path.join(dp, f) for dp, _, fn in os.walk(dirName) for f in fn]
        wavFilenames = [filename for filename in filenames if filename[-4:] == ".wav"]
        
        wavDataPoints = []
        for path in wavFilenames:
            _, wavData = wav.read(path)

            # filter out wav files that are too long to be single character pronunciations
            if len(wavData) / 44100 < 5:
                wavDataPoints.append(wavData)
            else:
                print("Warning: ", path, " is being discarded for being too long")
                discardCount += 1

        examples.append([wavDataPoints, label])

    print("Total filtered out of dataset: ", discardCount)
    return examples

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

def convertToMfccs(groupedExamples):
    """
    Convert a set of the form [[[wavData], label], ...]
    to a set of the form [[[mfccData], label], ...]
    """
    for examplesLabelPair in groupedExamples:
        examplesLabelPair[0] = list(map(lambda x: mfcc(x, SAMPLE_RATE), examplesLabelPair[0]))
    return groupedExamples

def splitDataAndLabels(dataset):
   """
   Convert a data set of the form [[data, label], ...]
   to a set of the form [[data], [labels]]
   """
   data = []
   labels = []
   for dataLabelPair in dataset:
       print(dataLabelPair[1])
       data.append(dataLabelPair[0])
       labels.append(dataLabelPair[1])
   return [data, labels]

if __name__ == '__main__':
    ############# Start of Script ##################################################
    print("[*] Loading male healthy files")
    examples = loadAndGroupWavFilesWithLabels(MALE_HEALTHY_DIR, HEALTHY_LABEL)
    print("[*] Loading male patient files")
    examples = examples + loadAndGroupWavFilesWithLabels(MALE_PATIENT_DIR, PATIENT_LABEL)
    print("[*] Loading female healthy files")
    examples = examples + loadAndGroupWavFilesWithLabels(FEMALE_HEALTHY_DIR, HEALTHY_LABEL)
    print("[*] Loading female patient files")
    examples = examples + loadAndGroupWavFilesWithLabels(FEMALE_PATIENT_DIR, PATIENT_LABEL)

    print("[*] Partitioning into training, validation, and testing set")
    random.shuffle(examples)

    if sys.argv[1] == '--no-partition':
        mfcc_full_set = convertToMfccs(examples)
        with open('mfcc_full_set.pkl', 'wb') as dataset_file:
            pickle.dump(mfcc_full_set, dataset_file)

    trainSplit = len(examples) // 2
    valSplit = trainSplit + int(.16 * len(examples))
    groupedTrainSet = examples[:35]
    groupedValSet = examples[35:40]
    groupedTestSet = examples[40:]

    print("[*] Converting to MFCCs")
    groupedTrainSet = convertToMfccs(groupedTrainSet)
    groupedValSet = convertToMfccs(groupedValSet)
    groupedTestSet = convertToMfccs(groupedTestSet)

    print("[*] creating ungrouped versioans of partitions")
    trainSet = splitIntoPatches(groupedTrainSet)
    valSet = splitIntoPatches(groupedValSet)
    testSet = splitIntoPatches(groupedTestSet) #This is mainly for debugging purposes so we can see
                                               #how the model is learning

    print("[*] Shuffling up speakers and pronunciations in ungrouped sets")
    random.shuffle(trainSet)
    random.shuffle(valSet)
    random.shuffle(testSet)

    # statistics
    numTrainSpeakers = len(groupedTrainSet)
    numValSpeakers = len(groupedValSet)
    numTestSpeakers = len(groupedTestSet)
    numTrainExamples = len(trainSet)
    numValExamples = len(valSet)
    numTestExamples = len(testSet)
    # Don't need these now
    groupedTrainSet = groupedValSet = None

    # Split the data and labels into their own lists
    trainSet = splitDataAndLabels(trainSet)
    valSet = splitDataAndLabels(valSet)
    testSet = splitDataAndLabels(testSet)
    groupedTestSet = splitDataAndLabels(groupedTestSet)

    print("[*] Saving train set to disk")
    with open("mfcc_train_set.pkl", "wb") as f:
        pickle.dump(trainSet, f)
    print("[*] Saving validation set to disk")
    with open("mfcc_val_set.pkl", "wb") as f:
        pickle.dump(valSet, f)
    print("[*] Saving grouped test set to disk")
    with open("mfcc_grouped_test_set.pkl", "wb") as f:
        pickle.dump(groupedTestSet, f)
    print("[*] Saving test set to disk")
    with open("mfcc_test_set.pkl", "wb") as f:
        pickle.dump(testSet, f)

    print("*" * 15 + " Data Set Stats " + "*" * 15)
    print("Set          | Speakers  | Examples  ")
    print("-------------+-----------+-----------")
    print("Train          %s          %s        " % (numTrainSpeakers, numTrainExamples))
    print("Val            %s          %s        " % (numValSpeakers, numValExamples))
    print("Test           %s          %s        " % (numTestSpeakers, numTestExamples))
