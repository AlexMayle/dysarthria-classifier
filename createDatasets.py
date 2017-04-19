import os
import numpy as np
import pickle
import scipy.io.wavfile as wav
import random

DATA_DIR = "data/"
MALE_HEALTHY_DIR = DATA_DIR + "male healthy"
MALE_PATIENT_DIR = DATA_DIR + "male patient"
FEMALE_HEALTHY_DIR = DATA_DIR + "female healthy"
FEMALE_PATIENT_DIR = DATA_DIR + "female patient"
HEALTHY_LABEL = 1
PATIENT_LABEL = 0

def loadWavFilesWithLabels(directory, label):
    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]
    wavFilenames = [filename for filename in filenames if filename[-4:] == ".wav"]

    examples = []
    for path in wavFilenames:
        example = []
        _, wavData = wav.read(path)
        example.append(wavData)
        example.append(label)
        examples.append(example)
    return examples

# Gather all .wav files and create associated labels
print("[*] Loading male healthy files")
examples = loadWavFilesWithLabels(MALE_HEALTHY_DIR, HEALTHY_LABEL)
print("[*] Loading male patient files")
examples = examples + loadWavFilesWithLabels(MALE_PATIENT_DIR, PATIENT_LABEL)
print("[*] Loading female healthy files")
examples = examples + loadWavFilesWithLabels(FEMALE_HEALTHY_DIR, HEALTHY_LABEL)
print("[*] Loading female patient files")
examples = examples + loadWavFilesWithLabels(FEMALE_PATIENT_DIR, PATIENT_LABEL)

# Shuffle the .wav, label pairs
print("[*] Shuffling examples")
random.shuffle(examples)

# Divide into sets
half = len(examples) // 2
quarter = half // 2
trainSet = examples[:half]
validationSet = examples[half: half + quarter]
testSet = examples[half + quarter:]

# Save newly organized data structures to disk
print("[*] Saving training set to disk")
with open("train_set.pkl", "wb") as f:
    pickle.dump(trainSet, f)

print("[*] Saving validation set to disk")
with open("validation_set.pkl", "wb") as f:
    pickle.dump(validationSet, f)

print("[*] Saving test set to disk")
with open("test_set.pkl", "wb") as f:
    pickle.dump(testSet, f)

print("*" * 20 + " Data Set Stats " + "*" * 20)
print("Set      | Size")
print("=========+======")
print("Train     " + str(len(trainSet)))
print("Validate  " + str(len(validationSet)))
print("Test      " + str(len(testSet)))

"""
print()
print("*" * 20 + " Debug Info " + "*" * 20)
print("Training Set")
print(trainSet)
print("\n\n Validation Set")
print(validationSet)
print("\n\n Test Set")
print(testSet)
"""
