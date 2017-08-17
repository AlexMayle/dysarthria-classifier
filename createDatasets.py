import sys
import os
import pickle
import random

import scipy.io.wavfile as wav
from python_speech_features import mfcc

MALE = "male"
FEMALE = "female"
DATA_DIR = "data/"
MALE_HEALTHY_DIR = DATA_DIR + "male healthy"
MALE_PATIENT_DIR = DATA_DIR + "male patient"
FEMALE_HEALTHY_DIR = DATA_DIR + "female healthy"
FEMALE_PATIENT_DIR = DATA_DIR + "female patient"
HEALTHY_LABEL = 0
PATIENT_LABEL = 1
SAMPLE_RATE = 44100
MAX_LENGTH = 5  # seconds
WAV_EXTENSION = ".wav"
LENGTH_ERROR_MSG =  "Warning: %s is being discarded due to being too long"

def  parse_speaker(speaker_dir):
    speaker_dir = speaker_dir.split('-')
    speaker = speaker_dir[1]
    notes = speaker_dir[2:]
    return speaker, notes


def parse_syllable(syllable_filename):
    syllable = syllable_filename.rstrip(WAV_EXTENSION)
    return syllable


def create_annotated_dataset(data_path, label, spkr_gender):
    """
    Loads all wav files associated with each speaker,
    groups them, and associates a label

    Automatically filters out files that aren't wav files and those that are
    too long to be single mandarin characters

    Returns: A list of lists of the form [[[wavFiles], label], ...]
    """
    examples = []
    speaker_directories = filter(lambda x: not x.startswith('.'), os.listdir(data_path))
    for spkr_directory in speaker_directories:
        spkr_name, spkr_notes = parse_speaker(spkr_directory)
        spkr_path = os.path.join(data_path, spkr_directory)

        wav_files = filter(lambda x: x.endswith(WAV_EXTENSION), os.listdir(spkr_path))
        spkr_datapoints = []
        for wav_filename in wav_files:
            _, data = wav.read(os.path.join(spkr_path, wav_filename))
            if len(data) / SAMPLE_RATE < MAX_LENGTH:
                data = mfcc(data, SAMPLE_RATE)
                spkr_datapoints.append((data, parse_syllable(wav_filename)))
            else:
                print(LENGTH_ERROR_MSG % wav_filename)

        random.shuffle(spkr_datapoints)
        spkr_datapoints, syllable_annotations = zip(*spkr_datapoints)
        annotation = (spkr_name, spkr_gender, syllable_annotations, spkr_notes)
        examples.append((spkr_datapoints, label, annotation))

    return examples


def run():
    print("[*] Loading male healthy files")
    dataset = create_annotated_dataset(MALE_HEALTHY_DIR, HEALTHY_LABEL, MALE)
    print("[*] Loading male patient files")
    dataset = dataset + create_annotated_dataset(MALE_PATIENT_DIR, PATIENT_LABEL, MALE)
    print("[*] Loading female healthy files")
    dataset = dataset + create_annotated_dataset(FEMALE_HEALTHY_DIR, HEALTHY_LABEL, FEMALE)
    print("[*] Loading female patient files")
    dataset = dataset + create_annotated_dataset(FEMALE_PATIENT_DIR, PATIENT_LABEL, FEMALE)

    print("[*] Shuffling speakers")
    random.shuffle(dataset)

    assert len(dataset) == 69
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
       data.append(dataLabelPair[0])
       labels.append(dataLabelPair[1])
   return [data, labels]


if __name__ == '__main__':
    try:
        SAVE_PATH = sys.argv[1]
    except IndexError:
        print("Please provide output file name of dataset")
        exit(-1)

    dataset = run()

    print("[*] Saving grouped test set to disk %s" % SAVE_PATH)
    with open(SAVE_PATH + ".pkl", "wb") as f:
        pickle.dump(dataset, f)
