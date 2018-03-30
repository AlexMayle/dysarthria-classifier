import os
import argparse
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
MAX_LENGTH = 5  # seconds
WAV_EXTENSION = ".wav"
META_SUFFIX = "_meta"

parser = argparse.ArgumentParser('Dataset Factory')
parser.add_argument("--numceps",
                    type=int,
                    default=13,
                    help="Number of cepstrums included in each MFCC feature. (default = 13)",)
parser.add_argument('--lowcep_offset',
                    type=int,
                    default=0,
                    help="offset from lowest cepstrum to include in MFCC features.")
parser.add_argument('--output_path',
                    type=str,
                    default='',
                    help='Directory to save the dataset')
parser.add_argument("--sample_rate",
                    type=int,
                    default="44100",
                    help="Sample rate of wav data")
parser.add_argument("--exclude_z",
                    action="store_true",
                    default=False,
                    help="Excludes syllables prefixed with 'z'")
parser.add_argument("--exclude_zz",
                    action="store_true",
                    default=False,
                    help="Excludes syllables prefixed with 'zz'")
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()


def parse_speaker(speaker_dir):
    speaker_dir = speaker_dir.split('-')
    speaker = speaker_dir[1]
    notes = speaker_dir[2:]
    return speaker, notes


def parse_syllable(syllable_filename):
    syllable = syllable_filename.rstrip(WAV_EXTENSION)
    return syllable


def extract_meta(dataset):
    data, labels, meta_data = zip(*dataset)
    data_and_labels = tuple(zip(data, labels))
    return data_and_labels, meta_data

def create_annotated_dataset(data_path, label, spkr_gender):
    examples = []
    speaker_directories = filter(lambda x: not x.startswith('.'), os.listdir(data_path))
    for spkr_directory in speaker_directories:
        spkr_name, spkr_notes = parse_speaker(spkr_directory)
        spkr_path = os.path.join(data_path, spkr_directory)

        wav_files = filter(lambda x: x.endswith(WAV_EXTENSION), os.listdir(spkr_path))
        spkr_datapoints = []
        for wav_filename in wav_files:
            # filter by syllable
            syllable = parse_syllable(wav_filename)
            if FLAGS.exclude_z and syllable.lower().startswith("z-"):
                continue
            if FLAGS.exclude_zz and syllable.lower().startswith("zz-"):
                continue

            _, data = wav.read(os.path.join(spkr_path, wav_filename))
            if len(data) / FLAGS.sample_rate < MAX_LENGTH:
                highest_cepstrum = FLAGS.lowcep_offset + FLAGS.numceps
                data = mfcc(data, FLAGS.sample_rate, numcep=highest_cepstrum)
                data = data[:, FLAGS.lowcep_offset:]
                spkr_datapoints.append((data, syllable))

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

    print("[*] Extracting meta data")
    data_and_labels, meta_data = extract_meta(dataset)
    return data_and_labels, meta_data



if __name__ == '__main__':
    print("Creating dataset with %d cepstrums" % FLAGS.numceps)
    if FLAGS.exclude_z:
        print("Excluding 'z' prefixed syllables from dataset")
    if FLAGS.exclude_zz:
        print("Excluding 'sh' prefixed syllables from dataset")

    dataset, meta_data = run()

    print("[*] Saving grouped test set to disk at %s" % FLAGS.output_path + ".pkl")
    with open(FLAGS.output_path + ".pkl", "wb") as f:
        pickle.dump(dataset, f)

    print("[*] Meta data saved to disk at %s" % FLAGS.output_path + META_SUFFIX + ".pkl")
    with open(FLAGS.output_path + META_SUFFIX + ".pkl", "wb") as f:
        pickle.dump(meta_data, f)
