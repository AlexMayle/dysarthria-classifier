from __future__ import print_function
import pickle

MALE = "male"
FEMALE = "female"
HEALTHY = 0
UNHEALTHY = 1
DATA_FILENAME = "mfcc_data/mfcc_full_set-2.pkl"
META_FILENAME = "mfcc_data/mfcc_full_set-2_meta.pkl"
SYL_COUNTS = {
    'HEALTHY': {
        'MALE': 0,
        'FEMALE': 0
    },
    'UNHEALTHY': {
        'MALE': 0,
        'FEMALE': 0
    }}

PAT_COUNTS = {
    'HEALTHY': {
        'MALE': 0,
        'FEMALE': 0
    },
    'UNHEALTHY': {
        'MALE': 0,
        'FEMALE': 0
    }}


with open(META_FILENAME, 'rb') as f:
    metadata = pickle.load(f)

with open(DATA_FILENAME, 'rb') as f:
    data_and_labels = pickle.load(f)

for data_and_label, meta in zip(data_and_labels, metadata):
    _, label = data_and_label
    _, gender, syls, _ = meta
    if label == HEALTHY:
        if gender == FEMALE:
            SYL_COUNTS['HEALTHY']['FEMALE'] += len(syls)
            PAT_COUNTS['HEALTHY']['FEMALE'] += 1
        else:
            SYL_COUNTS['HEALTHY']['MALE'] += len(syls)
            PAT_COUNTS['HEALTHY']['MALE'] += 1
    else:
        if gender == FEMALE:
            SYL_COUNTS['UNHEALTHY']['FEMALE'] += len(syls)
            PAT_COUNTS['UNHEALTHY']['FEMALE'] += 1
        else:
            SYL_COUNTS['UNHEALTHY']['MALE'] += len(syls)
            PAT_COUNTS['UNHEALTHY']['MALE'] += 1

print("***Syllables***")
print(SYL_COUNTS.items())
print("***Patients****")
print(PAT_COUNTS.items())
