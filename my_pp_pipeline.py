import pickle
from preprocess import meanAndVarNormalize

def load_data():
    # Unpickle`
    with open('mfcc_full_set.pkl', 'rb') as f:
        full_data_set = pickle.load(f)

    # First 9 are validation and the rest can be used to create folds
    cross_validation_data = full_data_set[9:]
    validation_data = full_data_set[:9]

    # Convert to standard scores
    cross_validation_data = meanAndVarNormalize(cross_validation_data, labels=True)
    validation_data = meanAndVarNormalize(validation_data, labels=True)

    # create your folds and feed the network...
    return cross_validation_data, validation_data
