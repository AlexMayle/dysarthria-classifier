import pickle

def loadDataSet(path):
    with open(path, "rb") as f:
        trainSet = pickle.load(f)

    data = []
    labels = []
    for wav, label in trainSet:
        data.append(wav)
        labels.append(label)

    return [data, labels]

