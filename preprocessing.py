import os
import librosa
import json

NUM_SAMPLES = 22050
DATASET_PATH = "dataset"
JSON_PATH = "data.json"

def preprocess(dataset_path, json_path, num_mfccs=13, n_fft=2048, hop_len=512):

    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all subfolders.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # checking if we are at the subfolder.
        if dirpath is not dataset_path:

            # saving labels
            labels = dirpath.split("dataset")[-1]
            labels = labels[1:]
            data["mappings"].append(labels)
            print("\nProcessing: '{}'".format(labels))

            # processing all the audio files and storing MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sample_rate = librosa.load(file_path)

                # checking length consistency
                if len(signal) >= NUM_SAMPLES:

                    signal = signal[:NUM_SAMPLES]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate,
                                                 n_mfcc=num_mfccs,
                                                 n_fft=n_fft,
                                                 hop_length=hop_len)

                    # storing data
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess(DATASET_PATH, JSON_PATH)