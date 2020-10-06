import pandas as pd
from tqdm import tqdm, trange


def memorability_of_image(participant, from_scratch=True):
    if from_scratch:
        images = pd.read_csv("behavioral/" + participant +
                             "_behavioral.csv")[["stimulus"]]
        images['accuracy_total'] = 0
    else:
        images = pd.read_csv("behavioral_final/stimulus_accuracy.csv")

    data = pd.merge(pd.read_csv("behavioral/" + participant +
                                "_behavioral.csv")[["stimulus", "acc"]], images, how="inner")

    data["accuracy_total"] = data["accuracy_total"] + data["acc"]
    data = data[["stimulus", "accuracy_total"]]

    return data


if __name__ == "__main__":
    for i in trange(47, 50):
        if i != 1:
            memorability_of_image(str(i), False).to_csv(
                "behavioral_final/stimulus_accuracy.csv")
        else:
            memorability_of_image(str(i)).to_csv(
                "behavioral_final/stimulus_accuracy.csv")
    pass
