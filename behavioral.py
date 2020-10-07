import pandas as pd
from tqdm import tqdm, trange


def memorability_of_image(participant, from_scratch=True):
    if from_scratch:
        images = pd.read_csv("behavioral/" + participant +
                             "_behavioral.csv")[["stimulus"]]
        images['accuracy_total'] = 0
        images['count'] = 0
    else:
        images = pd.read_csv("behavioral_final/stimulus_accuracy.csv")

    df = pd.read_csv("behavioral/" + participant +
                     "_behavioral.csv")[["stimulus", "acc_rec"]]

    data = pd.merge(df, images, how="inner")
    data["count"] += 1
    if data.shape[0] < 2:
        df["accuracy_total"] = 0
        df["count"] = 1
        images["acc_rec"] = 0
        data = pd.concat([images, df], ignore_index=True)

    data["accuracy_total"] = data["accuracy_total"] + data["acc_rec"]
    data["avg_accuracy"] = data["accuracy_total"] / data["count"]
    data = data[["stimulus", "accuracy_total", "count", "avg_accuracy"]]

    return data


if __name__ == "__main__":
    for i in trange(1, 6):
        if i != 1:
            memorability_of_image(str(i), False).to_csv(
                "behavioral_final/stimulus_accuracy.csv")
        else:
            memorability_of_image(str(i)).to_csv(
                "behavioral_final/stimulus_accuracy.csv")
    pass
