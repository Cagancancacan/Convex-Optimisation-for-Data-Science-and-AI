"""
load_adult.py  --  EBC4286 Convex Optimisation, Assignment 5

Reads the UCI Adult data (frozen copy on Canvas, adult.csv) and computes the
key numbers.

Usage
-----
    from load_adult import load_adult, group_table, key_numbers

    train, test = load_adult("adult.csv")
    ...                                        # your classifier and selection rule
    group_table(test["income"] == ">50K", selected, test["race"])   # rates per group
    key_numbers(test["income"] == ">50K", selected, test["race"])   # Appendix A

Data
----
* UCI Adult (Becker and Kohavi, 1996): 48,842 people from the 1994 US Census,
  https://archive.ics.uci.edu/dataset/2/adult . Variable descriptions are in
  adult.names.
* Training set: 32,561 people (split == "train"); test set: 16,281 people
  (split == "test"), the original split of the data set. Use the test set
  only to evaluate your final model, never to make choices.
* Label: income ">50K" or "<=50K". In the case, ">50K" plays the role of a
  declaration that deserves a check.
* Missing values ("?" in the original files) are empty (NaN) in workclass,
  occupation and native-country. What to do with them is your choice.
* fnlwgt is a census sampling weight, not a characteristic of the person.
* race has five categories. For the key numbers, Amer-Indian-Eskimo and Other
  are merged into one group "Other", giving four groups: White, Black,
  Asian-Pac-Islander and Other. You may use other groupings in your own
  model; argue your choice.
"""

import numpy as np
import pandas as pd

GROUPS = ["White", "Black", "Asian-Pac-Islander", "Other"]


def load_adult(path="adult.csv"):
    """Returns (train, test) as DataFrames with the original columns."""
    d = pd.read_csv(path)
    train = d[d["split"] == "train"].drop(columns="split").reset_index(drop=True)
    test = d[d["split"] == "test"].drop(columns="split").reset_index(drop=True)
    return train, test


def race_group(race):
    """Maps the five race categories to the four groups used in the key numbers."""
    race = pd.Series(race).astype(str)
    return race.where(race.isin(GROUPS[:3]), "Other").to_numpy()


def group_table(y_true, selected, race):
    """
    Per group: number of people, selection rate, false positive rate and
    true positive rate, all rates in %.

    y_true   : True / 1 for people with income >50K
    selected : True / 1 for people your rule selects for a check
    race     : the race column (five categories)
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(selected).astype(bool)
    g = race_group(race)
    rows = []
    for k in GROUPS + ["All"]:
        m = np.ones_like(y) if k == "All" else (g == k)
        rows.append({"group": k, "n": int(m.sum()),
                     "selection_rate": 100 * s[m].mean(),
                     "fpr": 100 * s[m & ~y].mean(),          # selected although income <=50K
                     "tpr": 100 * s[m & y].mean()})          # selected among income >50K
    return pd.DataFrame(rows).set_index("group")


def key_numbers(y_true, selected, race):
    """
    The key numbers for Appendix A, computed in the same way for everybody.

    Returns a dict with (all in %)
      error    : share of people misclassified (selected while <=50K, or not selected while >50K)
      sel_rate : share of people selected
      dp_gap   : largest minus smallest selection rate over the four groups
      fpr_gap  : largest minus smallest false positive rate over the four groups
      tpr_gap  : largest minus smallest true positive rate over the four groups
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(selected).astype(bool)
    t = group_table(y, s, race).loc[GROUPS]
    return {"error": 100.0 * float(np.mean(s != y)),
            "sel_rate": 100.0 * float(s.mean()),
            "dp_gap": float(t["selection_rate"].max() - t["selection_rate"].min()),
            "fpr_gap": float(t["fpr"].max() - t["fpr"].min()),
            "tpr_gap": float(t["tpr"].max() - t["tpr"].min())}


if __name__ == "__main__":
    import sys

    train, test = load_adult(sys.argv[1] if len(sys.argv) > 1 else "adult.csv")
    print(f"training set {train.shape}, test set {test.shape}")
    print("share with income >50K (train):", round(100 * (train["income"] == ">50K").mean(), 1), "%")
    print(pd.Series(race_group(train["race"])).value_counts().to_string())
