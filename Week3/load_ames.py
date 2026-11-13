"""
load_ames.py  --  EBC4286 Convex Optimisation, Assignment 3

Reads the Ames Housing data (De Cock, 2011; frozen copy on Canvas), fixes the
missing values that only mean "this house has no ...", and splits the sales
into an estimation set (2006-2009) and a test set (2010).

Usage
-----
    from load_ames import load_ames, split, key_numbers

    df = load_ames("ames_housing.csv")
    train, test = split(df)
    ...                                   # your model: valuation = f(features)
    key_numbers(test["SalePrice"], valuation_test)

Data
----
* 2,930 residential sales in Ames, Iowa, 2006-2010; 80 house characteristics.
* Variable descriptions: De Cock (2011), Journal of Statistics Education 19(3),
  data documentation at http://jse.amstat.org/v19n3/decock/DataDocumentation.txt
* Column names are kept exactly as in that documentation (with spaces),
  e.g. df["Gr Liv Area"], df["Neighborhood"], df["Sale Condition"].

What the loader changes
-----------------------
* Categorical columns where a missing value means "none" (no basement, no
  garage, no pool, no fence, ...) get the category "None".
* Numeric columns where a missing value means "none" (basement area, garage
  area, number of garage cars, ...) get the value 0.
* One missing "Electrical" value gets the most common category.
* Left as missing on purpose: "Lot Frontage" (unknown for 490 houses) and
  "Garage Yr Blt" (no garage). How to treat these is up to you.
* Nothing else is changed: unusual sales and outliers are still in the data.
"""

import numpy as np
import pandas as pd

_NONE_CATEGORICAL = ["Alley", "Mas Vnr Type", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
                     "BsmtFin Type 1", "BsmtFin Type 2", "Fireplace Qu", "Garage Type",
                     "Garage Finish", "Garage Qual", "Garage Cond", "Pool QC", "Fence",
                     "Misc Feature"]
_ZERO_NUMERIC = ["Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
                 "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath", "Garage Cars",
                 "Garage Area"]


def load_ames(path="ames_housing.csv"):
    """The Ames Housing data as a DataFrame (one row per sale), cleaned as described above."""
    df = pd.read_csv(path)
    for c in _NONE_CATEGORICAL:
        df[c] = df[c].fillna("None")
    for c in _ZERO_NUMERIC:
        df[c] = df[c].fillna(0.0)
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    return df


def split(df):
    """Estimation set: sales in 2006-2009. Test set: sales in 2010.

    Use the test set only for the final evaluation of your recommended model,
    never to choose features or parameters.
    """
    train = df[df["Yr Sold"] <= 2009].reset_index(drop=True)
    test = df[df["Yr Sold"] == 2010].reset_index(drop=True)
    return train, test


def key_numbers(price, valuation):
    """
    The key numbers for Appendix A, computed in the same way for everybody.

    price     : actual sale prices (dollars)
    valuation : your model's valuations of the same houses (dollars)

    Returns a dict with
      over_rate : % of houses valued ABOVE their sale price
      mae       : mean absolute error, in dollars
      mdape     : median absolute percentage error, in %
    """
    price = np.asarray(price, dtype=float)
    valuation = np.asarray(valuation, dtype=float)
    err = valuation - price
    return {
        "over_rate": 100.0 * np.mean(err > 0),
        "mae": float(np.mean(np.abs(err))),
        "mdape": 100.0 * float(np.median(np.abs(err) / price)),
    }


if __name__ == "__main__":
    import sys

    df = load_ames(sys.argv[1] if len(sys.argv) > 1 else "ames_housing.csv")
    train, test = split(df)
    print(f"{len(df)} sales, {df.shape[1]} columns; estimation set {len(train)}, test set {len(test)}")
    missing = df.isna().sum()
    print("Still missing:", missing[missing > 0].to_dict())
    print("Sale conditions:", df["Sale Condition"].value_counts().to_dict())
