"""
load_mnist.py  --  EBC4286 Convex Optimisation, Assignment 4

Reads the MNIST handwritten digits (frozen copy on Canvas, mnist.npz), shows
digit images, and computes the key numbers.

Usage
-----
    from load_mnist import load_mnist, show_digits, key_numbers

    X, y, X_test, y_test = load_mnist("mnist.npz")
    ...                                       # your classifier: pred = predicted digits
    key_numbers(y_test, pred_test)            # error in %, most confused pair
    show_digits(X[hard], titles=y[hard])      # look at the hard images

Data
----
* MNIST (LeCun, Cortes and Burges): 28 x 28 grey-scale images of handwritten
  digits 0-9, http://yann.lecun.com/exdb/mnist/
* Training set: 60,000 labelled images. Use (a subset of) these for
  estimation AND for choosing your model and its parameters (e.g. with a
  validation set or cross-validation). How many you use is your choice.
* Test set: 10,000 images. Use it only to evaluate your final classifier,
  never to choose a model, a parameter or the training size.
* Each image is one row of 784 pixel values (row by row), scaled to [0, 1]
  (0 = white background, 1 = black ink). Labels are the digits 0-9.
"""

import numpy as np


def load_mnist(path="mnist.npz"):
    """X (60000 x 784), y (60000,), X_test (10000 x 784), y_test (10000,); pixels in [0, 1], labels 0-9."""
    d = np.load(path)
    X = d["X_train"].astype(float) / 255.0
    X_test = d["X_test"].astype(float) / 255.0
    return X, d["y_train"].astype(int), X_test, d["y_test"].astype(int)


def subsample(y, n, seed=0):
    """Indices of n training images, drawn at random with (as far as possible) n/10 per digit."""
    rng = np.random.default_rng(seed)
    per = [rng.permutation(np.where(y == k)[0]) for k in range(10)]
    idx = np.concatenate([p[: n // 10 + (k < n % 10)] for k, p in enumerate(per)])
    return np.sort(idx)


def show_digits(images, titles=None, ncols=10, size=1.2):
    """Show images (rows of 784 pixels) in a grid; titles is an optional list of short strings."""
    import matplotlib.pyplot as plt

    images = np.atleast_2d(images)
    n = len(images)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows * 1.15), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k < n:
            ax.imshow(images[k].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(str(titles[k]), fontsize=8)
    fig.tight_layout()
    return fig


def confusion(y_true, pred):
    """10 x 10 confusion matrix: entry (i, j) = number of images of digit i classified as j."""
    M = np.zeros((10, 10), dtype=int)
    np.add.at(M, (np.asarray(y_true, int), np.asarray(pred, int)), 1)
    return M


def key_numbers(y_true, pred):
    """
    The key numbers for Appendix A, computed in the same way for everybody.

    y_true : true digits (0-9)
    pred   : your predicted digits for the same images

    Returns a dict with
      error         : % of images misclassified
      confused_pair : the pair of digits most often confused with each other
                      (both directions added up), e.g. "4-9"
      confused_n    : number of images in that pair that were confused
    """
    y_true, pred = np.asarray(y_true, int), np.asarray(pred, int)
    M = confusion(y_true, pred)
    S = M + M.T
    np.fill_diagonal(S, 0)
    i, j = np.unravel_index(np.argmax(np.triu(S)), S.shape)
    return {"error": 100.0 * float(np.mean(y_true != pred)),
            "confused_pair": f"{min(i, j)}-{max(i, j)}", "confused_n": int(S[i, j])}


if __name__ == "__main__":
    import sys

    X, y, X_test, y_test = load_mnist(sys.argv[1] if len(sys.argv) > 1 else "mnist.npz")
    print(f"training set {X.shape}, test set {X_test.shape}")
    print("images per digit (training):", np.bincount(y).tolist())
