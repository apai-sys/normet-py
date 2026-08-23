"""Regression guard for the batched-SVD ridge augmentation in scm().

`_ridge_augment` must stay numerically equivalent to fitting
``RidgeCV(alphas, fit_intercept=True)`` independently at every timestamp — that
equivalence is the entire justification for the optimisation.
"""

import numpy as np
import pytest

from normet.causal.scm import _ridge_augment

pytest.importorskip("sklearn")
from sklearn.linear_model import RidgeCV  # noqa: E402


def test_ridge_augment_matches_per_timestamp_ridgecv():
    rng = np.random.default_rng(11)
    J, p, T = 8, 30, 40  # donors, pre-features, timestamps
    Xd = rng.normal(size=(J, p))
    Xt = rng.normal(size=(1, p))
    beta = rng.normal(size=(p, T))
    Y = (Xd @ beta + rng.normal(0, 0.3, size=(J, T))).T  # (T, J)
    alphas = np.array([i / 10 for i in range(1, 101)], dtype=float)

    a_batch, mt_batch, md_batch = _ridge_augment(Xd, Xt, Y, alphas)

    a_ref, mt_ref, md_ref = [], [], []
    for t in range(T):
        m = RidgeCV(alphas=alphas, fit_intercept=True).fit(Xd, Y[t])
        a_ref.append(m.alpha_)
        mt_ref.append(float(m.predict(Xt)[0]))
        md_ref.append(m.predict(Xd))
    a_ref = np.asarray(a_ref)
    mt_ref = np.asarray(mt_ref)
    md_ref = np.vstack(md_ref)

    assert np.allclose(a_batch, a_ref), "selected alphas differ from RidgeCV"
    assert np.allclose(mt_batch, mt_ref, atol=1e-8)
    assert np.allclose(md_batch, md_ref, atol=1e-8)


def test_ridge_augment_shapes():
    rng = np.random.default_rng(3)
    J, p, T = 6, 20, 15
    Xd = rng.normal(size=(J, p))
    Xt = rng.normal(size=(1, p))
    Y = rng.normal(size=(T, J))
    a, mt, md = _ridge_augment(Xd, Xt, Y, np.array([0.5, 1.0, 2.0]))
    assert a.shape == (T,)
    assert mt.shape == (T,)
    assert md.shape == (T, J)
