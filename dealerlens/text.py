"""Condition signal from listing text (title + description, or a dealer's inspection notes).

TF-IDF + ridge regression on log price. Leak-safe:
- dollar amounts and price-like numbers are scrubbed before the text is used
- training rows get out-of-fold predictions (the model never scores text it was fit on)
"""
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

_NUM = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?k?|\b\d[\d,]*(?:\.\d+)?\b")


def scrub(text):
    """Lowercase and replace prices / large numbers with "num". Keeps model years and small numbers."""
    def rep(m):
        s = m.group(0)
        digits = re.sub(r"\D", "", s)
        if not s.startswith("$"):
            if re.fullmatch(r"(19|20)\d\d", digits) or len(digits) <= 3:
                return s
        return " num "
    return _NUM.sub(rep, str(text).lower())


def build_text(title=None, description=None, notes=None, max_chars=3000):
    parts = [p for p in (title, description, notes) if isinstance(p, str) and p.strip()]
    return " ".join(parts)[:max_chars]


class TextModel:
    def __init__(self, alpha=3.0):
        self.vec = TfidfVectorizer(preprocessor=scrub, ngram_range=(1, 2), min_df=20, max_features=50_000,
                                   sublinear_tf=True, dtype=np.float32)
        self.alpha = alpha
        self.ridge = None

    def fit_oof(self, texts, y, groups, n_splits=5):
        """Fit on all training text; return out-of-fold predictions for the training rows."""
        X = self.vec.fit_transform(texts)
        y = np.asarray(y)
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
            oof[va] = Ridge(alpha=self.alpha).fit(X[tr], y[tr]).predict(X[va])
        self.ridge = Ridge(alpha=self.alpha).fit(X, y)
        return oof

    def predict(self, texts):
        return self.ridge.predict(self.vec.transform(texts))

    def top_phrases(self, text, k=4):
        """Phrases in this text that push the price up or down the most."""
        x = self.vec.transform([text]).tocoo()
        names = self.vec.get_feature_names_out()
        contrib = sorted(((names[j] , float(v * self.ridge.coef_[j])) for j, v in zip(x.col, x.data)),
                         key=lambda t: t[1])
        neg = [p for p, c in contrib[:k] if c < 0]
        pos = [p for p, c in reversed(contrib[-k:]) if c > 0]
        return {"raises_value": pos, "lowers_value": neg}

    def save(self, path):
        joblib.dump({"vec": self.vec, "ridge": self.ridge}, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        d = joblib.load(path)
        obj.vec, obj.ridge = d["vec"], d["ridge"]
        return obj
