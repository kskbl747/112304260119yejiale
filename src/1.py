from __future__ import annotations
import csv
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

SCRIPT_DIR = Path(__file__).resolve().parent
LABELED_ZIP = SCRIPT_DIR / "labeledTrainData.tsv.zip"
TEST_ZIP = SCRIPT_DIR / "testData.tsv.zip"
UNLABELED_ZIP = SCRIPT_DIR / "unlabeledTrainData.tsv.zip"
SUBMISSION_PATH = SCRIPT_DIR / "submission.csv"

def strip_html(text: str) -> str:
    return BeautifulSoup(str(text), "html.parser").get_text(" ")

def normalize_word_text(text: str) -> str:
    text = strip_html(text).lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_char_text(text: str) -> str:
    text = strip_html(text).lower()
    return re.sub(r"\s+", " ", text).strip()

def read_tsv_zip(zip_path: Path, sep: str = "\t", skip_bad_lines: bool = False) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        file_name = zf.namelist()[0]
        with zf.open(file_name) as f:
            if skip_bad_lines:
                return pd.read_csv(f, sep=sep, engine="python", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
            return pd.read_csv(f, sep=sep, quoting=csv.QUOTE_MINIMAL)

def scaled_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)

def main():
    print("Loading data...")
    labeled_df = read_tsv_zip(LABELED_ZIP)
    test_df = read_tsv_zip(TEST_ZIP)
    unlabeled_df = read_tsv_zip(UNLABELED_ZIP, skip_bad_lines=True)

    print("Cleaning text...")
    labeled_df["word_text"] = labeled_df["review"].map(normalize_word_text)
    labeled_df["char_text"] = labeled_df["review"].map(normalize_char_text)
    test_df["word_text"] = test_df["review"].map(normalize_word_text)
    test_df["char_text"] = test_df["review"].map(normalize_char_text)
    unlabeled_df["word_text"] = unlabeled_df["review"].map(normalize_word_text)
    unlabeled_df["char_text"] = unlabeled_df["review"].map(normalize_char_text)

    print("Fitting vectorizers...")
    all_word_text = pd.concat([labeled_df["word_text"], test_df["word_text"], unlabeled_df["word_text"]])
    all_char_text = pd.concat([labeled_df["char_text"], test_df["char_text"]])

    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=50000)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=50000)

    vec_word.fit(all_word_text)
    vec_char.fit(all_char_text)

    x_word = vec_word.transform(labeled_df["word_text"])
    x_char = vec_char.transform(labeled_df["char_text"])
    x_test_word = vec_word.transform(test_df["word_text"])
    x_test_char = vec_char.transform(test_df["char_text"])

    y = labeled_df["sentiment"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_c = np.zeros(len(labeled_df))
    oof_s = np.zeros(len(labeled_df))
    test_c = []
    test_s = []

    print("Training...")
    for fold, (tr, va) in enumerate(skf.split(x_word, y)):
        print(f"Fold {fold+1}/5")

        clr = LogisticRegression(C=3, solver="liblinear")
        clr.fit(x_char[tr], y[tr])
        oof_c[va] = clr.predict_proba(x_char[va])[:,1]
        test_c.append(clr.predict_proba(x_test_char)[:,1])

        svc = LinearSVC(C=0.5, max_iter=1000)
        svc.fit(x_word[tr], y[tr])
        oof_s[va] = svc.decision_function(x_word[va])
        test_s.append(svc.decision_function(x_test_word))

    pred = 0.5*scaled_ranks(oof_c) + 0.5*scaled_ranks(oof_s)
    print(f"OOF AUC: {roc_auc_score(y, pred):.6f}")

    final = 0.5*scaled_ranks(np.mean(test_c, axis=0)) + 0.5*scaled_ranks(np.mean(test_s, axis=0))
    pd.DataFrame({"id": test_df["id"], "sentiment": final}).to_csv(SUBMISSION_PATH, index=False)
    print(f"Done! Saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()