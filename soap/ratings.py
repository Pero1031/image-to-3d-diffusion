"""石けんの解析で共通に使うもの: 評定の読み込みと、個体ごとの交差検証。"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
RATINGS = Path("D:/project/vision-experiments/Tobii_pro/data/master_trial_summary.csv")
ALPHAS = np.logspace(-1, 5, 13)


def load():
    """(特徴の npz, 写真の名前, 個体 ID, 参加者ごとに標準化した評定 Z (参加者, 写真), 写真ごとの平均 y)。"""
    data = np.load(ROOT / "data" / "soap_features.npz")
    names = list(data["names"])
    groups = np.array([n.split("_")[0] for n in names])
    rows = list(csv.DictReader(open(RATINGS, encoding="utf-8-sig")))
    people = sorted({r["participant_id"] for r in rows})
    R = np.full((len(people), len(names)), np.nan)
    for r in rows:
        R[people.index(r["participant_id"]), names.index(r["image_name"])] = float(r["rating"])
    Z = (R - np.nanmean(R, 1, keepdims=True)) / np.nanstd(R, 1, keepdims=True)
    return data, names, groups, Z, np.nanmean(Z, 0)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def cv_predict(X, target, groups):
    """個体を 1 つずつ評価側に回す交差検証（同じ石けんの 2 枚は必ず一緒）。評価側の予測を返す。"""
    pred = np.zeros(len(target))
    for tr, te in LeaveOneGroupOut().split(X, target, groups):
        pred[te] = ridge().fit(X[tr], target[tr]).predict(X[te])
    return pred
