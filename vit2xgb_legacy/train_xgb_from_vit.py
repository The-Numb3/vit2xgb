#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_xgb_from_vit.py

- 입력: vit_out/  (prepare_and_extract_vit_cls.py 산출물)
    - features.npy : (N, D=768) ViT CLS (또는 확장 특징)
    - meta.csv     : image_name, image_path, 라벨(등급/Total/Marbling 등)

- 사용 예:
  # 분류(등급)
  python train_xgb_from_vit.py --vit_out ".\\vit_out" --target "등급" --task clf

  # 회귀(서열 자동판단: 숫자 아님/소수의 정수 → ordinal, 그 외 → continuous)
  python train_xgb_from_vit.py --vit_out ".\\vit_out" --target "등급" --task reg

  # 회귀(강제 서열)
  python train_xgb_from_vit.py --vit_out ".\\vit_out" --target "등급" --task reg --reg_mode ordinal

  # 회귀(연속)
  python train_xgb_from_vit.py --vit_out ".\\vit_out" --target "Total" --task reg --reg_mode continuous
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBRegressor, XGBClassifier
import joblib


# -------------------------- 유틸 --------------------------

def ensure_files(vit_out: str):
    feat = os.path.join(vit_out, "features.npy")
    meta = os.path.join(vit_out, "meta.csv")
    if not (os.path.isfile(feat) and os.path.isfile(meta)):
        sys.exit(f"[ERR] features.npy / meta.csv를 찾지 못했습니다.\n  확인 경로: {vit_out}\n  예) --vit_out .\\vit_out")
    return feat, meta

def is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s)
        return True
    except Exception:
        return False

def to_ordinal(y_raw: pd.Series):
    """
    문자열/혼합형 라벨을 1..K 서열로 매핑
    - 자동 정렬(문자 사전순). 필요하면 여기서 커스텀 순서를 반영 가능.
    반환: y_ord(np.int32), classes(list[str]), mapping(dict[str->int])
    """
    y_s = y_raw.astype(str).str.strip()
    classes = sorted(y_s.unique(), key=lambda x: x)
    mapping = {c: i for i, c in enumerate(classes, start=1)}  # 1..K
    y_ord = y_s.map(mapping).astype(np.int32).values
    return y_ord, classes, mapping

def from_ordinal(y_pred_cont: np.ndarray, classes):
    """
    연속 예측값을 1..K로 반올림+클리핑 → 원래 라벨 문자열로 역매핑
    """
    K = len(classes)
    y_rounded = np.clip(np.rint(y_pred_cont), 1, K).astype(np.int32)
    inv = {i+1: c for i, c in enumerate(classes)}
    y_lbl = np.array([inv[i] for i in y_rounded])
    return y_lbl, y_rounded


def quadratic_weighted_kappa(y_true_1K: np.ndarray, y_pred_1K: np.ndarray, K: int) -> float:
    """
    QWK 계산 (y_true, y_pred는 1..K 정수)
    """
    O = np.zeros((K, K), dtype=np.float64)
    for t, p in zip(y_true_1K, y_pred_1K):
        O[t-1, p-1] += 1

    W = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            W[i, j] = ((i - j) ** 2) / ((K - 1) ** 2) if K > 1 else 0.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / max(O.sum(), 1.0)

    num = (W * O).sum()
    den = (W * E).sum() if (W * E).sum() != 0 else 1.0
    return 1.0 - (num / den)


# ----------------------- 메인 로직 ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit_out", required=True, help="vit_out 디렉터리 경로")
    ap.add_argument("--target", required=True, help="타깃 컬럼명 (예: 등급, Total 등)")
    ap.add_argument("--task", choices=["reg", "clf"], default="reg", help="reg(회귀) / clf(분류)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reg_mode", choices=["auto", "continuous", "ordinal"], default="auto",
                    help="회귀 모드: auto(기본), continuous(연속), ordinal(서열)")
    args = ap.parse_args()

    feat_path, meta_path = ensure_files(args.vit_out)
    X = np.load(feat_path)
    meta = pd.read_csv(meta_path)

    if args.target not in meta.columns:
        sys.exit(f"[ERR] meta.csv에 '{args.target}' 컬럼이 없습니다. 현재 컬럼 예시: {list(meta.columns)[:20]} ...")

    # 누락 필터링
    ok = meta["image_path"].astype(str).str.len() > 0
    ok &= ~meta[args.target].isna()
    X = X[ok.values]
    y_raw = meta.loc[ok, args.target]

    out_dir = Path(args.vit_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 분류 ----------------
    if args.task == "clf":
        # 문자형이면 factorize, 숫자형이면 그대로
        if not is_numeric_series(y_raw):
            y, classes = pd.factorize(y_raw.astype(str).str.strip())
            label_map_path = out_dir / f"label_map_{args.target}.csv"
            pd.DataFrame({"label_id": range(len(classes)), "label": classes}) \
              .to_csv(label_map_path, index=False, encoding="utf-8-sig")
            print(f"[Saved] {label_map_path}")
        else:
            # 숫자형이어도 다중분류를 원한다면 정수로 취급
            y = y_raw.astype(int).values
            classes = np.unique(y)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        model = XGBClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=args.seed
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")
        print(f"ACC : {acc:.4f}")
        print(f"F1  : {f1m:.4f}")

        # 리포트/혼동행렬 저장
        report_txt = classification_report(y_te, y_pred, digits=4)
        print(report_txt)
        (out_dir / f"report_{args.target}_clf.txt").write_text(report_txt, encoding="utf-8")

        cm = confusion_matrix(y_te, y_pred)
        pd.DataFrame(cm).to_csv(out_dir / f"confusion_matrix_{args.target}.csv", index=False, encoding="utf-8-sig")
        print(f"[Saved] confusion_matrix_{args.target}.csv")

        # 모델 저장
        model_path = out_dir / f"xgb_{args.target}_clf.joblib"
        joblib.dump(model, model_path)
        print(f"[Saved] {model_path}")

        # 중요도 저장
        if getattr(model, "feature_importances_", None) is not None:
            pd.DataFrame({
                "feature_idx": range(len(model.feature_importances_)),
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False) \
             .to_csv(out_dir / f"feat_importance_{args.target}.csv", index=False, encoding="utf-8-sig")
            print(f"[Saved] feat_importance_{args.target}.csv")

        # 예측 저장
        pd.DataFrame({"y_true": y_te, "y_pred": y_pred}).to_csv(
            out_dir / f"eval_{args.target}_clf.csv", index=False, encoding="utf-8-sig"
        )

        return

    # ---------------- 회귀 ----------------
    # reg_mode 결정
    reg_mode = args.reg_mode
    if reg_mode == "auto":
        if not is_numeric_series(y_raw):
            reg_mode = "ordinal"
        else:
            # 숫자형인데 고유값이 적은 '정수' 등급처럼 보이면 ordinal
            y_num = pd.to_numeric(y_raw, errors="coerce")
            unique_vals = np.unique(y_num.dropna())
            if (np.allclose(unique_vals, unique_vals.astype(int))) and (len(unique_vals) <= 10):
                reg_mode = "ordinal"
            else:
                reg_mode = "continuous"

    if reg_mode == "continuous":
        y = pd.to_numeric(y_raw, errors="coerce").values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )
        model = XGBRegressor(
            n_estimators=1200, max_depth=8, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=args.seed
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred = model.predict(X_te)

        print(f"R2  : {r2_score(y_te, y_pred):.4f}")
        print(f"MAE : {mean_absolute_error(y_te, y_pred):.4f}")

        # 저장
        model_path = out_dir / f"xgb_{args.target}_reg_cont.joblib"
        joblib.dump(model, model_path)
        print(f"[Saved] {model_path}")

        if getattr(model, "feature_importances_", None) is not None:
            pd.DataFrame({
                "feature_idx": range(len(model.feature_importances_)),
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False) \
             .to_csv(out_dir / f"feat_importance_{args.target}_cont.csv", index=False, encoding="utf-8-sig")
            print(f"[Saved] feat_importance_{args.target}_cont.csv")

        pd.DataFrame({"y_true": y_te, "y_pred": y_pred}).to_csv(
            out_dir / f"eval_{args.target}_reg_cont.csv", index=False, encoding="utf-8-sig"
        )
        return

    # ----- ordinal regression -----
    # 문자열/숫자 모두 서열화해서 1..K
    if not is_numeric_series(y_raw):
        y_ord, classes, mapping = to_ordinal(y_raw)
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce")
        # 숫자인데 정수/유니크 작으면 등급으로 취급, 그 외엔 강제 ordinal이므로 클래스 생성
        uniq = np.unique(y_num.dropna())
        if len(uniq) <= 50:  # 너무 많으면 자동으로는 등급이 아닐 가능성 큼(하지만 reg_mode가 ordinal이니 생성)
            classes = [str(int(u)) if float(u).is_integer() else str(u) for u in sorted(uniq)]
        else:
            classes = [str(int(u)) if float(u).is_integer() else str(u) for u in sorted(uniq)]
        mapping = {c: i for i, c in enumerate(classes, start=1)}
        y_ord = pd.Series(y_num).map(lambda v: mapping.get(str(int(v)) if float(v).is_integer() else str(v))).astype(int).values

    # 저장: 라벨 맵
    pd.DataFrame({"label_id": range(1, len(classes)+1), "label": classes}) \
      .to_csv(out_dir / f"label_map_{args.target}_ordinal.csv", index=False, encoding="utf-8-sig")
    print(f"[Saved] label_map_{args.target}_ordinal.csv")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_ord, test_size=args.test_size, random_state=args.seed, stratify=y_ord
    )

    model = XGBRegressor(
        n_estimators=1200, max_depth=8, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=args.seed
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred_cont = model.predict(X_te)

    # 회귀 지표(서열 공간에서)
    y_pred_round = np.clip(np.rint(y_pred_cont), 1, len(classes)).astype(int)
    mae_steps = mean_absolute_error(y_te, y_pred_round)  # 등급 스텝 단위 MAE
    r2_ord    = r2_score(y_te, y_pred_cont)
    print(f"MAE (in grade steps): {mae_steps:.3f}")
    print(f"R2  (on ordinal target): {r2_ord:.4f}")

    # 등급 문자열로 변환 후 ACC/F1/QWK
    y_pred_lbl, y_pred_1K = from_ordinal(y_pred_cont, classes)
    y_te_lbl,  y_te_1K    = from_ordinal(y_te, classes)
    acc = accuracy_score(y_te_lbl, y_pred_lbl)
    f1m = f1_score(y_te_lbl, y_pred_lbl, average="macro")
    qwk = quadratic_weighted_kappa(y_te_1K, y_pred_1K, len(classes))
    print(f"ACC (rounded grades): {acc:.4f}")
    print(f"F1  (macro, rounded): {f1m:.4f}")
    print(f"QWK (rounded)       : {qwk:.4f}")

    # 저장
    model_path = out_dir / f"xgb_{args.target}_reg_ordinal.joblib"
    joblib.dump(model, model_path)
    print(f"[Saved] {model_path}")

    if getattr(model, "feature_importances_", None) is not None:
        pd.DataFrame({
            "feature_idx": range(len(model.feature_importances_)),
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False) \
         .to_csv(out_dir / f"feat_importance_{args.target}_ordinal.csv", index=False, encoding="utf-8-sig")
        print(f"[Saved] feat_importance_{args.target}_ordinal.csv")

    pd.DataFrame({
        "y_true_ord": y_te, "y_pred_cont": y_pred_cont,
        "y_pred_ord": y_pred_1K, "y_true_lbl": y_te_lbl, "y_pred_lbl": y_pred_lbl
    }).to_csv(out_dir / f"eval_{args.target}_reg_ordinal.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
