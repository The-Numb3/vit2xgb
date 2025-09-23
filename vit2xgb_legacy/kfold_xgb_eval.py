#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kfold_xgb_eval.py

- 입력: vit_out/  (prepare_and_extract_vit_cls.py 산출물)
    - features.npy : (N, D)  ViT 특징
    - meta.csv     : image_name, image_path, (라벨들), slaughter/session/day 등

지원:
- --task {clf, reg}
- 회귀 모드: --reg_mode {auto, continuous, ordinal}
- KFold/StratifiedKFold/GroupKFold/StratifiedGroupKFold (가능 시) 자동 선택
- 그룹 분할 옵션: --group_by "slaughter" 또는 "slaughter,day" 등

저장:
- vit_out/kfold_<target>_<task>_[ordinal|cont]_metrics.csv  (fold별 지표)
- vit_out/kfold_<target>_<task>_[ordinal|cont]_summary.txt  (평균±표준편차)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
try:
    # sklearn>=1.3
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, f1_score
)
from xgboost import XGBRegressor, XGBClassifier

# ----------------------- 유틸 -----------------------

def ensure_files(vit_out: str):
    feat = os.path.join(vit_out, "features.npy")
    meta = os.path.join(vit_out, "meta.csv")
    if not (os.path.isfile(feat) and os.path.isfile(meta)):
        sys.exit(f"[ERR] features.npy / meta.csv를 찾지 못했습니다.\n  확인 경로: {vit_out}")
    return feat, meta

def is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s)
        return True
    except Exception:
        return False

def to_ordinal(y_raw: pd.Series):
    y_s = y_raw.astype(str).str.strip()
    classes = sorted(y_s.unique(), key=lambda x: x)
    mapping = {c: i for i, c in enumerate(classes, start=1)}  # 1..K
    y_ord = y_s.map(mapping).astype(np.int32).values
    return y_ord, classes, mapping

def from_ordinal(y_pred_cont: np.ndarray, classes):
    K = len(classes)
    y_rounded = np.clip(np.rint(y_pred_cont), 1, K).astype(np.int32)
    inv = {i+1: c for i, c in enumerate(classes)}
    y_lbl = np.array([inv[i] for i in y_rounded])
    return y_lbl, y_rounded

def quadratic_weighted_kappa(y_true_1K: np.ndarray, y_pred_1K: np.ndarray, K: int) -> float:
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

def parse_groups(meta: pd.DataFrame, group_by: str):
    if not group_by:
        return None
    cols = [c.strip() for c in group_by.split(",") if c.strip()]
    for c in cols:
        if c not in meta.columns:
            raise SystemExit(f"[ERR] group_by={c} 컬럼을 meta.csv에서 찾지 못했습니다.")
    if len(cols) == 1:
        return meta[cols[0]].astype(str).values
    # 여러 컬럼 결합
    return (meta[cols].astype(str).agg("__".join, axis=1)).values

# ----------------------- 메인 -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit_out", required=True, help="vit_out 디렉토리")
    ap.add_argument("--target", required=True, help="타깃 컬럼명 (예: 등급, Total 등)")
    ap.add_argument("--task", choices=["reg", "clf"], default="reg")
    ap.add_argument("--reg_mode", choices=["auto","continuous","ordinal"], default="auto")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_by", type=str, default="", help="그룹 분할 키 (예: 'slaughter' 또는 'slaughter,day')")
    ap.add_argument("--aggregate_by", type=str, default="", help="그룹 풀링 키. 예: 'slaughter,session,day' 또는 'slaughter,day'")
    ap.add_argument("--xgb_params", type=str, default="", help="추가 XGB 파라미터 JSON 문자열(optional)")
    args = ap.parse_args()

    feat_path, meta_path = ensure_files(args.vit_out)
    X = np.load(feat_path)
    meta = pd.read_csv(meta_path)

    if args.target not in meta.columns:
        sys.exit(f"[ERR] meta.csv에 '{args.target}' 컬럼 없음. 현재 컬럼 예시: {list(meta.columns)[:20]}...")

    # 유효 샘플 필터
    ok = meta["image_path"].astype(str).str.len() > 0
    ok &= ~meta[args.target].isna()
    X = X[ok.values]
    y_raw = meta.loc[ok, args.target]
    groups = parse_groups(meta.loc[ok], args.group_by)

    out_dir = Path(args.vit_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 공통 XGB 파라미터 (필요 시 CLI로 override)
    params_reg = dict(
        n_estimators=1200, max_depth=8, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=args.seed
    )
    params_clf = dict(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=args.seed
    )
    if args.xgb_params:
        import json
        override = json.loads(args.xgb_params)
        if args.task == "clf":
            params_clf.update(override)
        else:
            params_reg.update(override)

    # ---------------- 분류 ----------------
    if args.task == "clf":
        # y 인코딩
        if not is_numeric_series(y_raw):
            y, classes = pd.factorize(y_raw.astype(str).str.strip())
        else:
            y = y_raw.astype(int).values
            classes = np.unique(y)
        # 분할자 결정
        if groups is not None and HAS_SGKF:
            splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            splits = splitter.split(X, y, groups=groups)
        elif groups is not None:
            # SGKF가 없으면 그룹만 보장 (레이블 분포는 조금 틀어질 수 있음)
            splitter = GroupKFold(n_splits=args.n_splits)
            splits = splitter.split(X, y, groups=groups)
        else:
            splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            splits = splitter.split(X, y)

        accs, f1s = [], []
        rows = []
        for fold, (tr, te) in enumerate(splits, start=1):
            model = XGBClassifier(**params_clf)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            acc = accuracy_score(y[te], pred)
            f1m = f1_score(y[te], pred, average="macro")
            accs.append(acc); f1s.append(f1m)
            rows.append({"fold": fold, "ACC": acc, "F1_macro": f1m})

        metric_csv = out_dir / f"kfold_{args.target}_clf_metrics.csv"
        pd.DataFrame(rows).to_csv(metric_csv, index=False, encoding="utf-8-sig")
        summary = (
            f"KFOLD ({args.n_splits}) - target={args.target} [clf]\n"
            f"ACC mean±std : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n"
            f"F1  mean±std : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n"
        )
        print(summary)
        (out_dir / f"kfold_{args.target}_clf_summary.txt").write_text(summary, encoding="utf-8")
        return

    # ---------------- 회귀 ----------------
    # reg_mode 결정
    reg_mode = args.reg_mode
    if reg_mode == "auto":
        if not is_numeric_series(y_raw):
            reg_mode = "ordinal"
        else:
            y_num = pd.to_numeric(y_raw, errors="coerce")
            uniq = np.unique(y_num.dropna())
            if (np.allclose(uniq, uniq.astype(int))) and (len(uniq) <= 10):
                reg_mode = "ordinal"
            else:
                reg_mode = "continuous"

    if reg_mode == "continuous":
        y = pd.to_numeric(y_raw, errors="coerce").values
        # 분할자: 그룹 있으면 GroupKFold, 없으면 KFold
        if groups is not None:
            splitter = GroupKFold(n_splits=args.n_splits)
            splits = splitter.split(X, y, groups=groups)
        else:
            splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            splits = splitter.split(X, y)

        r2s, maes = [], []
        rows = []
        for fold, (tr, te) in enumerate(splits, start=1):
            model = XGBRegressor(**params_reg)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            r2 = r2_score(y[te], pred)
            mae = mean_absolute_error(y[te], pred)
            r2s.append(r2); maes.append(mae)
            rows.append({"fold": fold, "R2": r2, "MAE": mae})

        metric_csv = out_dir / f"kfold_{args.target}_reg_cont_metrics.csv"
        pd.DataFrame(rows).to_csv(metric_csv, index=False, encoding="utf-8-sig")
        summary = (
            f"KFOLD ({args.n_splits}) - target={args.target} [reg-continuous]\n"
            f"R2  mean±std : {np.mean(r2s):.4f} ± {np.std(r2s):.4f}\n"
            f"MAE mean±std: {np.mean(maes):.4f} ± {np.std(maes):.4f}\n"
        )
        print(summary)
        (out_dir / f"kfold_{args.target}_reg_cont_summary.txt").write_text(summary, encoding="utf-8")
        return

    # ordinal regression
    y_ord, classes, mapping = to_ordinal(y_raw)
    # 분할자: 그룹 있으면 StratifiedGroupKFold(가능시)→없으면 GroupKFold, 없으면 StratifiedKFold
    if groups is not None and HAS_SGKF:
        splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = splitter.split(X, y_ord, groups=groups)
    elif groups is not None:
        splitter = GroupKFold(n_splits=args.n_splits)
        splits = splitter.split(X, y_ord, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = splitter.split(X, y_ord)

    maes_steps, r2s, accs, f1s, qwks = [], [], [], [], []
    rows = []
    K = len(classes)
    for fold, (tr, te) in enumerate(splits, start=1):
        model = XGBRegressor(**params_reg)
        model.fit(X[tr], y_ord[tr])

        pred_cont = model.predict(X[te])
        pred_lbl, pred_1K = from_ordinal(pred_cont, classes)
        true_lbl, true_1K = from_ordinal(y_ord[te], classes)

        mae_steps = mean_absolute_error(true_1K, pred_1K)  # 등급 스텝 단위 MAE
        r2 = r2_score(y_ord[te], pred_cont)
        acc = accuracy_score(true_lbl, pred_lbl)
        f1m = f1_score(true_lbl, pred_lbl, average="macro")
        qwk = quadratic_weighted_kappa(true_1K, pred_1K, K)

        maes_steps.append(mae_steps); r2s.append(r2); accs.append(acc); f1s.append(f1m); qwks.append(qwk)
        rows.append({
            "fold": fold,
            "MAE_steps": mae_steps, "R2_ord": r2,
            "ACC_rounded": acc, "F1_macro_rounded": f1m, "QWK_rounded": qwk
        })

    metric_csv = out_dir / f"kfold_{args.target}_reg_ordinal_metrics.csv"
    pd.DataFrame(rows).to_csv(metric_csv, index=False, encoding="utf-8-sig")
    summary = (
        f"KFOLD ({args.n_splits}) - target={args.target} [reg-ordinal]\n"
        f"MAE(steps) mean±std : {np.mean(maes_steps):.4f} ± {np.std(maes_steps):.4f}\n"
        f"R2(ordinal) mean±std: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}\n"
        f"ACC(rounded) mean±std: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n"
        f"F1 (rounded) mean±std: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n"
        f"QWK(rounded) mean±std: {np.mean(qwks):.4f} ± {np.std(qwks):.4f}\n"
    )
    print(summary)
    (out_dir / f"kfold_{args.target}_reg_ordinal_summary.txt").write_text(summary, encoding="utf-8")
    # === 중략: 기존 import/유틸 그대로 ===
    def aggregate_by_keys(X: np.ndarray, meta: pd.DataFrame, keys: list, target: str, task: str):
        """
        같은 (keys) 조합의 행들을 하나의 샘플로 집계.
        - X: (N,D) -> (G,D)  평균
        - y: 회귀면 평균, 분류면 최빈/첫값(동일해야 정상)
        - meta_out: 그룹 대표 메타(첫 행 기준) + count
        """
        if not keys:  # 집계 안 함
            y = meta[target].values
            return X, y, meta.copy()

        df = meta.copy()
        df["_grp"] = df[keys].astype(str).agg("__".join, axis=1)

        # 인덱스 그룹
        groups = df.groupby("_grp").indices  # dict: grp -> [idx...]
        grp_ids = list(groups.keys())
        D = X.shape[1]
        Xg = np.zeros((len(grp_ids), D), dtype=X.dtype)
        yg = []

        meta_rows = []
        for gi, gid in enumerate(grp_ids):
            idx = np.fromiter(groups[gid], dtype=int)
            Xg[gi] = X[idx].mean(axis=0)
            if task == "reg":
                yg.append(pd.to_numeric(df.loc[idx, target], errors="coerce").mean())
            else:
                # 분류: 동일 라벨 가정. 다르면 최빈값.
                vals = df.loc[idx, target].astype(str)
                yg.append(vals.mode().iat[0])

            # 대표 메타(첫 행)
            rep = df.loc[idx[0]].to_dict()
            rep["group_key"] = gid
            rep["group_count"] = len(idx)
            meta_rows.append(rep)

        y_out = np.array(yg)
        meta_out = pd.DataFrame(meta_rows)
        return Xg, y_out, meta_out

    # --- main() 내부에 아래 변경/추가 ---
    # 1) 인자 추가
    ap.add_argument("--aggregate_by", type=str, default="",
                    help="그룹 풀링 키. 예: 'slaughter,session,day' 또는 'slaughter,day'")

    # 2) 파일 로드 후 유효샘플 필터까지 기존과 동일...
    #    그 다음에 아래 로직 추가

    # 집계 키 파싱
    agg_keys = [c.strip() for c in args.aggregate_by.split(",") if c.strip()]
    for c in agg_keys:
        if c and c not in meta.columns:
            sys.exit(f"[ERR] aggregate_by의 '{c}' 컬럼을 meta.csv에서 찾지 못했습니다.")

    # 원본 메타(필터 후)
    meta_ok = meta.loc[ok].reset_index(drop=True)

    # 집계 수행 (분류/회귀 공통)
    task_for_agg = "reg" if args.task == "reg" else "clf"
    X, y_agg, meta_after = aggregate_by_keys(X, meta_ok, agg_keys, args.target, task_for_agg)

    # 그룹 분할용 group id도 집계 기준으로 교체
    if args.group_by:
        # 집계 후 group_by도 동일 키를 meta_after에서 참조
        group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
        for c in group_cols:
            if c not in meta_after.columns:
                sys.exit(f"[ERR] group_by의 '{c}' 컬럼을 (집계 후) meta에서 찾지 못했습니다.")
        if len(group_cols) == 1:
            groups = meta_after[group_cols[0]].astype(str).values
        else:
            groups = meta_after[group_cols].astype(str).agg("__".join, axis=1).values
    else:
        groups = None

    # 이후 분기에서 y_raw 대신 y_agg를 사용:
    # - 회귀: y = y_agg (continuous / ordinal 판단은 기존 로직 그대로)
    # - 분류: y_raw = pd.Series(y_agg)

if __name__ == "__main__":
    main()
