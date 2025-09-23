#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orchestrator_from_config.py

- config.yaml (YAML) 파일을 읽어 다음 순서로 실행:
  1) incremental_vit_feature_store_ms.py  (증분 임베딩 + 다분광 대응 + export)
  2) kfold_xgb_eval.py                    (K-Fold 평가)
  3) 전체 데이터로 XGBoost 학습           (holdout 평가 + MLflow 모델/Registry 등록)
- 각 실험 블록(experiments[])을 순차 실행하고, MLflow에 로그/아티팩트 업로드.

필수 동반 파일:
- incremental_vit_feature_store_ms.py
- kfold_xgb_eval.py
같은 폴더에 두고 실행하세요.
"""

import os, sys, json, argparse, subprocess, time
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

HERE = Path(__file__).parent.resolve()

# ---------- shell helpers ----------
def run_cmd(cmd):
    print("[CMD]", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise SystemExit(f"[ERR] command failed: ({p.returncode})")
    return p.stdout

# ---------- kfold summary parser ----------
def parse_summary_text(path: Path):
    if not path.is_file():
        return {}
    t = path.read_text(encoding="utf-8", errors="ignore")
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    m = {}
    for L in lines:
        def grab(key, label):
            if key in L:
                x = L.split(":")[1].strip()
                mean, std = [s.strip() for s in x.split("±")]
                m[f"{label}_mean"] = float(mean)
                m[f"{label}_std"]  = float(std)
        grab("R2  mean±std", "R2")
        grab("MAE mean±std", "MAE")
        if "MAE(steps) mean±std" in L:
            x = L.split(":")[1].strip()
            mean, std = [s.strip() for s in x.split("±")]
            m["MAE_steps_mean"] = float(mean)
            m["MAE_steps_std"]  = float(std)
        if "ACC mean±std" in L or "ACC(rounded) mean±std" in L:
            x = L.split(":")[1].strip()
            mean, std = [s.strip() for s in x.split("±")]
            m["ACC_mean"] = float(mean); m["ACC_std"] = float(std)
        if "F1  mean±std" in L or "F1 (rounded) mean±std" in L:
            x = L.split(":")[1].strip()
            mean, std = [s.strip() for s in x.split("±")]
            m["F1_macro_mean"] = float(mean); m["F1_macro_std"] = float(std)
        if "QWK(rounded) mean±std" in L:
            x = L.split(":")[1].strip()
            mean, std = [s.strip() for s in x.split("±")]
            m["QWK_mean"] = float(mean); m["QWK_std"] = float(std)
    return m

# ---------- export loader ----------
def load_export(vit_out):
    X = np.load(os.path.join(vit_out, "features.npy"))
    meta = pd.read_csv(os.path.join(vit_out, "meta.csv"))
    return X, meta

# ---------- final train + log ----------
def train_full_and_log(vit_out, target, task, reg_mode, xgb_params, registry_name):
    X, meta = load_export(vit_out)
    ok = meta["image_path"].astype(str).str.len() > 0
    ok &= ~meta[target].isna()
    X = X[ok.values]
    meta = meta.loc[ok].reset_index(drop=True)
    y = meta[target]

    if task == "clf":
        if y.dtype.kind not in "iu":
            y, classes = pd.factorize(y.astype(str).str.strip())
            label_map = pd.DataFrame({"label_id": list(range(len(classes))), "label": classes})
        else:
            y = y.astype(int).values
            label_map = None

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        params = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=42)
        params.update(xgb_params or {})
        model = XGBClassifier(**params)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred); f1m = f1_score(yte, pred, average="macro")
        mlflow.log_metric("final_holdout_acc", acc)
        mlflow.log_metric("final_holdout_f1_macro", f1m)
        if label_map is not None:
            p = os.path.join(vit_out, f"label_map_{target}.csv")
            label_map.to_csv(p, index=False, encoding="utf-8-sig")
            mlflow.log_artifact(p, artifact_path="artifacts")
        mlflow.xgboost.log_model(model, "model", registered_model_name=registry_name)

    else:  # regression
        if reg_mode == "ordinal":
            ys = y.astype(str).str.strip()
            classes = sorted(ys.unique())
            mp = {c: i for i, c in enumerate(classes, 1)}
            y_ord = ys.map(mp).astype(int).values
            Xtr, Xte, ytr, yte = train_test_split(X, y_ord, test_size=0.2, random_state=42, stratify=y_ord)
            params = dict(n_estimators=1200, max_depth=8, learning_rate=0.03,
                          subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=42)
            params.update(xgb_params or {})
            model = XGBRegressor(**params)
            model.fit(Xtr, ytr)
            predc = model.predict(Xte)
            K = len(classes)
            roundp = np.clip(np.rint(predc), 1, K).astype(int)
            mae = mean_absolute_error(yte, roundp)
            r2 = r2_score(yte, predc)
            mlflow.log_metric("final_holdout_mae_steps", mae)
            mlflow.log_metric("final_holdout_r2_ordinal", r2)
            lm = pd.DataFrame({"label_id": list(range(1, K + 1)), "label": classes})
            p = os.path.join(vit_out, f"label_map_{target}_ordinal.csv")
            lm.to_csv(p, index=False, encoding="utf-8-sig")
            mlflow.log_artifact(p, artifact_path="artifacts")
            mlflow.xgboost.log_model(model, "model", registered_model_name=registry_name)
        else:
            ycont = pd.to_numeric(y, errors="coerce").values
            Xtr, Xte, ytr, yte = train_test_split(X, ycont, test_size=0.2, random_state=42)
            params = dict(n_estimators=1200, max_depth=8, learning_rate=0.03,
                          subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=42)
            params.update(xgb_params or {})
            model = XGBRegressor(**params)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            r2 = r2_score(yte, pred); mae = mean_absolute_error(yte, pred)
            mlflow.log_metric("final_holdout_r2", r2)
            mlflow.log_metric("final_holdout_mae", mae)
            mlflow.xgboost.log_model(model, "model", registered_model_name=registry_name)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--python_exec", default=sys.executable, help="(선택) 파이썬 실행기")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    # --- 공통 루트/MLflow 설정 ---
    base = cfg.get("base", {})
    dataset_root = base["dataset_root"]
    store_path   = base["store_path"]
    mlflow_uri   = base.get("mlflow_uri", "http://127.0.0.1:5000")
    mlflow_exp   = base.get("mlflow_experiment", "vit_xgb_pipeline")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_exp)

    # --- 실행 스크립트 경로 검증 ---
    inc = str(HERE / "incremental_vit_feature_store_ms.py")
    kfold = str(HERE / "kfold_xgb_eval.py")
    if not Path(inc).is_file() or not Path(kfold).is_file():
        raise SystemExit("[ERR] incremental_vit_feature_store_ms.py / kfold_xgb_eval.py 를 같은 폴더에 두세요.")

    # --- 여러 실험 반복 실행 ---
    for exp in cfg.get("experiments", []):
        name = exp["name"]
        print(f"\n========== RUN: {name} ==========")
        export_dir = exp.get("export_dir", f"./vit_out_{name}")
        Path(export_dir).mkdir(parents=True, exist_ok=True)

        # MLflow Run 시작
        run_name = exp.get("run_name", name)
        with mlflow.start_run(run_name=run_name):
            # 파라미터 로깅 (config 전체를 기록)
            mlflow.log_param("config_name", Path(args.config).name)
            mlflow.log_params({
                "name": name,
                **exp.get("embed", {}),
                **exp.get("eval", {}),
                **exp.get("final_train", {}),
            })
            mlflow.set_tags({
                "orchestrator": "config",
                "export_dir": export_dir,
                "dataset_root": dataset_root,
                "store_path": store_path,
            })

            # 1) 증분 임베딩 + export
            em = exp.get("embed", {})
            cmd_inc = [
                args.python_exec, inc,
                "--dataset_root", dataset_root,
                "--store_path",   store_path,
                "--model_name",   em.get("model_name", "google/vit-base-patch16-224-in21k"),
                "--feature_type", em.get("feature_type", "cls_mean"),
                "--preprocess_mode", em.get("preprocess_mode", "auto"),
                "--ms_mode", em.get("ms_mode", "per_band"),
                "--ms_wavelengths", em.get("ms_wavelengths", "430,540"),
                "--ms_fill", em.get("ms_fill", "nd"),
                "--batch_size", str(em.get("batch_size", 16)),
                "--device", em.get("device", "cuda"),
                "--export",
                "--export_dir", export_dir,
            ]
            if em.get("save_debug_vis", 0) > 0:
                cmd_inc += ["--save_debug_vis", str(em["save_debug_vis"])]
            run_cmd(cmd_inc)

            # 2) K-Fold 평가
            ev = exp.get("eval", {})
            cmd_kf = [
                args.python_exec, kfold,
                "--vit_out", export_dir,
                "--target",  ev["target"],
                "--task",    ev.get("task", "reg"),
                "--reg_mode", ev.get("reg_mode", "auto"),
                "--n_splits", str(ev.get("n_splits", 5)),
            ]
            if ev.get("aggregate_by"):
                cmd_kf += ["--aggregate_by", ev["aggregate_by"]]
            if ev.get("group_by"):
                cmd_kf += ["--group_by", ev["group_by"]]
            if ev.get("xgb_params"):
                cmd_kf += ["--xgb_params", json.dumps(ev["xgb_params"])]
            # pca 옵션 추가
            pca_cfg = ev.get("pca", {})
            if isinstance(pca_cfg, dict) and pca_cfg.get("enabled", False):
                cmd_kf += ["--use_pca"]
                if "dim" in pca_cfg:
                    cmd_kf += ["--pca_dim", str(pca_cfg["dim"])]
                if pca_cfg.get("whiten", False):
                    cmd_kf += ["--pca_whiten"]
                if "random_state" in pca_cfg:
                    cmd_kf += ["--pca_random_state", str(pca_cfg["random_state"])]
                
            run_cmd(cmd_kf)

            # 3) 요약 파일 파싱 + MLflow 기록
            if ev.get("task", "reg") == "clf":
                summary = Path(export_dir) / f"kfold_{ev['target']}_clf_summary.txt"
            else:
                cand1 = Path(export_dir) / f"kfold_{ev['target']}_reg_ordinal_summary.txt"
                cand2 = Path(export_dir) / f"kfold_{ev['target']}_reg_cont_summary.txt"
                summary = cand1 if cand1.is_file() else cand2
            mets = parse_summary_text(summary)
            for k, v in mets.items():
                mlflow.log_metric(k, float(v))
            if summary.is_file():
                mlflow.log_artifact(str(summary), artifact_path="reports")

            # 4) 전체 학습 + 모델 Registry 등록
            ft = exp.get("final_train", {})
            registry_name = ft.get("registry_name", f"{name}".replace(" ", "_"))
            xgb_params = ft.get("xgb_params", {})
            train_full_and_log(
                vit_out=export_dir,
                target=ev["target"],
                task=ev.get("task", "reg"),
                reg_mode=ev.get("reg_mode", "auto"),
                xgb_params=xgb_params,
                registry_name=registry_name
            )

            # 5) 리포트 JSON 저장 + 업로드
            report = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "name": name,
                "dataset_root": dataset_root,
                "store_path": store_path,
                "export_dir": export_dir,
                "embed": em,
                "eval": ev,
                "final_train": ft,
                "metrics": mets,
            }
            rp = Path(export_dir) / f"report_{name}_{report['timestamp']}.json"
            rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(rp), artifact_path="reports")
            print(f"[Saved] {rp}")
            print(f"[MLflow] run finished for {name}")

if __name__ == "__main__":
    main()
