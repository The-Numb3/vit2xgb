#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, subprocess, sys
from pathlib import Path
import yaml

HERE = Path(__file__).parent.resolve()

def run_cmd(cmd):
    print("[CMD]", " ".join(map(str, cmd)))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        sys.exit(f"[ERR] command failed: ({p.returncode})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--python_exec", default=sys.executable)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # ---- 경로 설정
    dataset_root = Path(cfg["base"]["dataset_root"]).resolve()
    processed_dir = Path(cfg["base"]["processed_dir"]).resolve()
    export_dir = Path(cfg["base"]["export_dir"]).resolve()
    device = cfg["base"].get("device", "cpu")
    batch_size = int(cfg["base"].get("batch_size", 16))
    overwrite = 1 if cfg.get("preprocess", {}).get("overwrite", False) else 0

    # ---- 스크립트 경로
    preproc_py = str(HERE / "preprocess_crop_ingest.py")
    embed_py   = str(HERE / "embed_features.py")
    kfold_py   = str(HERE.parent / "kfold_xgb_eval.py")

    # =========================
    # 1) 전처리 (증분)
    # =========================
    print("\n========== STEP 1: preprocess (incremental) ==========")
    cmd1 = [
        args.python_exec, preproc_py,
        "--dataset_root", str(dataset_root),
        "--out_dir", str(processed_dir),
        "--overwrite", str(overwrite),
    ]
    run_cmd(cmd1)

    # =========================
    # 2) 임베딩 (증분, config 사용)
    # =========================
    print("\n========== STEP 2: embedding (incremental) ==========")
    # embed_features_from_config.py 가 config 전체를 읽어 처리
    cmd2 = [args.python_exec, embed_py, "--config", str(cfg_path)]
    run_cmd(cmd2)

    # =========================
    # 3) K-Fold 평가 (여기에서만 PCA 전달)
    # =========================
    print("\n========== STEP 3: k-fold evaluation ==========")
    ev = cfg.get("eval", {})
    # 필수
    cmd3 = [
        args.python_exec, kfold_py,
        "--vit_out", str(export_dir),
        "--target",  str(ev["target"]),
        "--task",    str(ev.get("task", "reg")),
        "--reg_mode",str(ev.get("reg_mode", "auto")),
        "--n_splits",str(ev.get("n_splits", 5)),
    ]
    # 선택
    if ev.get("group_by"):     cmd3 += ["--group_by",     str(ev["group_by"])]
    if ev.get("aggregate_by"): cmd3 += ["--aggregate_by", str(ev["aggregate_by"])]
    if ev.get("xgb_params"):
        cmd3 += ["--xgb_params", json.dumps(ev["xgb_params"], ensure_ascii=False)]
    # ✅ PCA는 여기서만 반영
    pca_cfg = ev.get("pca", {})
    if isinstance(pca_cfg, dict) and pca_cfg.get("enabled", False):
        cmd3 += ["--use_pca", "--pca_dim", str(pca_cfg.get("dim", 32))]
        if pca_cfg.get("whiten", False):
            cmd3 += ["--pca_whiten"]
        if "random_state" in pca_cfg:
            cmd3 += ["--pca_random_state", str(pca_cfg["random_state"])]

    run_cmd(cmd3)

    print("\n✅ Pipeline done.")
    print(f" - processed_dir: {processed_dir}")
    print(f" - export_dir   : {export_dir}")

if __name__ == "__main__":
    main()
