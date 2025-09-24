#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image

import torch

# ===== 공용 유틸 =====
def device_autoselect(user_device: str) -> str:
    if user_device.lower().startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA not available. Falling back to CPU.")
        return "cpu"
    return user_device

def pil_open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

# ===== 멀티스펙트럼 보조 =====
def fuse_to_rgb(paths_by_wl: Dict[int, str], wls: List[int], fill: str="nd") -> Image.Image:
    assert len(wls) == 2, "fuse_rgb requires exactly 2 wavelengths"
    from numpy import array, float32, uint8, stack
    w1, w2 = wls
    def pick_img(w):
        return pil_open_rgb(paths_by_wl[w]) if w in paths_by_wl else None
    im1, im2 = pick_img(w1), pick_img(w2)
    if im1 is None and im2 is None:
        raise ValueError("No available bands for fuse_rgb")
    if im1 is None: im1 = im2.copy()
    if im2 is None: im2 = im1.copy()
    im1 = im1.resize(im2.size)
    a1 = array(im1); a2 = array(im2)
    if fill == "nd":
        b = ((a1[:,:,0].astype(float32) + a2[:,:,1].astype(float32))/2).astype(uint8)
        rgb = stack([a1[:,:,0], a2[:,:,1], b], axis=-1)
    elif fill == "avg":
        avg = ((a1.astype(float32) + a2.astype(float32))/2).astype(uint8)
        rgb = stack([a1[:,:,0], avg[:,:,1], a2[:,:,1]], axis=-1)
    elif fill == "repeat1":
        rgb = stack([a1[:,:,0], a1[:,:,0], a2[:,:,1]], axis=-1)
    elif fill == "repeat2":
        rgb = stack([a1[:,:,0], a2[:,:,1], a2[:,:,1]], axis=-1)
    else:
        raise ValueError(f"unknown fill: {fill}")
    from PIL import Image as _I
    return _I.fromarray(rgb)

def group_key(row: pd.Series, mode: str) -> str:
    s, sess = str(row["slaughter"]), str(row["session"])
    d = str(row.get("day","")).strip()
    return f"{s}__{sess}__{d}" if mode!="per_band" else str(row["record_key"])

# ===== 어댑터: 백본별 인터페이스 통일 =====
class BaseAdapter:
    def __init__(self, device="cpu"):
        self.device = device
        self.dim = None  # 출력 차원 (초기화 후 설정)
    def preprocess(self, pil_list: List[Image.Image]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    @torch.no_grad()
    def forward_features(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """B x D numpy 반환"""
        raise NotImplementedError
    def feature_dim(self) -> int:
        return int(self.dim)

# 1) HF ViT
class HFViTAdapter(BaseAdapter):
    def __init__(self, model_name: str, feature_type: str="cls", device="cpu"):
        super().__init__(device)
        from transformers import AutoImageProcessor, ViTModel
        self.proc = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device).eval()
        self.feature_type = feature_type
        self.dim = int(self.model.config.hidden_size) if feature_type!="cls_mean" else int(self.model.config.hidden_size*2)
    def preprocess(self, pil_list):
        x = self.proc(images=pil_list, return_tensors="pt")
        return {k: v.to(self.device) for k,v in x.items()}
    @torch.no_grad()
    def forward_features(self, batch):
        out = self.model(**batch).last_hidden_state  # (B, N, D)
        cls = out[:,0,:]                             # (B, D)
        if self.feature_type == "cls":
            feat = cls
        elif self.feature_type == "mean":
            feat = out.mean(dim=1)
        elif self.feature_type == "cls_mean":
            feat = torch.cat([cls, out.mean(dim=1)], dim=1)
        else:
            raise ValueError("feature_type must be cls|mean|cls_mean")
        return feat.detach().cpu().numpy()

# 2) timm CNN
class TIMMAdapter(BaseAdapter):
    def __init__(self, model_name: str, pretrained: bool=True, global_pool: str="avg", image_size: int=224, device="cpu"):
        super().__init__(device)
        import timm
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.model.to(self.device).eval()
        self.global_pool = global_pool
        # transforms
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        cfg = resolve_data_config({}, model=self.model)
        cfg["input_size"] = (3, image_size, image_size)
        self.transform = create_transform(**cfg)
        # 추출 차원 (피드포워드 후 풀링 전 채널 수)
        dummy = torch.zeros(1,3,image_size,image_size)
        with torch.no_grad():
            feats = self.model.forward_features(dummy)
            C = feats.shape[1] if feats.ndim==4 else feats.shape[-1]
        self.dim = C if global_pool!="cat" else C*2
    def preprocess(self, pil_list):
        ts = [self.transform(p) for p in pil_list]
        x = torch.stack(ts, dim=0).to(self.device)
        return {"x": x}
    @torch.no_grad()
    def forward_features(self, batch):
        x = batch["x"]
        feats = self.model.forward_features(x)   # (B,C,H,W) or (B,C)
        if feats.ndim == 4:
            if self.global_pool == "avg":
                feats = feats.mean(dim=[2,3])
            elif self.global_pool == "max":
                feats = feats.amax(dim=[2,3])
            elif self.global_pool == "cat":
                feats = torch.cat([feats.mean(dim=[2,3]), feats.amax(dim=[2,3])], dim=1)
            else:
                raise ValueError("global_pool must be avg|max|cat")
        return feats.detach().cpu().numpy()

# 3) MLflow Registered PyTorch
class MLflowTorchAdapter(BaseAdapter):
    def __init__(self, tracking_uri: str, model_uri: str, image_size: int=224,
                 normalize: Dict[str,List[float]]|None=None, device="cpu"):
        super().__init__(device)
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        # 모델은 mlflow.pytorch 또는 pyfunc일 수 있음 -> try 순서대로
        try:
            import mlflow.pytorch
            self.model = mlflow.pytorch.load_model(model_uri)
        except Exception:
            # pyfunc로 불러오되, predict에 텐서 넘길 수 있도록 래핑이 필요할 수 있음
            self.model = mlflow.pyfunc.load_model(model_uri)
        self.model.to(self.device) if hasattr(self.model, "to") else None
        self.model.eval() if hasattr(self.model, "eval") else None
        self.image_size = image_size
        nm = normalize or {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]}
        self.mean = torch.tensor(nm["mean"])[None,:,None,None]
        self.std  = torch.tensor(nm["std"] )[None,:,None,None]
        # dim 추정
        with torch.no_grad():
            dummy = torch.zeros(1,3,image_size,image_size)
            if self.device!="cpu": dummy = dummy.to(self.device)
            out = self._forward_internal(dummy)
            self.dim = int(out.shape[-1])
    def _forward_internal(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward"):
            y = self.model(x)
        else:
            # pyfunc 모델은 일반적으로 numpy 입력을 기대 → 변환
            y = torch.tensor(self.model.predict(x.detach().cpu().numpy()))
        if isinstance(y, (list, tuple)):  # 일부 모델이 (feat, logits) 같은 튜플 반환
            y = y[0]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        return y
    def preprocess(self, pil_list):
        import torchvision.transforms as T
        tfm = T.Compose([T.Resize(self.image_size), T.CenterCrop(self.image_size),
                         T.ToTensor()])
        xs = [tfm(p) for p in pil_list]
        x = torch.stack(xs, dim=0)
        x = (x - self.mean) / self.std
        return {"x": x.to(self.device)}
    @torch.no_grad()
    def forward_features(self, batch):
        y = self._forward_internal(batch["x"])
        return y.detach().cpu().numpy()

# ===== 추상화 팩토리 =====
def build_adapter(cfg: Dict[str,Any], device: str) -> BaseAdapter:
    src = cfg["embedding"]["model_source"]
    if src == "hf_vit":
        p = cfg["embedding"]["hf_vit"]
        return HFViTAdapter(p["model_name"], p.get("feature_type","cls"), device)
    elif src == "timm_cnn":
        p = cfg["embedding"]["timm_cnn"]
        return TIMMAdapter(p["model_name"], p.get("pretrained", True),
                           p.get("global_pool","avg"), p.get("image_size",224), device)
    elif src == "mlflow_torch":
        p = cfg["embedding"]["mlflow_torch"]
        return MLflowTorchAdapter(p["tracking_uri"], p["model_uri"],
                                  p.get("image_size",224), p.get("normalize", None), device)
    else:
        raise SystemExit(f"[ERR] unknown model_source: {src}")

# ===== 메타/증분/임베딩 루프 (기존 흐름 간결화 버전) =====
def read_processed_meta(processed_dir: Path) -> pd.DataFrame:
    m = processed_dir/"meta.csv"
    if not m.is_file(): raise SystemExit(f"[ERR] missing {m}")
    # CSV 파싱 오류 방지: 불일치하는 컬럼 수 문제 해결
    try:
        return pd.read_csv(m)
    except pd.errors.ParserError as e:
        print(f"[WARN] CSV parsing error, trying with error_bad_lines=False: {e}")
        return pd.read_csv(m, on_bad_lines='skip')

def load_export(export_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    f, m = export_dir/"features.npy", export_dir/"meta.csv"
    if f.is_file() and m.is_file():
        return np.load(f), pd.read_csv(m)
    return np.zeros((0,0),dtype=np.float32), pd.DataFrame()

def save_export(export_dir: Path, X: np.ndarray, meta: pd.DataFrame):
    export_dir.mkdir(parents=True, exist_ok=True)
    np.save(export_dir/"features.npy", X)
    meta.to_csv(export_dir/"meta.csv", index=False, encoding="utf-8-sig")

def run_from_config(cfg_path: Path):
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    processed_dir = Path(cfg["base"]["processed_dir"])
    export_dir    = Path(cfg["base"]["export_dir"])
    device        = device_autoselect(cfg["base"].get("device","cpu"))
    bs            = int(cfg["base"].get("batch_size",16))

    ms_cfg = cfg["multispectral"]
    ms_mode = ms_cfg.get("mode","per_band")
    wls     = [int(x) for x in ms_cfg.get("wavelengths",[430,540])]
    ms_fill = ms_cfg.get("fill","nd")

    meta_src = read_processed_meta(processed_dir)
    if "crop_path" not in meta_src.columns: raise SystemExit("[ERR] processed/meta.csv needs crop_path")
    X_old, meta_old = load_export(export_dir)
    have_old = (X_old.size>0) and (not meta_old.empty)

    # 그룹키
    meta_src["gkey"] = meta_src.apply(lambda r: group_key(r, ms_mode), axis=1)
    done_groups = set(meta_old["gkey"]) if have_old and "gkey" in meta_old.columns else set()
    done_keys   = set(meta_old["record_key"]) if have_old and "record_key" in meta_old.columns else set()

    # 어댑터 준비
    adapter = build_adapter(cfg, device)
    print(f"[Adapter] {cfg['embedding']['model_source']} -> dim={adapter.feature_dim()} on {device}")

    # 대상 모음
    work_units = []
    if ms_mode == "per_band":
        for _, r in meta_src.iterrows():
            if r["record_key"] in done_keys: continue
            work_units.append(("single",[r]))
    elif ms_mode == "fuse_rgb":
        for g, df_g in meta_src.groupby("gkey"):
            if g in done_groups: continue
            work_units.append(("fuse", [df_g]))
    else:  # concat_cls
        for g, df_g in meta_src.groupby("gkey"):
            if g in done_groups: continue
            work_units.append(("concat", [df_g]))

    X_new, M_new = [], []
    # 실행
    i = 0
    while i < len(work_units):
        kind, payload = work_units[i]
        if kind == "single":
            batch_rows = []
            pil_list = []
            # 배치로 묶기
            for j in range(i, min(i+bs, len(work_units))):
                k2, p2 = work_units[j]
                if k2!="single": break
                r = p2[0]
                if not os.path.isfile(r["crop_path"]): continue
                batch_rows.append(r)
                pil_list.append(pil_open_rgb(r["crop_path"]))
            if batch_rows:
                inputs = adapter.preprocess(pil_list)
                feats = adapter.forward_features(inputs)  # (B, D)
                X_new.append(feats)
                M_new.extend([r.to_dict() for r in batch_rows])
            i += len(batch_rows) if batch_rows else 1

        elif kind == "fuse":
            df_g = payload[0]
            by_wl = {int(r["wavelength_nm"]): r["crop_path"] for _, r in df_g.iterrows()}
            try:
                img = fuse_to_rgb(by_wl, wls, ms_fill)
            except Exception as e:
                print(f"[SKIP fuse] {df_g.iloc[0]['gkey']} -> {e}")
                i += 1; continue
            inputs = adapter.preprocess([img])
            feats = adapter.forward_features(inputs)  # (1, D)
            X_new.append(feats)
            base = df_g.iloc[0].to_dict()
            base["gkey"] = df_g.iloc[0]["gkey"]
            base["used_mode"] = f"fuse_rgb[{','.join(map(str,wls))}|fill={ms_fill}]"
            base["ms_wavelengths"] = ",".join(sorted({str(int(w)) for w in df_g["wavelength_nm"]}))
            base["group_count"] = int(len(df_g))
            M_new.append(base)
            i += 1

        else:  # concat_cls
            df_g = payload[0]
            pil_list, order = [], []
            for wl in wls:
                r_wl = df_g[df_g["wavelength_nm"]==wl]
                if r_wl.empty: continue
                cp = r_wl.iloc[0]["crop_path"]
                if not os.path.isfile(cp): continue
                pil_list.append(pil_open_rgb(cp)); order.append(wl)
            if not pil_list:
                i += 1; continue
            inputs = adapter.preprocess(pil_list)
            feats = adapter.forward_features(inputs)      # (k, D_adapt)
            concat = feats.reshape(1, -1)                 # (1, k*D)
            X_new.append(concat)
            base = df_g.iloc[0].to_dict()
            base["gkey"] = df_g.iloc[0]["gkey"]
            base["used_mode"] = f"concat_cls[{','.join(map(str,order))}]"
            base["ms_wavelengths"] = ",".join(map(str,order))
            base["group_count"] = int(len(df_g))
            M_new.append(base)
            i += 1

    if not X_new:
        print("[Info] nothing to embed.")
        return

    X_block = np.concatenate(X_new, axis=0).astype(np.float32)
    meta_block = pd.DataFrame(M_new)

    if have_old and X_old.size>0:
        if X_old.shape[0]==0:
            X_all, meta_all = X_block, meta_block
        else:
            if X_old.shape[1] != X_block.shape[1]:
                raise SystemExit(f"[ERR] feature dim mismatch: {X_old.shape[1]}(old) vs {X_block.shape[1]}(new)")
            X_all = np.vstack([X_old, X_block])
            meta_all = pd.concat([meta_old, meta_block], ignore_index=True)
    else:
        X_all, meta_all = X_block, meta_block

    export_dir.mkdir(parents=True, exist_ok=True)
    np.save(export_dir/"features.npy", X_all)
    meta_all.to_csv(export_dir/"meta.csv", index=False, encoding="utf-8-sig")
    # 모델/설정 메모 기록
    (export_dir/"embed_info.json").write_text(
        json.dumps({"adapter": cfg["embedding"]["model_source"], "dim": int(adapter.feature_dim()),
                    "multispectral": ms_cfg}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Saved] X: {X_all.shape}, rows: {len(meta_all)} @ {export_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_from_config(Path(args.config))

if __name__ == "__main__":
    main()
