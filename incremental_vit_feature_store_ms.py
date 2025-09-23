#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, sys, glob, json, sqlite3, hashlib, argparse
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import torch
from transformers import AutoImageProcessor, ViTModel

PTN_WL_COL = re.compile(r"^(TL|TR|BR|BL)\s*\(\s*(\d+)\s*nm\)\s*$", re.IGNORECASE)
PTN_DAY_IN_DIR = re.compile(r"_day(\d+)$", re.IGNORECASE)
PTN_IMG_STEM   = re.compile(r"^(?P<slaughter>\d+)_s(?P<session>\d+)_(?P<wavelength>\d+)nm_day(?P<day>\d+)$")

def as_int_str(x): 
    try: return str(int(float(x)))
    except: return str(x).strip()

def ensure_session_prefix(s):
    s=str(s).strip()
    return s if s.lower().startswith("s") else f"s{s}"

def parse_xy(s):
    if pd.isna(s): return None
    if not isinstance(s,str): s=str(s)
    m=re.match(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", s.strip())
    return (int(m.group(1)),int(m.group(2))) if m else None

def polygon_str(tl,tr,br,bl): 
    f=lambda t:f"{t[0]} {t[1]}"; return "; ".join([f(tl),f(tr),f(br),f(bl)])

def find_wavelengths(cols):
    w=set()
    for c in cols:
        m=PTN_WL_COL.match(str(c).strip())
        if m: w.add(int(m.group(2)))
    return sorted(w)

def polygon_to_bbox(pts):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return min(xs),min(ys),max(xs),max(ys)

def apply_bbox_crop(img,bbox,pad=6):
    h,w=img.shape[:2]; x1,y1,x2,y2=bbox
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w-1,x2+pad); y2=min(h-1,y2+pad)
    return img[y1:y2+1,x1:x2+1,:] if x2>x1 and y2>y1 else img

def apply_polygon_mask(img,pts):
    h,w=img.shape[:2]; m=Image.new("L",(w,h),0); ImageDraw.Draw(m).polygon(pts,1,1)
    m=np.array(m,dtype=np.uint8); o=img.copy(); o[m==0]=0; return o

def to_pil_rgb(img_bgr): return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
def read_gray(p): 
    img=cv2.imread(p,cv2.IMREAD_COLOR); 
    return None if img is None else cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def synth_third_channel(b1,b2,mode="nd"):
    b1f=b1.astype(np.float32); b2f=b2.astype(np.float32); eps=1e-6
    if mode=="nd":
        nd=(b2f-b1f)/(b2f+b1f+eps); nd=((nd+1)*127.5).clip(0,255); return nd.astype(np.uint8)
    if mode=="avg": return ((b1f+b2f)/2).clip(0,255).astype(np.uint8)
    if mode=="repeat1": return b1.copy()
    if mode=="repeat2": return b2.copy()
    return ((b1f+b2f)/2).clip(0,255).astype(np.uint8)

def excel_to_rows(xlsx, fb_day=None, label_cols=["등급","Marbling","Meat Color","Texture","Surface Moisture","Total"]):
    df=pd.read_excel(xlsx, sheet_name=0, engine="openpyxl")
    for col in ["이력번호","샘플번호","경과일수"]:
        if col not in df.columns: raise ValueError(f"[{xlsx}] '{col}' 없음")
    wls=find_wavelengths(df.columns); 
    if not wls: raise RuntimeError(f"[{xlsx}] TL/TR/BR/BL (###nm) 좌표 없음")
    rows=[]
    for _,r in df.iterrows():
        slaughter=as_int_str(r["이력번호"]); session=ensure_session_prefix(r["샘플번호"])
        day_raw=r["경과일수"]; day=as_int_str(day_raw) if (not pd.isna(day_raw) and str(day_raw).strip()!="") else (str(fb_day) if fb_day is not None else None)
        if day is None: raise ValueError(f"[{xlsx}] 경과일수 비어있고 폴더 추정 실패")
        labels={lc:r[lc] for lc in label_cols if lc in df.columns}
        for wl in wls:
            keys=[f"TL ({wl}nm)",f"TR ({wl}nm)",f"BR ({wl}nm)",f"BL ({wl}nm)"]
            keymap={}
            for need in keys:
                found=next((c for c in df.columns if str(c).strip().lower()==need.lower()),None)
                if found is None: keymap=None; break
                keymap[need]=found
            if keymap is None: continue
            tl=parse_xy(r[keymap[keys[0]]]); tr=parse_xy(r[keymap[keys[1]]]); br=parse_xy(r[keymap[keys[2]]]); bl=parse_xy(r[keymap[keys[3]]])
            if None in (tl,tr,br,bl): continue
            image=f"{slaughter}_{session}_{wl}nm_day{day}"
            rows.append({"sample_id":image,"base_id":f"{slaughter}_{session}_day{day}","image_name":image,
                        "polygon":polygon_str(tl,tr,br,bl),"wavelength_nm":wl,"slaughter":slaughter,"session":session,"day":day,**labels})
    return rows

def glob_images(root):
    exts=["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]; m={}
    for ext in exts:
        for p in glob.iglob(os.path.join(root,"**",ext), recursive=True):
            m[os.path.splitext(os.path.basename(p))[0]]=p
    return m

def file_sha1(path,block=1024*1024):
    h=hashlib.sha1()
    with open(path,"rb") as f:
        while True:
            b=f.read(block); 
            if not b: break
            h.update(b)
    return h.hexdigest()

DDL="""
CREATE TABLE IF NOT EXISTS features(
  sample_id TEXT NOT NULL, model_sig TEXT NOT NULL, dim INTEGER NOT NULL,
  embedding BLOB NOT NULL, meta_json TEXT NOT NULL, image_path TEXT NOT NULL,
  image_hash TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(sample_id,model_sig)
);
CREATE INDEX IF NOT EXISTS idx_features_model ON features(model_sig);
"""

def open_store(path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    conn=sqlite3.connect(path); conn.execute("PRAGMA journal_mode=WAL;")
    for stmt in DDL.strip().split(";"):
        s=stmt.strip(); 
        if s: conn.execute(s)
    return conn

def upsert_feature(conn, sid, sig, dim, feat, meta_json, ip, ih):
    emb=memoryview(feat.astype(np.float32).tobytes(order="C"))
    conn.execute("""INSERT INTO features(sample_id,model_sig,dim,embedding,meta_json,image_path,image_hash,updated_at)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(sample_id,model_sig) DO UPDATE SET
                    dim=excluded.dim, embedding=excluded.embedding, meta_json=excluded.meta_json,
                    image_path=excluded.image_path, image_hash=excluded.image_hash, updated_at=excluded.updated_at""",
                 (sid,sig,int(dim),emb,meta_json,ip,ih,datetime.utcnow().isoformat(timespec="seconds")))
    conn.commit()

def fetch_hashes(conn, sig):
    cur=conn.execute("SELECT sample_id,image_hash FROM features WHERE model_sig=?", (sig,))
    return {sid:ih for sid,ih in cur.fetchall()}

def export_to_files(conn, sig, out_dir):
    os.makedirs(out_dir,exist_ok=True)
    cur=conn.execute("SELECT sample_id,dim,embedding,meta_json,image_path FROM features WHERE model_sig=? ORDER BY sample_id",(sig,))
    feats=[]; metas=[]
    for sid,dim,emb,meta_json,ip in cur:
        arr=np.frombuffer(emb,dtype=np.float32); feats.append(arr.reshape(-1))
        md=json.loads(meta_json); md["image_path"]=ip; metas.append(md)
    if not feats: raise RuntimeError("해당 model_sig 임베딩 없음")
    X=np.vstack(feats); meta=pd.DataFrame(metas)
    np.save(os.path.join(out_dir,"features.npy"),X)
    meta.to_csv(os.path.join(out_dir,"meta.csv"),index=False,encoding="utf-8-sig")
    print(f"[Exported] {X.shape} -> {os.path.join(out_dir,'features.npy')}"); print(f"[Exported] rows={len(meta)}")

def build_model(name,device):
    proc=AutoImageProcessor.from_pretrained(name); model=ViTModel.from_pretrained(name).to(device).eval(); return proc,model

def run_inference(proc,model,images,feature_type,device):
    with torch.no_grad():
        inputs=proc(images=images,return_tensors="pt"); inputs={k:v.to(device) for k,v in inputs.items()}
        out=model(**inputs); tok=out.last_hidden_state; cls=tok[:,0,:]; mean=tok[:,1:,:].mean(1)
        feat=cls if feature_type=="cls" else (mean if feature_type=="mean" else torch.cat([cls,mean],1))
        return feat.detach().cpu().numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_root",required=True); ap.add_argument("--store_path",required=True)
    ap.add_argument("--model_name",default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--feature_type",choices=["cls","mean","cls_mean"],default="cls_mean")
    ap.add_argument("--preprocess_mode",choices=["auto","crop","mask"],default="auto")
    ap.add_argument("--ms_mode",choices=["per_band","fuse_rgb","concat_cls"],default="per_band")
    ap.add_argument("--ms_wavelengths",default="430,540"); ap.add_argument("--ms_fill",choices=["nd","repeat1","repeat2","avg"],default="nd")
    ap.add_argument("--batch_size",type=int,default=16); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--export",action="store_true"); ap.add_argument("--export_dir",default="./vit_out")
    ap.add_argument("--save_debug_vis",type=int,default=0)
    args=ap.parse_args()

    ms_wls=sorted([int(s) for s in str(args.ms_wavelengths).split(",") if s.strip()])
    sig=json.dumps({"model":args.model_name,"feature_type":args.feature_type,"preprocess":args.preprocess_mode,
                    "ms_mode":args.ms_mode,"ms_wls":ms_wls,"ms_fill":args.ms_fill,"v":"ms_v1"},sort_keys=True)
    print("[model_sig]",sig)

    result_root=os.path.join(args.dataset_root,"result")
    if not os.path.isdir(result_root): raise FileNotFoundError(f"result 폴더 없음: {result_root}")
    xlsx_list=[p for p in glob.iglob(os.path.join(result_root,"**","*.xlsx"),recursive=True)]
    rows=[]
    for x in tqdm(xlsx_list,desc="Excel->rows"):
        base=os.path.basename(os.path.dirname(x)); m=PTN_DAY_IN_DIR.search(base); fb=int(m.group(1)) if m else None
        rows.extend(excel_to_rows(x, fb_day=fb))
    seg=pd.DataFrame(rows); 
    if seg.empty: print("[Warn] no rows"); sys.exit(0)
    img_map=glob_images(args.dataset_root); seg["image_path"]=seg["image_name"].map(lambda s: img_map.get(s,""))

    conn=open_store(args.store_path)

    if args.ms_mode=="per_band":
        existing=fetch_hashes(conn,sig); todo=[]
        for r in seg.itertuples(index=False):
            rd=r._asdict(); sid=rd["sample_id"]; ip=rd.get("image_path","")
            if not ip or not os.path.isfile(ip): continue
            ih=file_sha1(ip)
            if existing.get(sid)==ih: continue
            todo.append((sid,ip,ih,rd))
        print(f"[Plan] per_band={len(todo)}")
        if todo:
            proc,model=build_model(args.model_name,args.device); batch_imgs=[]; batch_meta=[]; dbg=args.save_debug_vis
            for sid,ip,ih,rd in tqdm(todo,desc="Embedding(per_band)"):
                img=cv2.imread(ip,cv2.IMREAD_COLOR); 
                if img is None: continue
                pts=[]
                if isinstance(rd.get("polygon"),str) and rd["polygon"].strip():
                    try:
                        for pair in rd["polygon"].split(";"):
                            x,y=pair.strip().split(); pts.append((int(float(x)),int(float(y))))
                    except: pts=[]
                used="none"; modes=["crop","mask"] if args.preprocess_mode=="auto" else [args.preprocess_mode]
                for m in modes:
                    if m=="crop" and pts: img=apply_bbox_crop(img,polygon_to_bbox(pts)); used="crop"; break
                    if m=="mask" and pts: img=apply_polygon_mask(img,pts); used="mask"; break
                batch_imgs.append(to_pil_rgb(img)); batch_meta.append((sid,ip,ih,rd,used))
                if len(batch_imgs)>=args.batch_size:
                    feats=run_inference(proc,model,batch_imgs,args.feature_type,args.device)
                    for ft,mt in zip(feats,batch_meta):
                        sid,ip,ih,rd,used=mt; md=dict(rd); md["used_mode"]=used; md["feature_type"]=args.feature_type
                        upsert_feature(conn,sid,sig,ft.shape[-1],ft,json.dumps(md,ensure_ascii=False),ip,ih)
                    batch_imgs.clear(); batch_meta.clear()
            if batch_imgs:
                feats=run_inference(proc,model,batch_imgs,args.feature_type,args.device)
                for ft,mt in zip(feats,batch_meta):
                    sid,ip,ih,rd,used=mt; md=dict(rd); md["used_mode"]=used; md["feature_type"]=args.feature_type
                    upsert_feature(conn,sid,sig,ft.shape[-1],ft,json.dumps(md,ensure_ascii=False),ip,ih)

    elif args.ms_mode=="fuse_rgb":
        existing=fetch_hashes(conn,sig); proc,model=build_model(args.model_name,args.device)
        groups=seg.groupby("base_id"); todo=[]
        for base_id,gdf in groups:
            bywl={int(w):row for w,row in zip(gdf["wavelength_nm"],gdf.to_dict("records"))}
            if not all(w in bywl for w in ms_wls): continue
            sid=f"{base_id}__fuse[{','.join(map(str,ms_wls))}]"
            imgps=[]; hashes=[]
            for w in ms_wls:
                ip=img_map.get(bywl[w]["image_name"],""); 
                if not ip or not os.path.isfile(ip): imgps=[]; break
                imgps.append(ip); hashes.append(file_sha1(ip))
            if not imgps: continue
            ih_all=hashlib.sha1((":".join(hashes)).encode()).hexdigest()
            if existing.get(sid)==ih_all: continue
            todo.append((sid,bywl,imgps,ih_all))
        print(f"[Plan] fuse_rgb={len(todo)}")
        for sid,bywl,imgps,ih in tqdm(todo,desc="Embedding(fuse_rgb)"):
            bands=[]; bboxes=[]
            for w,ip in zip(ms_wls,imgps):
                gray=read_gray(ip); 
                if gray is None: bands=None; break
                rd=bywl[w]; pts=[]
                if isinstance(rd.get("polygon"),str) and rd["polygon"].strip():
                    try:
                        for pair in rd["polygon"].split(";"):
                            x,y=pair.strip().split(); pts.append((int(float(x)),int(float(y))))
                    except: pts=[]
                if pts: bboxes.append(polygon_to_bbox(pts))
                bands.append((w,gray,pts))
            if bands is None: continue
            common=None
            if bboxes:
                xs,ys,xe,ye=zip(*bboxes); common=(min(xs),min(ys),max(xe),max(ye))
            proc_b=[]
            for _,g,pts in bands:
                img=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
                if common is not None: img=apply_bbox_crop(img,common)
                if pts:
                    pts_adj=[(p[0]-(common[0] if common else 0), p[1]-(common[1] if common else 0)) for p in pts]
                    img=apply_polygon_mask(img,pts_adj)
                proc_b.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
            h=min(b.shape[0] for b in proc_b); w=min(b.shape[1] for b in proc_b); proc_b=[b[:h,:w] for b in proc_b]
            b1,b2=proc_b[0],proc_b[1]; b3=synth_third_channel(b1,b2,mode=args.ms_fill)
            fuse=np.stack([b1,b2,b3],2); pil=Image.fromarray(fuse,mode="RGB")
            rep=dict(bywl[ms_wls[0]]); rep["ms_used_wavelengths"]=ms_wls; rep["ms_mode"]="fuse_rgb"; rep["image_path_list"]=imgps
            ft=run_inference(proc,model,[pil],args.feature_type,args.device)[0]
            upsert_feature(conn,sid,sig,ft.shape[-1],ft,json.dumps(rep,ensure_ascii=False),"|".join(imgps),ih)

    elif args.ms_mode=="concat_cls":
        existing=fetch_hashes(conn,sig); proc,model=build_model(args.model_name,args.device)
        groups=seg.groupby("base_id"); todo=[]
        for base_id,gdf in groups:
            bywl={int(w):row for w,row in zip(gdf["wavelength_nm"],gdf.to_dict("records"))}
            if not all(w in bywl for w in ms_wls): continue
            sid=f"{base_id}__concat[{','.join(map(str,ms_wls))}]"
            imgps=[]; hashes=[]
            for w in ms_wls:
                ip=img_map.get(bywl[w]["image_name"],""); 
                if not ip or not os.path.isfile(ip): imgps=[]; break
                imgps.append(ip); hashes.append(file_sha1(ip))
            if not imgps: continue
            ih_all=hashlib.sha1((":".join(hashes)).encode()).hexdigest()
            if existing.get(sid)==ih_all: continue
            feats=[]
            rep=dict(bywl[ms_wls[0]]); rep["ms_used_wavelengths"]=ms_wls; rep["ms_mode"]="concat_cls"; rep["image_path_list"]=imgps
            for w in ms_wls:
                ip=img_map.get(bywl[w]["image_name"],""); img=cv2.imread(ip,cv2.IMREAD_COLOR)
                if img is None: feats=None; break
                rd=bywl[w]; pts=[]
                if isinstance(rd.get("polygon"),str) and rd["polygon"].strip():
                    try:
                        for pair in rd["polygon"].split(";"):
                            x,y=pair.strip().split(); pts.append((int(float(x)),int(float(y))))
                    except: pts=[]
                used="none"; modes=["crop","mask"] if args.preprocess_mode=="auto" else [args.preprocess_mode]
                for m in modes:
                    if m=="crop" and pts: img=apply_bbox_crop(img,polygon_to_bbox(pts)); used="crop"; break
                    if m=="mask" and pts: img=apply_polygon_mask(img,pts); used="mask"; break
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                pil=Image.fromarray(rgb); ft=run_inference(proc,model,[pil],args.feature_type,args.device)[0]; feats.append(ft)
            if feats is None: continue
            feat=np.concatenate(feats,-1)
            upsert_feature(conn,sid,sig,feat.shape[-1],feat,json.dumps(rep,ensure_ascii=False),"|".join(imgps),ih_all)
    else:
        raise ValueError("unknown ms_mode")
    if args.export: export_to_files(conn,sig,args.export_dir)
    conn.close(); print("[Done] incremental(ms) finished.")

if __name__=="__main__": main()
