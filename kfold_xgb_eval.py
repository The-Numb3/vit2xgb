#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF=True
except:
    HAS_SGKF=False
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from xgboost import XGBRegressor, XGBClassifier

# >>> NEW: PCA utils
try:
    from pca_utils import apply_pca_train_val
    HAS_PCA_UTILS = True
except Exception as e:
    HAS_PCA_UTILS = False
    _pca_import_err = e

def ensure(v): 
    f=os.path.join(v,"features.npy"); m=os.path.join(v,"meta.csv")
    if not (os.path.isfile(f) and os.path.isfile(m)): sys.exit(f"[ERR] not found: {v}")
    return f,m

def is_num(s):
    try: pd.to_numeric(s); return True
    except: return False

def to_ordinal(y_raw):
    ys=y_raw.astype(str).str.strip(); classes=sorted(ys.unique())
    mp={c:i for i,c in enumerate(classes,1)}; y=ys.map(mp).astype(np.int32).values
    return y,classes,mp

def from_ordinal(y_cont,classes):
    K=len(classes); r=np.clip(np.rint(y_cont),1,K).astype(int); inv={i+1:c for i,c in enumerate(classes)}
    return np.array([inv[i] for i in r]), r

def parse_groups(df, s):
    if not s: return None
    cols=[c.strip() for c in s.split(",") if c.strip()]
    for c in cols:
        if c not in df.columns: sys.exit(f"[ERR] group_by col missing: {c}")
    return df[cols[0]].astype(str).values if len(cols)==1 else df[cols].astype(str).agg("__".join,axis=1).values

def agg_by_keys(X, meta, keys, target, task):
    if not keys: return X, meta[target].values, meta.copy()
    df=meta.copy(); df["_g"]=df[keys].astype(str).agg("__".join,axis=1)
    groups=df.groupby("_g").indices; gids=list(groups.keys()); D=X.shape[1]
    Xg=np.zeros((len(gids),D),dtype=X.dtype); yg=[]; rows=[]
    for gi,gid in enumerate(gids):
        idx=np.fromiter(groups[gid],dtype=int); Xg[gi]=X[idx].mean(0)
        if task=="reg": yg.append(pd.to_numeric(df.loc[idx,target],errors="coerce").mean())
        else: yg.append(df.loc[idx,target].astype(str).mode().iat[0])
        rep=df.loc[idx[0]].to_dict(); rep["group_key"]=gid; rep["group_count"]=len(idx); rows.append(rep)
    return Xg, np.array(yg), pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vit_out",required=True); ap.add_argument("--target",required=True)
    ap.add_argument("--task",choices=["reg","clf"],default="reg")
    ap.add_argument("--reg_mode",choices=["auto","continuous","ordinal"],default="auto")
    ap.add_argument("--n_splits",type=int,default=5); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--group_by",default=""); ap.add_argument("--aggregate_by",default="")
    ap.add_argument("--xgb_params",default="")
    # >>> NEW: PCA options
    ap.add_argument("--use_pca",action="store_true",help="Enable PCA(dim=pca_dim) inside each fold")
    ap.add_argument("--pca_dim",type=int,default=64)
    ap.add_argument("--pca_whiten",action="store_true")
    ap.add_argument("--pca_random_state",type=int,default=42)
    args=ap.parse_args()

    # sanity for PCA import
    if args.use_pca and not HAS_PCA_UTILS:
        sys.exit(f"[ERR] --use_pca set but pca_utils.py import failed: {_pca_import_err}")

    fpath,mpath=ensure(args.vit_out); X=np.load(fpath); meta=pd.read_csv(mpath)
    if args.target not in meta.columns: sys.exit(f"[ERR] target '{args.target}' missing")
    ok=meta["image_path"].astype(str).str.len()>0; ok &= ~meta[args.target].isna()
    X=X[ok.values]; meta=meta.loc[ok].reset_index(drop=True)
    groups=parse_groups(meta,args.group_by)
    agg_keys=[c.strip() for c in args.aggregate_by.split(",") if c.strip()]
    for c in agg_keys:
        if c not in meta.columns: sys.exit(f"[ERR] aggregate_by col missing: {c}")
    X,y,meta_after=agg_by_keys(X,meta,agg_keys,args.target,"reg" if args.task=="reg" else "clf")
    if args.group_by:
        gcols=[c.strip() for c in args.group_by.split(",") if c.strip()]
        groups=meta_after[gcols[0]].astype(str).values if len(gcols)==1 else meta_after[gcols].astype(str).agg("__".join,axis=1).values

    out=Path(args.vit_out)
    params_reg=dict(n_estimators=1200,max_depth=8,learning_rate=0.03,subsample=0.9,colsample_bytree=0.9,tree_method="hist",random_state=args.seed)
    params_clf=dict(n_estimators=800,max_depth=6,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,tree_method="hist",random_state=args.seed)
    if args.xgb_params:
        import json; ov=json.loads(args.xgb_params)
        (params_clf if args.task=="clf" else params_reg).update(ov)

    pca_tag = f" [+PCA-d{args.pca_dim}]" if args.use_pca else ""

    if args.task=="clf":
        if not is_num(pd.Series(y)): 
            y,classes=pd.factorize(pd.Series(y).astype(str).str.strip())
        else:
            y=pd.Series(y).astype(int).values; classes=np.unique(y)
        if groups is not None and HAS_SGKF:
            splits=StratifiedGroupKFold(args.n_splits,shuffle=True,random_state=args.seed).split(X,y,groups)
        elif groups is not None:
            splits=GroupKFold(args.n_splits).split(X,y,groups)
        else:
            splits=StratifiedKFold(args.n_splits,shuffle=True,random_state=args.seed).split(X,y)
        accs,f1s,rows=[],[],[]
        for k,(tr,te) in enumerate(splits,1):
            Xtr, Xte = X[tr], X[te]
            if args.use_pca:
                Xtr, Xte = apply_pca_train_val(Xtr, Xte, n_components=args.pca_dim, whiten=args.pca_whiten, random_state=args.pca_random_state)
            m=XGBClassifier(**params_clf); m.fit(Xtr,y[tr]); p=m.predict(Xte)
            a=accuracy_score(y[te],p); f=f1_score(y[te],p,average="macro")
            accs.append(a); f1s.append(f); rows.append({"fold":k,"ACC":a,"F1_macro":f})
        pd.DataFrame(rows).to_csv(out/f"kfold_{args.target}_clf_metrics.csv",index=False,encoding="utf-8-sig")
        s=(f"KFOLD ({args.n_splits}) - target={args.target} [clf{pca_tag}]\n"
           f"ACC mean±std : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n"
           f"F1  mean±std : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")
        print(s); (out/f"kfold_{args.target}_clf_summary.txt").write_text(s,encoding="utf-8"); return

    # --- regression ---
    reg_mode=args.reg_mode
    if reg_mode=="auto":
        if not is_num(pd.Series(y)): reg_mode="ordinal"
        else:
            ynum=pd.to_numeric(pd.Series(y),errors="coerce"); uq=np.unique(ynum[~np.isnan(ynum)])
            reg_mode="ordinal" if (np.allclose(uq,uq.astype(int)) and len(uq)<=10) else "continuous"

    if reg_mode=="continuous":
        y=pd.to_numeric(pd.Series(y),errors="coerce").values
        splits=GroupKFold(args.n_splits).split(X,y,groups) if groups is not None else KFold(args.n_splits,shuffle=True,random_state=args.seed).split(X,y)
        r2s,maes,rows=[],[],[]
        for k,(tr,te) in enumerate(splits,1):
            Xtr, Xte = X[tr], X[te]
            if args.use_pca:
                Xtr, Xte = apply_pca_train_val(Xtr, Xte, n_components=args.pca_dim, whiten=args.pca_whiten, random_state=args.pca_random_state)
            m=XGBRegressor(**params_reg); m.fit(Xtr,y[tr]); p=m.predict(Xte)
            r2=r2_score(y[te],p); mae=mean_absolute_error(y[te],p)
            r2s.append(r2); maes.append(mae); rows.append({"fold":k,"R2":r2,"MAE":mae})
        pd.DataFrame(rows).to_csv(out/f"kfold_{args.target}_reg_cont_metrics.csv",index=False,encoding="utf-8-sig")
        s=(f"KFOLD ({args.n_splits}) - target={args.target} [reg-continuous{pca_tag}]\n"
           f"R2  mean±std : {np.mean(r2s):.4f} ± {np.std(r2s):.4f}\n"
           f"MAE mean±std: {np.mean(maes):.4f} ± {np.std(maes):.4f}\n")
        print(s); (out/f"kfold_{args.target}_reg_cont_summary.txt").write_text(s,encoding="utf-8"); return

    # ordinal
    y_ord,classes,_=to_ordinal(pd.Series(y))
    if groups is not None and HAS_SGKF:
        splits=StratifiedGroupKFold(args.n_splits,shuffle=True,random_state=args.seed).split(X,y_ord,groups)
    elif groups is not None:
        splits=GroupKFold(args.n_splits).split(X,y_ord,groups)
    else:
        splits=StratifiedKFold(args.n_splits,shuffle=True,random_state=args.seed).split(X,y_ord)
    def qwk(t,p,K):
        O=np.zeros((K,K)); 
        for a,b in zip(t,p): O[a-1,b-1]+=1
        W=np.zeros_like(O); 
        for i in range(K):
            for j in range(K): W[i,j]=((i-j)**2)/((K-1)**2) if K>1 else 0
        act=O.sum(1); pred=O.sum(0); E=np.outer(act,pred)/max(O.sum(),1)
        num=(W*O).sum(); den=(W*E).sum() if (W*E).sum()!=0 else 1.0
        return 1-num/den
    maes,r2s,accs,f1s,qwks=[],[],[],[],[]; rows=[]; K=len(classes)
    for k,(tr,te) in enumerate(splits,1):
        Xtr, Xte = X[tr], X[te]
        if args.use_pca:
            Xtr, Xte = apply_pca_train_val(Xtr, Xte, n_components=args.pca_dim, whiten=args.pca_whiten, random_state=args.pca_random_state)
        m=XGBRegressor(**params_reg); m.fit(Xtr,y_ord[tr]); pc=m.predict(Xte)
        _,p1=from_ordinal(pc,classes); _,t1=from_ordinal(y_ord[te],classes)
        mae=mean_absolute_error(t1,p1); r2=r2_score(y_ord[te],pc)
        acc=accuracy_score(t1,p1); f1=f1_score(t1,p1,average="macro"); kappa=qwk(t1,p1,K)
        maes.append(mae); r2s.append(r2); accs.append(acc); f1s.append(f1); qwks.append(kappa)
        rows.append({"fold":k,"MAE_steps":mae,"R2_ord":r2,"ACC_rounded":acc,"F1_macro_rounded":f1,"QWK_rounded":kappa})
    pd.DataFrame(rows).to_csv(out/f"kfold_{args.target}_reg_ordinal_metrics.csv",index=False,encoding="utf-8-sig")
    s=(f"KFOLD ({args.n_splits}) - target={args.target} [reg-ordinal{pca_tag}]\n"
       f"MAE(steps) mean±std : {np.mean(maes):.4f} ± {np.std(maes):.4f}\n"
       f"R2(ordinal) mean±std: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}\n"
       f"ACC(rounded) mean±std: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n"
       f"F1 (rounded) mean±std: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n"
       f"QWK(rounded) mean±std: {np.mean(qwks):.4f} ± {np.std(qwks):.4f}\n")
    print(s); (out/f"kfold_{args.target}_reg_ordinal_summary.txt").write_text(s,encoding="utf-8")

if __name__=="__main__": main()
