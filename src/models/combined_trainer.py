"""
combined_trainer.py  �  Models on the full merged ESI dataset.
  A. Regression      � reconstruct ESI from indicators
  B. Classification  � predict Low/Medium/High regime
  C. Anomaly         � detect crisis years
  D. Regime Analysis � rolling stats, transitions, structural breaks + all plots
"""
import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, IsolationForest)
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from src.config import (PLOTS_DIR, TABLES_DIR, MODELS_DIR, PALETTE,
                         RANDOM_STATE, CV_FOLDS, ANOMALY_CONTAMINATION)
warnings.filterwarnings("ignore")

_CE = PALETTE["esi"]
_RC = {"Low Stress": PALETTE["low"], "Medium Stress": PALETTE["medium"], "High Stress": PALETTE["high"]}

def _fmt(ax):
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.tick_params(axis="x", rotation=45)

def _sp(name):
    plt.savefig(PLOTS_DIR / name, dpi=150, bbox_inches="tight")
    print(f"    Saved -> {PLOTS_DIR / name}")
    plt.close()

def _save(model, name):
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

def train_combined_models(merged_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}\n  Combined ESI Models\n{'='*60}")
    df = merged_df.copy().sort_values("Year").reset_index(drop=True)
    _regression(df)
    df = _classification(df)
    df = _anomaly(df)
    df = _regime_analysis(df)
    out = TABLES_DIR / "combined_esi_full.csv"
    df.to_csv(out, index=False)
    print(f"\n  Full dataset saved -> {out}")
    return df

def _regression(df):
    print(f"\n  [A] Regression")
    ic = [c for c in df.columns if c not in ("Year","esi_score")]
    X,y,years = df[ic].values, df["esi_score"].values, df["Year"].values
    sp = int(len(df)*0.8)
    Xtr,Xte,ytr,yte,yrs = X[:sp],X[sp:],y[:sp],y[sp:],years[sp:]
    models = {"LinearRegression": LinearRegression(), "Ridge": Ridge(1.0),
               "RandomForest": RandomForestRegressor(n_estimators=200,random_state=RANDOM_STATE),
               "GradientBoosting": GradientBoostingRegressor(n_estimators=200,random_state=RANDOM_STATE)}
    records, best_rmse, best_name, best_preds, best_m = [], np.inf, None, None, None
    for name, m in models.items():
        m.fit(Xtr,ytr); p = m.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte,p)); r2 = r2_score(yte,p)
        cv = -cross_val_score(m,X,y,cv=min(CV_FOLDS,len(df)//2),
                              scoring="neg_root_mean_squared_error").mean()
        records.append({"Model":name,"RMSE":round(rmse,4),"R2":round(r2,4),"CV_RMSE":round(cv,4)})
        print(f"    {name:<22} RMSE={rmse:.4f}  R2={r2:.4f}  CV={cv:.4f}")
        _save(m, f"comb_reg_{name.lower()}")
        if rmse < best_rmse:
            best_rmse,best_name,best_preds,best_m = rmse,name,p,m
    pd.DataFrame(records).sort_values("RMSE").to_csv(TABLES_DIR/"combined_reg_metrics.csv",index=False)
    print(f"    Best: {best_name} (RMSE={best_rmse:.4f})")
    fig,ax = plt.subplots(figsize=(11,5))
    ax.plot(yrs,yte,"o-",label="Actual",color=_CE,lw=2)
    ax.plot(yrs,best_preds,"x--",label=f"Pred ({best_name})",color=PALETTE["highlight"],lw=2)
    ax.set_title(f"ESI Actual vs Predicted",fontweight="bold"); ax.set_xlabel("Year"); ax.set_ylabel("ESI")
    ax.legend(); ax.grid(True,linestyle="--",alpha=0.4); _fmt(ax); plt.tight_layout(); _sp("comb_reg_avp.png")
    fig,ax = plt.subplots(figsize=(11,4))
    ax.bar(yrs,yte-best_preds,color="#457B9D",alpha=0.75); ax.axhline(0,color="black",lw=0.8)
    ax.set_title("ESI Residuals",fontweight="bold"); ax.set_xlabel("Year"); ax.grid(axis="y",linestyle="--",alpha=0.4)
    _fmt(ax); plt.tight_layout(); _sp("comb_reg_residuals.png")
    if hasattr(best_m,"feature_importances_"):
        idx=np.argsort(best_m.feature_importances_)
        fig,ax = plt.subplots(figsize=(9,5))
        ax.barh([ic[i] for i in idx],best_m.feature_importances_[idx],color="#2A9D8F")
        ax.set_title(f"Feature Importance � {best_name}",fontweight="bold"); ax.grid(axis="x",linestyle="--",alpha=0.4)
        plt.tight_layout(); _sp("comb_reg_feat_importance.png")

def _classification(df):
    print(f"\n  [B] Classification")
    df = df.copy()
    df["stress_level"] = pd.qcut(df["esi_score"],q=3,labels=["Low","Medium","High"])
    ic = [c for c in df.columns if c not in ("Year","esi_score","stress_level")]
    X,y = df[ic].values, df["stress_level"].astype(str).values
    sp = int(len(df)*0.8); Xtr,Xte,ytr,yte = X[:sp],X[sp:],y[:sp],y[sp:]
    models = {"LogisticRegression": LogisticRegression(max_iter=1000,random_state=RANDOM_STATE),
               "RandomForest": RandomForestClassifier(n_estimators=200,random_state=RANDOM_STATE)}
    records,best_cv,best_name = [],0,None
    for name, m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        acc=accuracy_score(yte,p)
        cv=cross_val_score(m,X,y,cv=min(CV_FOLDS,len(df)//2),scoring="accuracy").mean()
        records.append({"Model":name,"Test_Acc":round(acc,4),"CV_Acc":round(cv,4)})
        print(f"    {name:<22} Acc={acc:.4f}  CV={cv:.4f}")
        _save(m,f"comb_clf_{name.lower()}")
        cm=confusion_matrix(yte,p,labels=m.classes_)
        fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm,cmap="Blues"); plt.colorbar(im,ax=ax)
        ax.set_xticks(range(len(m.classes_))); ax.set_xticklabels(m.classes_)
        ax.set_yticks(range(len(m.classes_))); ax.set_yticklabels(m.classes_)
        [[ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                  color="white" if cm[i,j]>cm.max()/2 else "black")
          for j in range(cm.shape[1])] for i in range(cm.shape[0])]
        ax.set_title(f"CM � {name}",fontweight="bold"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout(); _sp(f"comb_clf_cm_{name.lower()}.png")
        if cv>best_cv: best_cv,best_name=cv,name
    pd.DataFrame(records).to_csv(TABLES_DIR/"combined_clf_metrics.csv",index=False)
    print(f"    Best: {best_name} (CV={best_cv:.4f})")
    order=["Low","Medium","High"]
    counts=[df["stress_level"].astype(str).value_counts().get(l,0) for l in order]
    fig,ax=plt.subplots(figsize=(6,5))
    bars=ax.bar(order,counts,color=[PALETTE["low"],PALETTE["medium"],PALETTE["high"]],alpha=0.85,width=0.5)
    ax.bar_label(bars); ax.set_title("Stress Distribution",fontweight="bold")
    ax.grid(axis="y",linestyle="--",alpha=0.4); plt.tight_layout(); _sp("comb_clf_stress_dist.png")
    return df

def _anomaly(df):
    print(f"\n  [C] Anomaly Detection")
    df = df.copy()
    mu,sigma = df["esi_score"].mean(), df["esi_score"].std()
    df["z_score"]   = (df["esi_score"]-mu)/sigma
    df["anomaly_z"] = df["z_score"]>2.0
    iso = IsolationForest(contamination=ANOMALY_CONTAMINATION,random_state=RANDOM_STATE)
    df["anomaly_iso"] = iso.fit_predict(df[["esi_score"]])==-1
    _save(iso,"comb_anomaly_iforest")
    df["crisis_year"] = df["anomaly_z"] | df["anomaly_iso"]
    crisis = df[df["crisis_year"]]["Year"].tolist()
    print(f"    Crisis years ({len(crisis)}): {crisis}")
    df[["Year","esi_score","z_score","crisis_year"]].to_csv(TABLES_DIR/"combined_anomaly.csv",index=False)
    fig,ax=plt.subplots(figsize=(13,5)); ax.plot(df["Year"],df["esi_score"],color=_CE,lw=2,label="ESI")
    c=df[df["crisis_year"]]
    ax.scatter(c["Year"],c["esi_score"],color="#D62828",s=100,zorder=5,marker="X",label="Crisis")
    [ax.text(row["Year"],row["esi_score"]+0.01,str(int(row["Year"])),fontsize=8,
             color="#D62828",fontweight="bold",ha="center") for _,row in c.iterrows()]
    ax.set_title("India ESI � Crisis Years",fontweight="bold"); ax.set_xlabel("Year"); ax.set_ylabel("ESI")
    ax.legend(); ax.grid(True,linestyle="--",alpha=0.4); _fmt(ax); plt.tight_layout(); _sp("comb_anomaly_crises.png")
    return df

def _regime_analysis(df):
    print(f"\n  [D] Regime Analysis")
    df = df.copy().sort_values("Year").reset_index(drop=True)
    df["esi_3yr_avg"] = df["esi_score"].shift(1).rolling(3,min_periods=1).mean()
    df["esi_5yr_avg"] = df["esi_score"].shift(1).rolling(5,min_periods=1).mean()
    try:
        df["analysis_regime"] = pd.qcut(df["esi_score"],q=3,
            labels=["Low Stress","Medium Stress","High Stress"]).astype(str)
    except ValueError:
        df["analysis_regime"] = pd.cut(df["esi_score"],bins=3,
            labels=["Low Stress","Medium Stress","High Stress"]).astype(str)
    rg,yr = df["analysis_regime"].tolist(), df["Year"].tolist()
    trans = [(int(yr[i]),rg[i-1],rg[i]) for i in range(1,len(rg)) if rg[i]!=rg[i-1]]
    print(f"    Transitions ({len(trans)}):")
    [print(f"      {y}: {f} -> {t}") for y,f,t in trans]
    diffs=df["esi_score"].diff().abs(); thresh=diffs.mean()+1.5*diffs.std()
    cps=[(int(df.loc[i,"Year"]),"Spike" if df.loc[i,"esi_score"]>df.loc[i-1,"esi_score"] else "Drop")
         for i in diffs.index[1:] if diffs[i]>thresh]
    print(f"    Structural breaks: {[c[0] for c in cps]}")
    df[["Year","esi_score","analysis_regime","esi_3yr_avg","esi_5yr_avg","crisis_year"]].to_csv(
        TABLES_DIR/"combined_regime.csv",index=False)
    # Rolling avg plot
    fig,ax=plt.subplots(figsize=(13,6))
    ax.plot(df["Year"],df["esi_score"],lw=1.5,color=_CE,alpha=0.45,marker="o",
            markersize=3.5,markerfacecolor="white",label="ESI (annual)")
    ax.plot(df["Year"],df["esi_3yr_avg"],lw=2.2,color="#E63946",label="3-Year Avg")
    ax.plot(df["Year"],df["esi_5yr_avg"],lw=2.5,color="#F4A261",linestyle="--",label="5-Year Avg")
    ax.set_title("India ESI � Rolling Averages",fontweight="bold"); ax.set_xlabel("Year"); ax.set_ylabel("ESI")
    ax.legend(fontsize=9); ax.grid(axis="y",linestyle="--",alpha=0.35); _fmt(ax); plt.tight_layout(); _sp("comb_esi_rolling.png")
    # Regime timeline plot
    fig,ax=plt.subplots(figsize=(13,6))
    [ax.axvspan(r["Year"]-0.5,r["Year"]+0.5,
                facecolor=_RC.get(str(r["analysis_regime"]),"#DDD"),alpha=0.25,edgecolor=None)
     for _,r in df.iterrows()]
    ax.plot(df["Year"],df["esi_score"],lw=2,color=_CE,alpha=0.7,zorder=4)
    [ax.scatter(df.loc[df["analysis_regime"].astype(str)==r,"Year"],
                df.loc[df["analysis_regime"].astype(str)==r,"esi_score"],
                color=c,s=55,zorder=5,edgecolors=_CE,linewidths=0.6,label=r)
     for r,c in _RC.items()]
    ax.set_title("India ESI � Regime Timeline",fontweight="bold"); ax.set_xlabel("Year"); ax.set_ylabel("ESI")
    ax.legend(fontsize=9); ax.grid(axis="y",linestyle="--",alpha=0.35); _fmt(ax); plt.tight_layout(); _sp("comb_esi_regime_timeline.png")
    # Change points plot
    fig,ax=plt.subplots(figsize=(13,6)); ax.plot(df["Year"],df["esi_score"],lw=2.2,color=_CE,zorder=5)
    ytop=df["esi_score"].max()
    [ax.axvline(x=yr,color="#D62828",linestyle="--",lw=1.5,alpha=0.8) or
     ax.text(yr,ytop*0.97,str(yr),rotation=90,color="#D62828",fontsize=9,fontweight="bold",ha="right")
     for yr,_ in cps]
    ax.set_title("India ESI � Structural Breaks",fontweight="bold"); ax.set_xlabel("Year"); ax.set_ylabel("ESI")
    ax.grid(axis="y",linestyle="--",alpha=0.4); _fmt(ax); plt.tight_layout(); _sp("comb_esi_change_points.png")
    # Stacked contributions plot
    ic=[c for c in df.columns if c not in ("Year","esi_score","stress_level","analysis_regime",
        "crisis_year","esi_3yr_avg","esi_5yr_avg","z_score","anomaly_z","anomaly_iso")]
    if ic:
        colors=[PALETTE.get(c.replace("_rate","").replace("_growth",""),"#888") for c in ic]
        fig,ax=plt.subplots(figsize=(13,6))
        ax.stackplot(df["Year"],[df[c] for c in ic],
                     labels=[c.replace("_"," ").title() for c in ic],colors=colors,alpha=0.72)
        ax.set_title("India ESI � Stressor Contributions",fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Normalised Stress")
        ax.legend(loc="upper left",fontsize=9,framealpha=0.85)
        ax.grid(axis="y",linestyle="--",alpha=0.3); _fmt(ax); plt.tight_layout(); _sp("comb_esi_stacked.png")
    # Insights
    print(f"\n  Key Insights:")
    bl=bs=be=0; cl=cs=0
    for _,row in df.iterrows():
        if str(row["analysis_regime"])=="High Stress":
            if cl==0: cs=int(row["Year"])
            cl+=1; ce=int(row["Year"])
            if cl>bl: bl,bs,be=cl,cs,ce
        else: cl=0
    print(f"    Longest high-stress streak : {bl} yr(s) ({bs}-{be})")
    pk=int(df.loc[df["esi_score"].idxmax(),"Year"]); pv=df["esi_score"].max()
    print(f"    Peak ESI year              : {pk} (score={pv:.4f})")
    vals=df.sort_values("Year")["esi_score"].tolist(); yrs=df.sort_values("Year")["Year"].tolist()
    bs2,bd=0,int(yrs[0])
    for i in range(len(yrs)-9):
        s=pd.Series(vals[i:i+10]).std()
        if s>bs2: bs2,bd=s,int(yrs[i])
    print(f"    Most volatile decade       : {bd}-{bd+9} (sigma={bs2:.4f})")
    return df

