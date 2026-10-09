"""Scientific figures from the saved, frozen experiment outcomes."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"figures"
INK="#192b3b";GRID="#dce1e4";GRAY="#687783";ORANGE="#be553c";BLUE="#236a98";GREEN="#507967"
COLORS={"baseline":GRAY,"teacher_student":ORANGE,"pseudo_student":BLUE,"oracle_student":GREEN}
NAMES={"baseline":"Splink baseline","teacher_student":"LLM-label student","pseudo_student":"Splink-label student","oracle_student":"Oracle-label student"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"text.color":INK,"axes.labelcolor":INK,
                     "xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.spines.left":False,"axes.spines.bottom":False,"axes.titleweight":"bold",
                     "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})


def save(fig,name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/f"{name}.png",dpi=170,bbox_inches="tight")
    fig.savefig(OUT/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)


def acquisition():
    df=pd.read_csv(HERE/"results.csv").query("fp_cost==2")
    fig,axes=plt.subplots(2,2,figsize=(12,7.2),sharex=True,sharey=True)
    fig.suptitle("Diagnostic grouping added no held-out linkage gain",x=.06,ha="left",fontsize=19)
    fig.text(.06,.915,"Every policy and update branch retained 779 of 781 true links, with zero false links.",fontsize=12.5)
    arms=[("random","Random review",GRAY),("uncertainty_diversity","Uncertainty + diversity",BLUE),
          ("clusters","Diagnostic groups",ORANGE),("error_risk","Predicted error risk",GREEN)]
    for ax,(arm,title,color) in zip(axes.ravel(),arms):
        ax.set_title(title,loc="left",fontsize=14,pad=13)
        ax.axhline(2,color=GRAY,linestyle=(0,(4,3)),lw=1.4,zorder=1,label="Splink baseline")
        for branch,marker,label in [("ordinary","o","Ordinary update"),("repair","x","Repair + update")]:
            values=df[(df.arm==arm)&(df.branch==branch)].groupby("additional_budget").decision_loss.agg(["mean","min","max"])
            ax.plot(values.index,values["mean"],marker=marker,color=color,lw=2.0,markersize=9,
                    markeredgewidth=2,label=label,zorder=3 if marker=="x" else 2)
            ax.vlines(values.index,values["min"],values["max"],color=color,lw=2)
        for budget in [100,200]:
            ax.text(budget,2.24,"2",ha="center",fontsize=13,fontweight="bold",color=color)
        ax.set_ylim(0,4);ax.set_xlim(75,225);ax.set_xticks([100,200]);ax.set_yticks([0,1,2,3,4]);ax.grid(axis="y",color=GRID,zorder=0)
        ax.text(.05,.1,"Ordinary = repair",transform=ax.transAxes,color=color,fontsize=11)
    axes[0,0].set_ylabel("Loss: 2 FP + FN")
    axes[1,0].set_ylabel("Loss: 2 FP + FN")
    for ax in axes[1]:ax.set_xlabel("Additional oracle pair queries")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.52,.12),ncol=3,frameon=False,fontsize=11)
    fig.text(.06,.045,"Five paired acquisition seeds per point; min–max ranges are zero. Each arm also receives 100 common seed queries.\nCommon overhead: 400 audit + 241 validation + 255 calibration queries. Shuffled and score-only controls also tie.\nSynthetic FEBRL4; complete test denominator. Oracle queries do not measure staff review time.",fontsize=10,color=GRAY,linespacing=1.45)
    fig.subplots_adjust(left=.09,right=.96,top=.84,bottom=.28,hspace=.38,wspace=.23)
    save(fig,"acquisition-loss")


def student_panel():
    df=pd.read_csv(HERE/"results.csv").query("fp_cost==2").set_index("arm")
    names=["baseline","teacher_student","pseudo_student","oracle_student"]
    values=[int(df.loc[name,"fn"]) for name in names]
    fig,ax=plt.subplots(figsize=(11.6,5.4))
    y=np.arange(4)
    bars=ax.barh(y,values,color=[COLORS[n] for n in names],height=.55)
    ax.set_yticks(y,[NAMES[n] for n in names]);ax.invert_yaxis();ax.set_xlim(0,33);ax.set_xticks([0,5,10,15,20,25,30])
    ax.grid(axis="x",color=GRID);ax.set_axisbelow(True)
    for bar,n,value in zip(bars,names,values):
        ax.text(value+.5,bar.get_y()+bar.get_height()/2,str(value),va="center",fontweight="bold",fontsize=14,color=COLORS[n])
    ax.set_xlabel("Missed true links, out of 781")
    fig.suptitle("LLM supervision lost accuracy on the structured control",x=.055,ha="left",fontsize=18)
    fig.text(.055,.865,"The same cheap pair scorer and 289 retained training pairs; only the label source changes.",fontsize=12)
    ax.text(17,3.1,"False links: 0 in every arm\nFalse co-clustered pairs: 0\nOne-to-one assignment: identical errors",fontsize=11,color=GRAY,va="center")
    fig.text(.055,.06,"Fresh teacher: 300 calls; 11 insufficient-evidence answers excluded from every student's training.\nIndependent calibration: 255 oracle queries; FP cost 2, FN cost 1. No teacher calls on test entities.\nFEBRL4 has two records per entity. This is a learned pair score; no record embedding was trained.",fontsize=10.5,color=GRAY,linespacing=1.45)
    fig.subplots_adjust(left=.245,right=.95,top=.77,bottom=.265)
    save(fig,"student-febrl4")


def transfer_panel():
    df=pd.read_csv(HERE/"febrl3-transfer/results.csv").query("fp_cost==2").set_index("arm")
    names=["baseline","teacher_student","pseudo_student","oracle_student"]
    fig,axes=plt.subplots(1,2,figsize=(12.7,6.2),sharey=True)
    y=np.arange(4)
    for ax,fpcol,fncol,title in [(axes[0],"fp","fn","Accepted edge errors"),
                                (axes[1],"implied_false_pairs","missed_true_cluster_pairs","Relationships after graph closure")]:
        fp=[int(df.loc[n,fpcol]) for n in names];fn=[int(df.loc[n,fncol]) for n in names]
        ax.barh(y-.15,fp,height=.28,color=ORANGE,label="False identity")
        ax.barh(y+.15,fn,height=.28,color=BLUE,label="Missed identity")
        for i,(a,b) in enumerate(zip(fp,fn)):
            ax.text(a+.4,i-.15,str(a),va="center",fontsize=12,fontweight="bold",color=ORANGE)
            ax.text(b+.4,i+.15,str(b),va="center",fontsize=12,fontweight="bold",color=BLUE)
        ax.set_title(title,loc="left",fontsize=13.5,pad=13);ax.set_xlim(0,33)
        ax.set_xticks([0,5,10,15,20,25,30]);ax.grid(axis="x",color=GRID);ax.set_axisbelow(True)
        ax.set_xlabel("Error count")
    axes[0].set_yticks(y,[NAMES[n] for n in names]);axes[0].invert_yaxis()
    fig.suptitle("Three false edges became 27 false cluster relationships",x=.05,ha="left",fontsize=18)
    fig.text(.05,.88,"Frozen FEBRL4 models transferred to multi-record FEBRL3: 733 records, 296 entities, groups of up to 6.",fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.58,.18),ncol=2,frameon=False,fontsize=11)
    fig.text(.05,.065,"943 true identity pairs among 268,278 possible pairs. Same sparse candidates; 400 common calibration queries.\nClosure recovers one missing Splink edge through a valid path. The teacher student creates 3 false-merge components.\nSynthetic cross-corpus transfer; no new teacher calls or student retraining. Repeated observations permit larger groups.",fontsize=10.5,color=GRAY,linespacing=1.45)
    fig.subplots_adjust(left=.205,right=.97,top=.78,bottom=.34,wspace=.18)
    save(fig,"student-febrl3-graph")


if __name__=="__main__":
    acquisition();student_panel();transfer_panel()
