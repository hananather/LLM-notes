"""Render the saved evaluation; no labels, fitting or API calls are needed."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT=json.loads((HERE/"results/evaluation.json").read_text())
ARMS=[(RESULT["primary_raw"],"risk_2","Tree · 32 features"),
      (RESULT["primary_extracted"],"risk_2","Extraction → Splink"),
      ("semantic_selector","fixed","Semantic selector")]
COLORS=["#737574","#427E86","#CC6D4D"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.labelcolor":"#242424",
    "text.color":"#242424","axes.spines.top":False,"axes.spines.right":False})


def save(fig,name):
    fig.savefig(HERE/(name+".png"),dpi=180,bbox_inches="tight")
    fig.savefig(HERE/(name+".svg"),bbox_inches="tight")
    plt.close(fig)


def compare():
    fig,axs=plt.subplots(1,3,figsize=(15,5.3),gridspec_kw={"wspace":.42})
    measures=[("exact_set_accuracy","Complete organization sets","% of all 1,104 source rows",100),
        ("false_edges_per_100_queries","False organization allocations","Edges per 100 source rows",1),
        ("automatic_coverage","Automatic decisions","% of all 1,104 source rows",100)]
    for ax,(key,title,ylabel,mult) in zip(axs,measures):
        vals=[RESULT["scores"][a][p]["all"]["automatic"][key]*mult for a,p,_ in ARMS]
        bars=ax.bar(range(3),vals,color=COLORS,width=.62)
        ax.set_title(title,loc="left",fontweight="bold",fontsize=13,pad=18)
        ax.set_ylabel(ylabel);ax.set_xticks(range(3),["Tree\n32 features","Extraction\n→ Splink","Semantic\nselector"],fontsize=10)
        ax.set_ylim(0,110 if mult==100 else max(vals)*1.2);ax.set_axisbelow(True);ax.grid(axis="y",alpha=.17)
        for bar,value in zip(bars,vals):
            ax.annotate(f"{value:.1f}"+("%" if mult==100 else ""),(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,6),textcoords="offset points",ha="center",fontweight="bold")
    fig.suptitle("Native affiliations: complete sets, false allocations and review",x=.075,y=1.0,ha="left",fontweight="bold",fontsize=17)
    fig.text(.075,.015,"All 1,104 public affiliation rows; ROR v1.41. Tree and expanded supervised Splink policies selected on original S2AFF validation.\nReview and invalid responses remain unresolved. Descriptive benchmark; source rows are not established unique publications.",fontsize=9)
    fig.subplots_adjust(bottom=.25,top=.82,left=.075,right=.98);save(fig,"comparison")


def counts():
    fig,ax=plt.subplots(figsize=(10,5.1))
    vals=[RESULT["allocation_count_errors"][a][p]["all"]["organization_count_l1_error"] for a,p,_ in ARMS]
    bars=ax.barh(range(3),vals,color=COLORS,height=.6)
    ax.set_yticks(range(3),[label for _,_,label in ARMS]);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.2)
    ax.set_axisbelow(True);ax.grid(axis="x",alpha=.17)
    ax.set_title("Error in organization-level affiliation-row counts",loc="left",fontweight="bold",fontsize=16,pad=20)
    ax.set_xlabel("Sum of absolute organization count errors, Σ |predicted − reference|")
    for bar,value in zip(bars,vals):
        ax.annotate(f"{value:,}",(bar.get_width(),bar.get_y()+bar.get_height()/2),xytext=(7,0),textcoords="offset points",va="center",fontweight="bold")
    fig.text(.03,.02,"Counts refer to benchmark affiliation-row allocations. Review is unresolved. These are not official publication counts.\nOver- and under-allocation can cancel within an organization; false and missing edges are reported separately.",fontsize=9)
    fig.subplots_adjust(bottom=.22,left=.27,right=.94,top=.83);save(fig,"allocation-count-error")


def cohorts():
    fig,ax=plt.subplots(figsize=(11,5.2));x=np.arange(3);width=.24
    populations=["french","multilingual","multi-org"]
    for i,(arm,policy,label) in enumerate(ARMS):
        vals=[RESULT["scores"][arm][policy][c]["automatic"]["exact_set_accuracy"]*100 for c in populations]
        bars=ax.bar(x+(i-1)*width,vals,width,color=COLORS[i],label=label)
        for bar,value in zip(bars,vals):
            ax.annotate(f"{value:.1f}%",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords="offset points",ha="center",fontsize=9)
    ax.set_ylim(0,110);ax.set_xticks(x,["French · 614 rows","Multilingual · 322 rows","Multiple organizations · 168 rows"])
    ax.set_ylabel("Complete sets (% of all source rows)")
    ax.set_axisbelow(True);ax.grid(axis="y",alpha=.17)
    ax.set_title("Complete organization sets in each native-text collection",loc="left",fontweight="bold",fontsize=16,pad=20)
    ax.legend(loc="upper left",bbox_to_anchor=(0,1.0),frameon=False,fontsize=9,ncol=3)
    fig.text(.07,.02,"Frozen methods and policies; review and invalid remain unresolved. Historical complete-set annotations.\nThese collections describe a public benchmark and do not estimate a national research-output population.",fontsize=9)
    fig.subplots_adjust(bottom=.23,top=.8,left=.08,right=.98);save(fig,"cohort-comparison")


if __name__=="__main__":
    compare();counts();cohorts()
    print("Rendered three saved-result figures in PNG and SVG; no API calls.")
