"""Saved-result figure for the separately frozen697-row comparison."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
INK="#192b3b";GRID="#dce1e4";GRAY="#687783";BLUE="#236a98";ORANGE="#be553c"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11.5,"text.color":INK,"axes.labelcolor":INK,
    "xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,
    "axes.spines.left":False,"axes.spines.bottom":False,"axes.titleweight":"bold","savefig.facecolor":"white"})


def main():
    result=json.loads((HERE/"results.json").read_text())
    methods=["legacy_"+result["legacy_champions"]["raw"],result["champions"]["raw"],"semantic_selector"]
    records=[]
    for arm in methods:
        policy="fixed" if arm=="semantic_selector" else "risk_2"
        records.append(next(r for r in result["outcomes"] if r["arm"]==arm and r["policy"]==policy))
    fig,axes=plt.subplots(1,3,figsize=(13.8,6.2),sharey=True)
    y=np.arange(3);colors=[GRAY,BLUE,ORANGE]
    specifications=[("exact_sets","Complete organization sets",500,[0,100,200,300,400],"Correct sets, out of 697 rows"),
                    ("fp","False allocations",190,[0,40,80,120,160],"Wrong organization edges"),
                    ("fn","Missed allocations",1520,[0,300,600,900,1200],"Missed edges, out of 1,693 true edges")]
    for ax,(field,title,maximum,ticks,label) in zip(axes,specifications):
        values=[r[field] for r in records]
        ax.barh(y,values,height=.5,color=colors)
        for i,(value,row) in enumerate(zip(values,records)):
            text=f"{value} ({100*row['exact_set_accuracy']:.1f}%)" if field=="exact_sets" else f"{value:,}"
            ax.text(value+maximum*.025,i,text,va="center",fontweight="bold",fontsize=12,color=colors[i])
        ax.set_title(title,loc="left",fontsize=13.5,pad=16)
        ax.set_xlim(0,maximum);ax.set_xticks(ticks);ax.grid(axis="x",color=GRID);ax.set_axisbelow(True)
        ax.set_xlabel(label,fontsize=11,labelpad=13)
    axes[0].set_yticks(y,["External-label tree","Native-label tree","Semantic selector"]);axes[0].invert_yaxis()
    fig.suptitle("Semantic gains survive native-label adaptation",x=.04,ha="left",fontsize=20)
    fig.text(.04,.875,"Same 697 held-out affiliation rows and 40 candidates per row; 1,693 reference organization edges.",fontsize=12)
    fig.text(.04,.115,"Native supervision: 195 training and 212 validation rows. Both tree champions use 32 raw features and validation-selected policies.\nSemantic selection recovers 160 more complete sets than native adaptation: +22.96 percentage points (paired 95% interval 18.79–27.37).",fontsize=10.5,color=GRAY,linespacing=1.5)
    fig.text(.04,.045,"Semantic review: 135 rows remain unresolved and count in the denominator. Unseen texts within the curated cohort; organizations can recur.\nThe complete 1,104-row external evaluation remains separate. Allocation counts concern source affiliations, not official publication totals.",fontsize=10.3,color=GRAY,linespacing=1.5)
    fig.subplots_adjust(left=.18,right=.98,top=.76,bottom=.30,wspace=.38)
    out=HERE/"figures";out.mkdir(exist_ok=True)
    for extension in ["png","svg"]:fig.savefig(out/("adaptation-comparison."+extension),dpi=175,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":main()
