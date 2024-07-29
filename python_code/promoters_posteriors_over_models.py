# draw bar chart of posterior probabilities of different promoter models
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import genutils as gu

direc = "fitting_2024/"
errors = False

models = [
    "M2",
    "M1",
    "M1_M2",
    "S",
    "S_M2",
    "S_M1",
    "S_M1_M2",
    "M",
    "M_M2",
    "M_M1",
    "M_M1_M2",
    "M_S",
    "M_S_M2",
    "M_S_M1",
    "M_S_M1_M2",
]

if False:
    models = [
        "____ ____ ____ Mig2",
        "____ ____ Mig1 ____",
        "____ ____ Mig1 Mig2",
        "____ Std1 ____ ____",
        "____ Std1 ____ Mig2",
        "____ Std1 Mig1 ____",
        "____ Std1 Mig1 Mig2",
        "Mth1 ____ ____ ____",
        "Mth1 ____ ____ Mig2",
        "Mth1 ____ Mig1 ____",
        "Mth1 ____ Mig1 Mig2",
        "Mth1 Std1 ____ ____",
        "Mth1 Std1 ____ Mig2",
        "Mth1 Std1 Mig1 ____",
        "Mth1 Std1 Mig1 Mig2",
    ]


priors = {
    "hxt1": [4, 5, 6, 7, 12, 13, 14, 15],
    "hxt2": [8, 9, 10, 11, 12, 13, 14, 15],
    "hxt3": [4, 5, 6, 7, 12, 13, 14, 15],
    "hxt6": [8, 9, 10, 11, 12, 13, 14, 15],
    "hxt7": [8, 9, 10, 11, 12, 13, 14, 15],
}

hxts = ["hxt1", "hxt2", "hxt3", "hxt6", "hxt7"]
regs = ["Std1", "Mth1", "Mig1", "Mig2"]

# load data into dataframe
if errors:
    ddict = {
        "gene": [],
        "error": [],
        "model": [],
        "prob": [],
        "Std1": [],
        "Mth1": [],
        "Mig1": [],
        "Mig2": [],
    }
else:
    ddict = {
        "gene": [],
        "model": [],
        "prob": [],
        "Std1": [],
        "Mth1": [],
        "Mig1": [],
        "Mig2": [],
    }


no_reps = 10
for hxt in hxts:
    if errors:
        errors = np.loadtxt(direc + "final_error_" + hxt + ".txt")
    # posterior probabilities
    probs = np.zeros((len(models), no_reps))
    loaded_probs = np.loadtxt(direc + "final_p" + hxt[-1] + ".txt")
    for i, model_i in enumerate(priors[hxt]):
        probs[model_i - 1] = loaded_probs[i, :]
    # store in dict
    for i in range(len(models)):
        for j in range(no_reps):
            ddict["gene"].append(hxt.upper())
            ddict["model"].append(models[i])
            if errors:
                ddict["error"].append(errors[j])
            ddict["prob"].append(probs[i, j])
            for tf in regs:
                if tf in models[i]:
                    ddict[tf].append(1)
                else:
                    ddict[tf].append(0)
df = pd.DataFrame.from_dict(ddict)

# find relative errors
if errors:
    df["relerror"] = df["error"]
    for hxt in hxts:
        df.loc[df.gene == hxt.upper(), "relerror"] = (
            df[df.gene == hxt.upper()].error / df[df.gene == hxt.upper()].error.min()
        )

# stacked plot of posteriors
fdf = df[["gene", "model", "prob"]].reset_index(drop=True)
fdf.prob /= no_reps
# add HXT4
fdf.loc[-1] = ["HXT4", "Mth1 Std1 Mig1 Mig2", 1]
fdf.reset_index(drop=True)
# remove rejected model
chosen_models = list(fdf[fdf.prob > 0].model.unique())
fdf = fdf.loc[fdf.model.str.contains("|".join(chosen_models))]
# define order
fdf["gene"] = pd.Categorical(
    fdf["gene"], ["HXT1", "HXT3", "HXT2", "HXT4", "HXT6", "HXT7"]
)
fdf = fdf.sort_values(by=["gene", "model"])
# bar graph
plt.figure()
ax = sns.histplot(
    fdf[fdf.prob > 0],
    x="gene",
    hue="model",
    weights="prob",
    multiple="stack",
    shrink=0.8,
    palette=sns.color_palette("Set2"),  # sns.color_palette("tab10"),
)
ax.set_ylabel("prob")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show(block=False)


# to pdf
# gu.figs2pdf("promotor_posteriors.pdf")
