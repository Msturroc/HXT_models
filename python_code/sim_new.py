import importlib
import os
import sys

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from genutils import AttributeDict
from matplotlib import gridspec
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import hxt4auxillary
from impose_priors import impose_new_priors
from plothxt4 import plothxt4

# from SDB:
# Mth1 is 3600 molecules; Std1 is 700; Mig1 is 2000;
# Mig2 is 2200; Snf1 is 5600; Gal83 is 3500

# defaults
plot_hat_mutants = False
export_for_promoters = False
cols = sns.color_palette("tab10")

ode_method = "LSODA"
rtol = 1.0e-11
atol = rtol / 1e3
max_step = 0.01

paramdir = "fitting_results/"
# 901, 915, 940, 943, 946 - Std1 drops immediately
# 950 - odeint crashes
# 945 - large Std1
# 928, 936, 949, 955, 977, 979 - k2=k3
# *922 - Mig2 irrelevant for HXT4; strong bias of sensors; Mig2 represses MTH1
# 963, 979 similar to 922
# 970 - Std1 irrelevant for HXT4
# 971 - Std1 probably drops too rapidly; poor fit Std1 delta
# 903, 909 - fit to snf3 delta too poor
# 924, 944, 948, 960 - snf3 fit poorer than 955 and 977
# 934 - OK but green peak below orange in WT
# 906, 964 - Ok
#
# ---
#
# 906: Std1 decreases in 0.2%; Std1 > Mig1
# 924: Std1 decreases in 0.2%; Std1 = Mth1 > Mig1
# 964: Std1 decreases only slightly; Std1 = Mth1 > Mig1
# 944: Std1 decreases; Std1 = Mth1 > Mig1
# 948: Std1 decreases; Std1=Mth1 of order Mig1
# 960: Std1 decreases; Std1=Mth1 of order Mig1
paramset = "int_g_midpoint_parameters_964.txt"

# specify model
sys.path.insert(0, os.getcwd() + "/hxt4models")
hxt4model_name = "hxt4model_ig"

# import model
model = importlib.import_module(hxt4model_name)
hxt4model = model.hxt4model
species = model.species

# define parameters
p, juliaparams = hxt4auxillary.convert_parameters_from_julia(
    hxt4model_name, paramdir + paramset
)
p = impose_new_priors(p, paramset)

# load data
newdata = "../data/mean_HXTs_for_fitting.csv"
df = pd.read_csv(newdata)

# define glucose as a function of time
glu = {}
for experiment in df.experiment_type.unique():
    for strain in df.strain.unique():
        if "HXT4" in strain:
            sdf = df[
                (df.experiment_type == experiment)
                & (df.strain == strain)
                & (df.signal_type == "glucose_proxy")
            ]
            t = sdf.time.to_numpy()
            glu_proxy = sdf.signal_value.to_numpy()
            if np.any(glu_proxy):
                glu[f"{experiment}_{strain}"] = interp1d(
                    t, glu_proxy, fill_value="extrapolate"
                )


def gluconstant(t, gc):
    """Return constant glucose."""
    return gc


def zero_intracellular_glucose(t):
    """Return zero glucose."""
    return 0


def init_constant_glucose(p, gc=0):
    """Define initial conditions as steady state with no glucose."""
    init = np.zeros(len(species))
    sol = solve_ivp(
        hxt4model,
        (0, 1.0e6),
        init,
        args=(p, gluconstant, zero_intracellular_glucose, gc),
        method=ode_method,
        rtol=rtol,
        atol=atol,
    )
    return sol.y[:, -1]


def initialise(p, fdf):
    """Set initial conditions."""
    init = init_constant_glucose(p, gc=0)
    # Hxt4 set to empirical value
    empHxt4 = fdf[(fdf.time == 0)].signal_value.to_numpy()[0]
    t = fdf.time.to_numpy()
    init[0] = empHxt4
    print(f" Initial Hxt4 is {empHxt4:.2f}")
    return t, init


def intracellular_glucose(t, g, r, type):
    """Find intracellular glucose assuming logistic growth."""
    gmax = np.max(g)
    if "hat" in type:
        # indices where glucose rises and falls
        updown_indices = np.where((g > 1e-6) & (g < gmax - 1e-6))[0]
        sep_i = np.diff(updown_indices).argmax() + 1
        # mean time during rise of glucose
        up_t = np.mean(t[updown_indices[:sep_i]])
        up_i = np.argmin((t - up_t) ** 2)
        # mean time during fall of glucose
        down_t = np.mean(t[updown_indices[sep_i:]])
        down_i = np.argmin((t - down_t) ** 2)
    elif "step" in type:
        # indices where glucose rises
        up_indices = np.where(g > 1e-6)[0]
        up_i = up_indices[0]
        # up_t = t[up_i]
        down_i = t.size - 1
    # small initial value
    gmin = 1e-7  # 10 pM so 1 molecule given 1% glucose is 55 mM
    gi = gmin * np.ones(t.shape)
    # logistic growth in extracellular glucose only
    gi_in_glucose = gmax / (
        1 + (gmax - gmin) / gmin * np.exp(-r * (t - t[up_i]))
    )
    gi[up_i : down_i + 1] = gi_in_glucose[up_i : down_i + 1]
    # linearly interpolate
    gi_fun = interp1d(t, gi)
    return gi_fun


# helper functions
def print_params(s):
    """Print params containing string s."""
    for k in p:
        if s in k:
            print(f"{k:10} {p[k]:4.3f}")


def print_extrema(sol):
    """Print maxima for each species."""
    for g in sol:
        print()
        for s in species:
            print(
                f"{g}: {s} from {sol[g].y[species.index(s),:].min():.2f}"
                f" to {sol[g].y[species.index(s),:].max():.2f}"
            )


def get_conc(exp_key, point=False):
    """Find concentration of glucose."""
    conc = exp_key.split("percent")[0].split("_")[2]
    if point:
        conc = conc.replace("p", ".")
    return conc


def find_r_name(exp_key):
    """Find name of parameter for intracellular glucose."""
    conc = get_conc(exp_key)
    strain = exp_key.split("percent")[1][1:]
    if "hat" in exp_key:
        if "delta" in strain:
            r_name = f"r_{strain.split('_')[2]}_{conc}"
        else:
            r_name = f"r_{conc}"
    elif "step" in exp_key:
        r_name = f"r_step_{conc}"
    return r_name


###
# run for all glucose concentrations
###

# test params
hxt4auxillary.test_bounds(p)
hxt4auxillary.testparams(p, paramset)

# simulate
sol = {}
for exp_key in glu:
    print(exp_key)
    # define mutants
    mp = AttributeDict(p.copy())
    if "snf3_delta" in exp_key:
        mp.dmth1snf3 = 0
        mp.estd1snf3 = 0
    elif "rgt2_delta" in exp_key:
        mp.dmth1rgt2 = 0
        mp.estd1rgt2 = 0
    elif "mth1_delta" in exp_key:
        mp.smth1 = 0
    elif "std1_delta" in exp_key:
        mp.std1tot = 0
    elif "mig1_delta" in exp_key:
        mp.mig1tot = 0
    # for intracellular glucose
    r_name = find_r_name(exp_key)
    print(f" Using {r_name}")
    r_ig = p[r_name]
    # data to find initial value of Hxt4-GFP
    fdf = df[
        (df.strain == exp_key.split("percent")[1][1:])
        & (df.experiment_type.str.contains(exp_key.split("percent")[0][:-1]))
        & (df.signal_type == "fluorescence")
    ]
    t, init = initialise(mp, fdf)
    ig_fun = intracellular_glucose(t, glu[exp_key](t), r_ig, type=exp_key)
    # simulate
    sol[exp_key] = solve_ivp(
        hxt4model,
        (0, np.max(t)),
        init,
        args=(mp, glu[exp_key], ig_fun),
        t_eval=t,
        method=ode_method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )


if export_for_promoters:
    # for promoter comparisons
    res = sol["hat_glucose_1p0_percent_HXT4_GFP"]
    # t, 'mig1', 'mig2', 'mth1', 'std1'
    np.savetxt(
        f"TFs_1p0_hat_{paramset.split('_')[-1]}",
        np.concatenate((res.t.reshape(1, -1), res.y[1:, :])).T,
    )
    sys.exit()


###
# plot results
###


# plot TFs for wild-type hats
wt_hat = [key for key in sol.keys() if "hat" in key and "delta" not in key]
for exp_key in wt_hat:
    plothxt4(sol[exp_key], p, glu[exp_key], species, title=paramset)


# plot the concentraton of each species for hats
marc_allstrains = [
    "WT",
    "mig1_delta",
    "mth1_delta",
    "rgt2_delta",
    "snf3_delta",
    "std1_delta",
]
allstrains = marc_allstrains
# one plot for each species: Hxt4, Mig1, etc.
# we have data only for Hxt4
for protein_to_plot in ["hxt4"]:  # species:
    fig = plt.figure(figsize=(5, 10))
    plt.suptitle(f"{paramset} : {protein_to_plot}")
    gs = gridspec.GridSpec(nrows=6, ncols=1, figure=fig)
    # plot protein_to_plot for each strain
    for i, strain in enumerate(allstrains):
        axL = fig.add_subplot(gs[i, 0])
        if strain == "WT":
            exp_keys = [
                key for key in sol if "hat" in key and "delta" not in key
            ]
        else:
            exp_keys = [key for key in sol if "hat" in key and strain in key]
        # plot three concentrations of glucose per strain
        for j, exp_key in enumerate(exp_keys):
            if protein_to_plot == "hxt4":
                # find data
                sdf = df[
                    df.experiment_type.str.contains("hat")
                    & df.experiment_type.str.contains(get_conc(exp_key))
                    & (df.signal_type == "fluorescence")
                ]
                if strain == "WT":
                    sdf = sdf[sdf.strain == "HXT4_GFP"]
                else:
                    sdf = sdf[sdf.strain.str.contains(strain)]
                # plot data
                axL.plot(
                    sdf.time.to_numpy(),
                    sdf.signal_value.to_numpy(),
                    ":",
                    alpha=0.7,
                )
                fit_score = np.mean(
                    (sdf.signal_value.to_numpy() - sol[exp_key].y[0, :]) ** 2
                )
            # plot protein
            axL.plot(
                sol[exp_key].t,
                sol[exp_key].y[species.index(protein_to_plot), :],
                color=cols[j],
                label=get_conc(exp_key, point=True),
            )
        if protein_to_plot == "hxt4":
            axL.text(2, 0.8 * axL.get_ylim()[1], f"{strain}; {fit_score:0.2e}")
        else:
            axL.text(2, 0.8 * axL.get_ylim()[1], f"{strain}")
        if strain == "WT":
            plt.legend()
    axL.set_xlabel("time (h)", horizontalalignment="center")
    plt.tight_layout()
    plt.show(block=False)

if False:
    # plot extracellular glucose
    plt.figure()
    for g in glu:
        plt.plot(t, glu(t), label=g)
    plt.xlabel("time")
    # plt.legend()
    plt.show(block=False)


# plot results for steps
wt_step = [key for key in sol.keys() if "step" in key and "delta" not in key]
plt.figure(figsize=(5, 2.25))
plt.suptitle(paramset)
for j, exp_key in enumerate(wt_step):
    sdf = df[
        df.experiment_type.str.contains("step")
        & df.experiment_type.str.contains(get_conc(exp_key))
        & (df.strain == "HXT4_GFP")
        & (df.signal_type == "fluorescence")
    ]
    plt.plot(sdf.time.to_numpy(), sdf.signal_value.to_numpy(), ":", alpha=0.7)
    plt.plot(
        sol[exp_key].t,
        sol[exp_key].y[0, :],
        color=cols[j],
        label=get_conc(exp_key, point=True),
    )
plt.legend()
plt.xlabel("time (h)", horizontalalignment="center")
plt.tight_layout()
plt.show(block=False)


if False:
    # plot intracellular g values
    r_dict = {
        "experiment": [key for key in p if "r_" in key],
        "r_value": [p[key] for key in p if "r_" in key],
    }
    r_df = pd.DataFrame.from_dict(r_dict)
    r_df["strain"] = r_df.experiment.str.split("_", expand=True)[1]
    r_df.strain = r_df.strain.replace(
        {s: "WT" for s in r_df.strain.values if (s == "step") or ("0" in s)}
    )
    # bar plot of r
    plt.figure()
    g = sns.barplot(r_df, x="r_value", y="experiment", hue="strain")
    g.set_xscale("log")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show(block=False)
    # plots of g
    t = np.linspace(0, 20, 200)
    for strain in r_df.strain.unique():
        if strain == "WT":
            exp_keys = [key for key in sol if "delta" not in key]
        else:
            exp_keys = [key for key in sol if strain in key]
        plt.figure(figsize=(8, 6))
        for exp_key in exp_keys:
            r_ig = p[find_r_name(exp_key)]
            ig_fun = intracellular_glucose(t, glu[exp_key](t), r_ig, exp_key)
            plt.plot(t, ig_fun(t), label=experiment)
        plt.yscale("log")
        plt.legend()
        plt.xlabel("time")
        plt.title(f"intracellular glucose for {strain}")
        plt.tight_layout()
        plt.show(block=False)
