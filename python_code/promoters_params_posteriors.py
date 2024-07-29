import importlib
import os
import sys

import matplotlib.pylab as plt
import numpy as np

import hxt4auxillary
from impose_priors import impose_new_priors

# we expect repression that
# Mth1: Hxt4 > Hxt2 > Hxt6,7
# Mig1: Hxt6,7 > Hxt2 > Hxt4 > Hxt3
# Std1: Hxt1 > Hxt3 > Hxt4

hxt4model_name = "hxt4model_ig"
posteriors = False


paramsets = {
    "hxt1": "hxt_1_best_particle_model_12.txt",
    # "hxt2": "hxt_2_best_particle_model_11.txt",
    # "hxt3": "hxt_3_best_particle_model_6.txt",
    "hxt4": "int_g_midpoint_parameters_846.txt",
    "hxt6": "hxt_6_best_particle_model_15.txt",
    # "hxt7": "hxt_7_best_particle_model_11.txt",
}
TF_sim_file = "TFs_1p0_hat_846.txt"


paramdir = "fitting_results/"
model_number = {
    k: int(v.split("_")[-1].split(".")[0]) for k, v in paramsets.items()
}
model_number["hxt4"] = 15

if TF_sim_file.split(".")[0].split("_")[-1] not in paramsets["hxt4"]:
    raise Exception("TF simulations are likely with the wrong parameters.")
TF_sims = np.loadtxt(TF_sim_file)
TF_sims = {
    "t": TF_sims[:, 0],
    "mig1": TF_sims[:, 1],
    "mig2": TF_sims[:, 2],
    "mth1": TF_sims[:, 3],
    "std1": TF_sims[:, 4],
}
hxts = list(paramsets.keys())
TFs = ["mth1", "std1", "mig1", "mig2"]


def clean_params(res):
    """Replace forced extreme values of parameters with NaN."""
    for key, value in res.items():
        if key[0] == "n" and (value == 10000000000.0 or value == 100000000.0):
            res[key] = np.nan
            res["k" + key[1:]] = np.nan
        elif key[0] == "k" and (
            value == 10000000000.0 or value == 100000000.0
        ):
            res[key] = np.nan
            res["n" + key[1:]] = np.nan


def load_posterior(posterior, hxt=None, raisepower=False):
    """Load posterior."""
    params = np.loadtxt(paramdir + posterior)
    params_list = [np.array(param) for param in params.tolist()]
    paramset = "hxt4_dummy_variable"
    # define parameters
    p, _ = hxt4auxillary.convert_parameters_from_julia(
        hxt4model_name, paramset, raisepower=raisepower, params=params_list
    )
    p = impose_new_priors(p, paramset)
    # relabel
    if hxt is None:
        hxt = "".join(posterior.split("_")[:2])
    p = {k.replace("hxt4", hxt): v for k, v in p.items()}
    return p


def find_repression(hxt, res, TF_sims, model_number):
    """Calculate the fraction of repression contributed by each TF."""
    mig1_rep = (TF_sims["mig1"] / res[f"k{hxt}mig1"]) ** res[f"n{hxt}mig1"]
    mig2_rep = (TF_sims["mig2"] / res[f"k{hxt}mig2"]) ** res[f"n{hxt}mig2"]
    std1_rep = (TF_sims["std1"] / res[f"k{hxt}std1"]) ** res[f"n{hxt}std1"]
    mth1_rep = (TF_sims["mth1"] / res[f"k{hxt}mth1"]) ** res[f"n{hxt}mth1"]
    ntpts = TF_sims["t"].size
    rres = np.ones((5, ntpts))
    if model_number == 10:
        rres[0, :] = mth1_rep + mig1_rep
        rres[species.index("mig1"), :] = mig1_rep
        rres[species.index("mig2"), :] = np.zeros(ntpts)
        rres[species.index("std1"), :] = np.zeros(ntpts)
        rres[species.index("mth1"), :] = mth1_rep
    elif model_number == 11:
        rres[0, :] = mth1_rep + mig1_rep + mig2_rep
        rres[species.index("mig1"), :] = mig1_rep
        rres[species.index("mig2"), :] = mig2_rep
        rres[species.index("std1"), :] = np.zeros(ntpts)
        rres[species.index("mth1"), :] = mth1_rep
    elif model_number == 12:
        rres[0, :] = mth1_rep + std1_rep
        rres[species.index("mig1"), :] = np.zeros(ntpts)
        rres[species.index("mig2"), :] = np.zeros(ntpts)
        rres[species.index("std1"), :] = std1_rep
        rres[species.index("mth1"), :] = mth1_rep
    elif model_number == 15:
        rres[0, :] = mth1_rep + mig1_rep + mig2_rep + std1_rep
        rres[species.index("mig1"), :] = mig1_rep
        rres[species.index("mig2"), :] = mig2_rep
        rres[species.index("std1"), :] = std1_rep
        rres[species.index("mth1"), :] = mth1_rep
    elif model_number == 6:
        rres[0, :] = mig1_rep + std1_rep
        rres[species.index("mig1"), :] = mig1_rep
        rres[species.index("mig2"), :] = np.zeros(ntpts)
        rres[species.index("std1"), :] = std1_rep
        rres[species.index("mth1"), :] = np.zeros(ntpts)
    else:
        raise Exception(f"Model number {model_number} unrecognised.")
    return rres


def plot_repression_fraction(hxt, res, TF_sims, model_number):
    """Find and plot the repression fraction for all TFs as a stack."""
    rres = find_repression(hxt, res, TF_sims, model_number)
    t = TF_sims["t"]
    # plot
    plt.figure()
    plt.plot(t, rres[0, :], "k")
    plt.fill_between(t, rres[1, :], 0, label="Mig1")
    plt.fill_between(t, np.sum(rres[1:3, :], axis=0), rres[1, :], label="Mig2")
    plt.fill_between(
        t,
        np.sum(rres[1:4:], axis=0),
        np.sum(rres[1:3:], axis=0),
        label="Mth1",
    )
    plt.fill_between(
        t,
        np.sum(rres[1:, :], axis=0),
        np.sum(rres[1:4:], axis=0),
        label="Std1",
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.xlim([0, t.max()])
    plt.title(hxt)
    plt.xlabel("time")
    plt.ylabel("")
    plt.tight_layout()
    plt.show(block=False)


# import model
sys.path.insert(0, os.getcwd() + "/hxt4models")
model = importlib.import_module(hxt4model_name)
hxt4model = model.hxt4model
species = model.species

# define parameters
res = {}
for hxt, paramset in paramsets.items():
    if hxt == "hxt4":
        raisepower = True
        promoters = False
    else:
        raisepower = False
        promoters = True
    p, juliaparams = hxt4auxillary.convert_parameters_from_julia(
        hxt4model_name,
        paramdir + paramset,
        raisepower=raisepower,
        promoters=promoters,
    )
    p = impose_new_priors(p, paramset)
    p = {k.replace("hxt4", hxt): v for k, v in p.items()}
    for key in p:
        if hxt in key:
            # keep only parameters specific to a HXT
            res[key] = p[key]
clean_params(res)

# max concs for TFs
max_conc = {
    "std1": p["std1tot"],
    "mth1": p["smth1"] / p["dmth1"],
    "mig1": p["mig1tot"],
    "mig2": p["smig2"] / p["dmig2"],
}

# optimal values
# plot association constant 1/K for TFs
plt.figure()
for i, TF in enumerate(TFs):
    x, y = [], []
    for key, value in res.items():
        if "khxt" in key and TF in key:
            y.append(1 / value)
            x.append(key.split(TF)[0][1:])
    plt.subplot(eval("22" + str(i + 1)))
    plt.barh(x, y, color=[hxt4auxillary.luiscols[hxt] for hxt in x])
    plt.title(f"1/K for {TF}")
plt.tight_layout()
plt.show(block=False)

# plot Hill number for TF
plt.figure()
for i, TF in enumerate(TFs):
    x, y = [], []
    for key, value in res.items():
        if "nhxt" in key and TF in key:
            y.append(value)
            x.append(key.split(TF)[0][1:])
    plt.subplot(eval("22" + str(i + 1)))
    plt.barh(x, y, color=[hxt4auxillary.luiscols[hxt] for hxt in x])
    plt.title(f"Hill no for {TF}")
plt.tight_layout()
plt.show(block=False)

# plot 1/K**n
plt.figure()
for i, TF in enumerate(TFs):
    x, y = [], []
    for hxt in hxts:
        x.append(hxt)
        y.append(1 / (res["k" + hxt + TF] ** res["n" + hxt + TF]))
    plt.subplot(eval("22" + str(i + 1)))
    plt.barh(x, y, color=[hxt4auxillary.luiscols[hxt] for hxt in x])
    # plt.xscale("log")
    plt.title(f"1/K**n for {TF}")
plt.tight_layout()
plt.show(block=False)

# plot free energy of binding -n*log K
nonspecific = 1e1  # for normalisation
plt.figure()
for i, TF in enumerate(TFs):
    x, y = [], []
    for hxt in hxts:
        x.append(hxt)
        y.append(
            -np.log(res["k" + hxt + TF]) * res["n" + hxt + TF]
            + res["n" + hxt + TF] * nonspecific
        )
    plt.subplot(eval("22" + str(i + 1)))
    plt.barh(x, y, color=[hxt4auxillary.luiscols[hxt] for hxt in x])
    # plt.xscale("log")
    plt.title(f"free energy of binding for {TF}")
plt.tight_layout()
plt.show(block=False)


# plot (TF_max/K)**n
plt.figure()
for i, TF in enumerate(TFs):
    x, y = [], []
    for hxt in hxts:
        x.append(hxt)
        y.append((max_conc[TF] / res["k" + hxt + TF]) ** res["n" + hxt + TF])
    plt.subplot(eval("22" + str(i + 1)))
    plt.barh(x, y, color=[hxt4auxillary.luiscols[hxt] for hxt in x])
    # plt.xscale("log")
    plt.title(f"absolute max repression for {TF}")
plt.tight_layout()
plt.show(block=False)

# plot (TF_max_from_sim/K)**n
plt.figure()
for i, TF in enumerate(TFs):
    x, y = [], []
    for hxt in hxts:
        x.append(hxt)
        y.append(
            (np.max(TF_sims[TF]) / res["k" + hxt + TF]) ** res["n" + hxt + TF]
        )
    plt.subplot(eval("22" + str(i + 1)))
    plt.barh(x, y, color=[hxt4auxillary.luiscols[hxt] for hxt in x])
    # plt.xscale("log")
    plt.title(f"max repression for {TF}")
plt.tight_layout()
plt.show(block=False)

# plot repression fraction
for hxt in hxts:
    plot_repression_fraction(hxt, res, TF_sims, model_number[hxt])


# load posteriors
if posteriors:
    post = {}
    for hxt, paramset in paramsets.items():
        if "modal_parameters" in paramset:
            # for HXT4
            model = 15
            post[hxt] = load_posterior(
                "ABC_distribution_around_central_particle3.txt",
                hxt,
                raisepower=True,
            )
        else:
            post[hxt] = load_posterior(
                paramset.split(".")[0] + "_posterior.txt"
            )
            model = int(paramset.split("_")[-1].split(".")[0])
        if model == 12:
            post[hxt]["k" + hxt + "mig1"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["n" + hxt + "mig1"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["k" + hxt + "mig2"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["n" + hxt + "mig2"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
        elif model == 11:
            post[hxt]["k" + hxt + "std1"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["n" + hxt + "std1"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
        elif model == 10:
            post[hxt]["k" + hxt + "std1"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["n" + hxt + "std1"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["k" + hxt + "mig2"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["n" + hxt + "mig2"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
        elif model == 6:
            post[hxt]["k" + hxt + "mth1"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["n" + hxt + "mth1"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["k" + hxt + "mig2"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )
            post[hxt]["n" + hxt + "mig2"] = np.nan * np.ones(
                post[hxt]["smth1"].size
            )

    # posteriors for affinities
    for TF in TFs:
        plt.figure()
        for hxt in hxts:
            if np.any(~np.isnan(post[hxt]["k" + hxt + TF])):
                plt.hist(
                    post[hxt]["k" + hxt + TF],
                    color=hxt4auxillary.luiscols[hxt],
                    label=hxt,
                    alpha=0.5,
                )
        plt.xscale("log")
        plt.legend()
        plt.title(f"K{TF}")
        plt.show(block=False)
