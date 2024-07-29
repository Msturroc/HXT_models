import matplotlib.pylab as plt
import numpy as np
from matplotlib import gridspec


def plothxt4(sol, p, glu_function, species, title=None):
    """
    Plot results for Hxt4 and TFs.

    TFs scaled by thresholds for binding HXT4's promoter.
    """
    nt = sol.t
    ny = sol.y
    g = glu_function(nt)
    # change subplot heights
    fig = plt.figure(figsize=(5, 3))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 3], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    # plot model's Hxt4
    ymax = 4
    ax1.plot(nt, ny[0, :], label="hxt4", color="dimgray")
    ax1.set_ylim(bottom=0, top=ymax)
    ax1.set_xlim([0, np.max(nt)])
    ax1.set_ylabel("Hxt4")
    ax1.set_xticklabels([])
    # plot glucose
    ax1.fill_between(x=nt, y1=g / np.max(g) * ymax, color="orange", alpha=0.1)
    # TFs' activities
    colors = ["#B40F20", "#E58601", "navy", "#46ACC8"]
    TFs = ["mig1", "mig2", "mth1", "std1"]
    activity, activity_Hill = [], []
    for TF in TFs:
        activity.append((ny[species.index(TF), :] / eval("p.khxt4" + TF)))
        activity_Hill.append(
            (ny[species.index(TF), :] / eval("p.khxt4" + TF))
            ** eval("p.nhxt4" + TF)
        )
    activity = np.array(activity)
    activity_Hill = np.array(activity_Hill)
    if True:
        # plot TF activities
        for cindex, TF in enumerate(TFs):
            TF_activity = activity[TFs.index(TF), :]
            ax2.plot(
                nt,
                np.log2(TF_activity),
                label=TF,
                color=colors[cindex],
            )
            ax2.fill_between(
                x=nt[np.log2(TF_activity) > 0],
                y1=np.log2(TF_activity)[np.log2(TF_activity) > 0],
                color="silver",
                alpha=0.2,
            )
    else:
        # plot concentrations
        for sp in species[1:]:
            ax2.plot(nt, np.log10(ny[species.index(sp), :]), label=sp)
    ax2.set_xlim([0, np.max(nt)])
    ax2.set_ylim(
        [1.1 * np.log2(np.min(activity)), 1.1 * np.log2(np.max(activity))]
    )
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.set_xlabel("time (hrs)")
    ax2.grid()
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)


def plothxt4_old(sol, p, gluvalue, glu_function, species, title=None):
    nt = sol.t
    ny = sol.y
    g = glu_function[gluvalue](nt)
    # change subplot heights
    fig = plt.figure(figsize=(5, 3))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 3], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    # plot model's Hxt4
    ymax = 4
    ax1.plot(nt, ny[0, :], label="hxt4", color="dimgray")
    ax1.set_ylim(bottom=0, top=ymax)
    ax1.set_xlim([0, np.max(nt)])
    ax1.set_ylabel("Hxt4")
    ax1.set_xticklabels([])
    # plot glucose
    ax1.fill_between(x=nt, y1=g / np.max(g) * ymax, color="orange", alpha=0.1)
    # TFs' activities
    colors = ["#B40F20", "#E58601", "navy", "#46ACC8"]
    TFs = ["mig1", "mig2", "mth1", "std1"]
    activity, activity_Hill = [], []
    for TF in TFs:
        activity.append((ny[species.index(TF), :] / eval("p.khxt4" + TF)))
        activity_Hill.append(
            (ny[species.index(TF), :] / eval("p.khxt4" + TF))
            ** eval("p.nhxt4" + TF)
        )
    activity = np.array(activity)
    activity_Hill = np.array(activity_Hill)
    if True:
        # plot TF activities
        for cindex, TF in enumerate(TFs):
            TF_activity = activity[TFs.index(TF), :]
            ax2.plot(
                nt,
                np.log2(TF_activity),
                label=TF,
                color=colors[cindex],
            )
            ax2.fill_between(
                x=nt[np.log2(TF_activity) > 0],
                y1=np.log2(TF_activity)[np.log2(TF_activity) > 0],
                color="silver",
                alpha=0.2,
            )
    else:
        # plot concentrations
        for sp in species[1:]:
            ax2.plot(nt, np.log10(ny[species.index(sp), :]), label=sp)
    ax2.set_xlim([0, np.max(nt)])
    ax2.set_ylim(
        [1.1 * np.log2(np.min(activity)), 1.1 * np.log2(np.max(activity))]
    )
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.set_xlabel("time (hrs)")
    ax2.grid()
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
