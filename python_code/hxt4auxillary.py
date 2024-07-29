import numpy as np
from genutils import AttributeDict

luiscols = {
    "HXT1": "#EA49B8",
    "HXT2": "#5DDFED",
    "HXT3": "#8B20D1",
    "HXT4": "#000099",
    "HXT6": "#8DC63F",
    "HXT7": "#41AB5D",
    "hxt1": "#EA49B8",
    "hxt2": "#5DDFED",
    "hxt3": "#8B20D1",
    "hxt4": "#000099",
    "hxt6": "#8DC63F",
    "hxt7": "#41AB5D",
}


def convert_parameters_from_julia(
    smodel,
    paramset,
    raisepower=True,
    promoters=False,
    params=None,
):
    """Create dict of parameters with the standard names."""
    juliaparamnames = define_juliaparamnames(paramset, promoters)
    if params is None:
        params = np.loadtxt(paramset)
    if raisepower:
        if isinstance(paramset, str) and "original" in paramset:
            juliaparams = np.exp(params)
        elif isinstance(paramset, str):
            if isinstance(params, list):
                juliaparams = [10**param for param in params]
            else:
                juliaparams = 10**params
    else:
        juliaparams = params
    # dict of juliaparams with values
    juliaparamdict = {
        paramname: juliaparams[i]
        for i, paramname in enumerate(juliaparamnames)
    }
    # translation dictionary
    transdict = {
        "Snf1tot": "snf1tot",
        "nsnf2": "msnf1",
        "ndhxt4": "mdhxt4",
        "emig1max": "emig1",
        "nmth1snf3": "mmth1snf3",
        "nmth1rgt2": "mmth1rgt2",
        "estd1max3": "estd1snf3",
        "nstd3": "mstd1snf3",
        "estd1max2": "estd1rgt2",
        "nstd1": "mstd1rgt2",
        "dhxt4max": "dhxt4g",
    }
    # dict of parameters for sencillo model
    paramdict = {}
    for paramname in juliaparamnames:
        if paramname in transdict:
            paramdict[transdict[paramname]] = juliaparamdict[paramname]
        else:
            paramdict[paramname] = juliaparamdict[paramname]
    # define parameters
    for paramname in paramdict:
        exec(paramname + "= paramdict[paramname]")
    p = AttributeDict(paramdict)
    return p, juliaparamnames


def define_juliaparamnames(paramset, promoters=False):
    """Define the parameters used for the Julia simulations."""
    if isinstance(paramset, str) and "original" in paramset:
        # PNAS model with SNF1 degrading Mig2 and Snf1tot
        # note "original" in paramset not the model name
        juliaparamnames = [
            "k3",
            "thk",
            "ksnf1",
            "ksnf1std1",
            "nsnf1",
            "nsnf2",
            "Snf1tot",
            "dmth1",
            "nmth1snf3",
            "nmth1rgt2",
            "thsnf3",
            "dmth1rgt2",
            "thmth1",
            "kmth1mig1",
            "nmth1mig1",
            "kmth1mig2",
            "nmth1mig2",
            "std1tot",
            "istd1",
            "nstd1",
            "nstd3",
            "thrgt2",
            "imig1",
            "kmig1snf1",
            "theta2",
            "dmig2",
            "theta",
            "kmig2snf1",
            "smig2",
            "kmig2std1",
            "nmig2std1",
            "kmig2mth1",
            "nmig2mth1",
            "dhxt4",
            "dhxt4max",
            "kdhxt4",
            "ndhxt4",
            "shxt4",
            "khxt4mth1",
            "nhxt4mth1",
            "khxt4std1",
            "nhxt4std1",
            "khxt4mig1",
            "khxt4mig2",
            "nhxt4mig1",
            "nhxt4mig2",
            "estd1max3",
            "ell",
        ]
    elif isinstance(paramset, str):
        juliaparamnames = [
            "k3",
            "thk",
            "ksnf1",
            "ksnf1std1",
            "nsnf1",
            "nsnf2",
            "dmth1",
            "nmth1snf3",
            "nmth1rgt2",
            "thsnf3",  # 10
            "dmth1rgt2",
            "thmth1",
            "kmth1mig1",
            "nmth1mig1",
            "kmth1mig2",
            "nmth1mig2",
            "thstd1",
            "istd1",
            "nstd1",
            "nstd3",  # 20
            "thrgt2",
            "imig1",
            "kmig1snf1",
            "theta2",
            "dmig2",
            "kmig2std1",
            "nmig2std1",
            "kmig2mth1",
            "nmig2mth1",
            "dhxt4",  # 30
            "dhxt4max",
            "kdhxt4",
            "ndhxt4",
            "shxt4",
            "th",  # "khxt4mth1",
            "nhxt4mth1",
            "khxt4std1",
            "nhxt4std1",
            "khxt4mig1",
            "khxt4mig2",
            "nhxt4mig1",
            "nhxt4mig2",
            "estd1max3",
            "ell",
            "thmig2",  # 45 - new ones after here
            "smig2",  # "dmig2g",
            "kdmig2",
            "mdmig2",
            "r_0p2",
            "r_0p4",
            "r_1p0",
            "r_step_0p01",
            "r_step_0p1",
            "r_step_1p0",
            "r_mig1_0p2",
            "r_mig1_0p4",
            "r_mig1_1p0",
            "r_mth1_0p2",
            "r_mth1_0p4",
            "r_mth1_1p0",
            "r_rgt2_0p2",
            "r_rgt2_0p4",
            "r_rgt2_1p0",
            "r_snf3_0p2",
            "r_snf3_0p4",
            "r_snf3_1p0",
            "r_std1_0p2",
            "r_std1_0p4",
            "r_std1_1p0",
        ]
    if promoters:
        # restrict to promoter experiments
        juliaparamnames = juliaparamnames[:52]
        juliaparamnames[-4:] = [
            "r_1p0",
            "r_step_0p01",
            "r_step_0p1",
            "r_step_1p0",
        ]
    return juliaparamnames


def testparams(p, paramset):
    """Check that parameter values are not unphysical."""
    # maximum values
    std1max = p.std1tot
    mth1max = p.smth1 / p.dmth1
    mig1max = p.mig1tot
    mig2max = p.smig2 / p.dmig2
    mig2max_ng = p.smig2 / (p.dmig2 + p.dmig2g)
    gmax = 1
    # regulation
    reg = [
        "Snf1 - std1max/p.ksnf1std1",
        "Snf1 - gmax/p.ksnf1",
        "Hxt4 - mig1max/p.khxt4mig1",
        "Hxt4 - mig2max/p.khxt4mig2",
        "Hxt4 - std1max/p.khxt4std1",
        "Hxt4 - mth1max/p.khxt4mth1",
        f"Hxt4 - {gmax}/p.kdhxt4",
        "Mig2 - mth1max/p.kmig2mth1",
        "Mig2 - std1max/p.kmig2std1",
        "Mth1 - mig1max/p.kmth1mig1",
        "Mth1 - mig2max/p.kmth1mig2",
        "Mth1 Std1 - p.k3",
        "Mth1 Std1 - p.k2",
    ]
    print("maximal numbers\n--")
    print("{:20} = {:.2f}".format("Mth1", mth1max))
    print("{:20} = {:.2f}".format("Std1", std1max))
    print("{:20} = {:.2f}".format("Mig1", mig1max))
    print("{:20} = {:.2f}, {:.2f}".format("Mig2", mig2max, mig2max_ng))
    print("---\nregulation\n---")
    for r in reg:
        bits = r.split("-")
        fc = eval(bits[1].strip())
        print("{:10} {:20}= {:.3f}".format(bits[0], bits[1].strip(), fc))
    print("---\nbias of sensors\n---")
    print(
        "{:20} = {:.2f}".format(
            "Rgt2: Std1 vs Mth1", p.estd1rgt2 / p.dmth1rgt2
        )
    )
    print(
        "{:20} = {:.2f}".format(
            "Snf3: Mth1 vs Std1", p.dmth1snf3 / p.estd1snf3
        )
    )
    print("---\nglucose induced degradation (max/min)\n---")
    print("{:20} = {:.2f}".format("Hxt4", (p.dhxt4 + p.dhxt4g) / p.dhxt4))
    print("{:20} = {:.2f}".format("Mig2", (p.dmig2 + p.dmig2g) / p.dmig2))
    print(
        "{:20} = {:.2f} (Snf3) and {:.2f} (Rgt2)".format(
            "Mth1",
            (p.dmth1 + p.dmth1snf3) / p.dmth1,
            (p.dmth1 + p.dmth1snf3 + p.dmth1rgt2) / (p.dmth1 + p.dmth1snf3),
        )
    )
    print(
        "{:20} = {:.2f} (Snf3) and {:.2f} (Rgt2)".format(
            "Std1",
            (p.istd1 + p.estd1snf3) / p.istd1,
            (p.istd1 + p.estd1snf3 + p.estd1rgt2) / (p.istd1 + p.estd1snf3),
        )
    )
    print()


def near_bounds(param, bnds, eb=0.1):
    """Check if a parameter is near one of its bounds."""
    if (param / bnds[0] < 1 + eb) or (param / bnds[1] > 1 - eb):
        return True
    else:
        return False


def test_bounds(params):
    """Print all parameters close to a boundary value."""
    bnds = {
        "n": (1, 5),
        "m": (1, 5),
        "k": (1e-4, 1e4),
        "s": (1e-2, 1e2),
        "d": (1e-4, 1e2),
        "i": (1e-2, 1e2),
        "e": (1e-2, 1e2),
        "ell": (1e-3, 1e3),
        "std1tot": (1e-2, 2),
    }
    for param in params:
        if param == "mig1tot":
            pass
        elif param in ["ell", "std1tot"]:
            param_bnds = bnds[param]
        elif param[0] in ["n", "m", "k", "s", "d", "i", "e"]:
            param_bnds = bnds[param[0]]
        else:
            param_bnds = False
        if param_bnds and near_bounds(params[param], param_bnds):
            print(f"{param:10} : {params[param]:3.2g}")
