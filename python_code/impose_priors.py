import numpy as np


def impose_new_priors(p, paramset):
    """With Mig2 degradation regulated by glucose."""
    # impose priors
    p.mig1tot = 1
    p.snf1tot = 1
    # Snf3 has a lower affinity than Rgt2
    p.k2 = p.k3 + p.thk
    # Rgt2 affects Std1 more than Snf3
    # Snf3 affects Mth1 more than Rgt2
    p.estd1rgt2 = p.dmth1rgt2 + p.thrgt2
    p.dmth1snf3 = p.estd1snf3 + p.thsnf3
    # enough Std1 to bind pMIG2
    p.std1tot = p.kmig2std1 + p.thstd1
    # effect of glucose on Mig2
    p.dmig2g = (10**p.thmig2) * p.dmig2
    # effect of SNF1 on Mig1
    p.emig1 = (10**p.theta2) * p.imig1
    # more Mth1 than Std1
    p.smth1 = p.dmth1 * (p.std1tot + p.thmth1)
    # Mth1 binds pHXT4 - maximal Mth1 > khxt4mth1
    p.khxt4mth1 = p.smth1 / p.dmth1 * np.exp(-p.th)
    return p


def impose_priors_off(p, paramset):
    """OLD. With Mig2 degradation regulated by SNF1."""
    # impose priors
    p.mig1tot = 1
    # Snf3 has a lower affinity than Rgt2
    p.k2 = p.k3 + p.thk
    if "original" in paramset:
        # for compatibility with PNAS version only
        # Rgt2 affects Std1 more than Snf3
        # Snf3 affects Mth1 more than Rgt2
        p.estd1rgt2 = p.dmth1rgt2 * p.istd1 / p.dmth1 + p.thrgt2
        p.dmth1snf3 = p.estd1snf3 * p.dmth1 / p.istd1 + p.thsnf3
    else:
        # Rgt2 affects Std1 more than Snf3
        # Snf3 affects Mth1 more than Rgt2
        p.estd1rgt2 = p.dmth1rgt2 + p.thrgt2
        p.dmth1snf3 = p.estd1snf3 + p.thsnf3
        # enough Std1 to bind pHXT4
        p.std1tot = p.khxt4std1 + p.thstd1
        # enough Mig2 to bind pHXT4
        p.smig2 = p.dmig2 * p.khxt4mig2 + p.thmig2
    # effect of SNF1 on Mig1
    p.emig1 = (10**p.theta2) * p.imig1
    # effect of SNF1 on Mig2
    p.dmig2snf1 = (10**p.theta) * p.dmig2
    # more Mth1 than Std1
    p.smth1 = p.dmth1 * (p.std1tot + p.thmth1)
    return p


def impose_new_priors_old(p, paramset):
    """With Mig2 degradation regulated by glucose."""
    # impose priors
    p.mig1tot = 1
    # Snf3 has a lower affinity than Rgt2
    p.k2 = p.k3 + p.thk
    # Rgt2 affects Std1 more than Snf3
    # Snf3 affects Mth1 more than Rgt2
    p.estd1rgt2 = p.dmth1rgt2 + p.thrgt2
    p.dmth1snf3 = p.estd1snf3 + p.thsnf3
    # enough Std1 to bind pHXT4
    p.std1tot = p.khxt4std1 + p.thstd1
    # enough Mig2 to bind pHXT4
    p.smig2 = p.dmig2 * p.khxt4mig2 + p.thmig2
    # effect of SNF1 on Mig1
    p.emig1 = (10**p.theta2) * p.imig1
    # more Mth1 than Std1
    p.smth1 = p.dmth1 * (p.std1tot + p.thmth1)
    return p
