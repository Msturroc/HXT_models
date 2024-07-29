import numpy as np

species = ["hxt4", "mig1", "mig2", "mth1", "std1"]


def hxt4model(t, y, p, glucose, intracellular_glucose, *args):
    # extracellular glucose
    g = glucose(t, *args)
    # intracellular glucose
    ig = intracellular_glucose(t)
    # proteins
    hxt4 = y[0]
    mig1 = y[1]
    mig2 = y[2]
    mth1 = y[3]
    std1 = y[4]
    if np.any(y < 0):
        print(y, "negative at", t)
    dydt = np.empty(len(y))
    # SNF1
    snf1 = (
        p.snf1tot
        * (1 + std1 / p.ksnf1std1) ** p.nsnf1
        / ((1 + std1 / p.ksnf1std1) ** p.nsnf1 + p.ell * (1 + ig / p.ksnf1) ** p.msnf1)
    )
    # Hxt4
    dydt[0] = (
        p.shxt4
        / (
            1
            + (mig1 / p.khxt4mig1) ** p.nhxt4mig1
            + (mig2 / p.khxt4mig2) ** p.nhxt4mig2
            + (mth1 / p.khxt4mth1) ** p.nhxt4mth1
            + (std1 / p.khxt4std1) ** p.nhxt4std1
        )
        - p.dhxt4 * hxt4
        - p.dhxt4g * hxt4 / (1 + (ig / p.kdhxt4) ** p.mdhxt4)
    )
    # Mig1
    dydt[1] = p.imig1 * (p.mig1tot - mig1) - p.emig1 * snf1 * mig1 / (
        p.kmig1snf1 + mig1
    )
    # Mig2
    dydt[2] = (
        p.smig2
        / (
            1
            + (mth1 / p.kmig2mth1) ** p.nmig2mth1
            + (std1 / p.kmig2std1) ** p.nmig2std1
        )
        - p.dmig2 * mig2
        - p.dmig2snf1 * snf1 * mig2 / (p.kmig2snf1 + mig2)
    )
    # Mth1
    dydt[3] = (
        p.smth1
        / (
            1
            + (mig1 / p.kmth1mig1) ** p.nmth1mig1
            + (mig2 / p.kmth1mig2) ** p.nmth1mig2
        )
        - p.dmth1 * mth1
        - p.dmth1snf3 * mth1 * g**p.mmth1snf3 / (p.k3**p.mmth1snf3 + g**p.mmth1snf3)
        - p.dmth1rgt2 * mth1 * g**p.mmth1rgt2 / (p.k2**p.mmth1rgt2 + g**p.mmth1rgt2)
    )
    # Std1
    dydt[4] = (
        p.istd1 * (p.std1tot - std1)
        - p.estd1snf3 * std1 * g**p.mstd1snf3 / (p.k3**p.mstd1snf3 + g**p.mstd1snf3)
        - p.estd1rgt2 * std1 * g**p.mstd1rgt2 / (p.k2**p.mstd1rgt2 + g**p.mstd1rgt2)
    )
    return dydt
