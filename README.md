## Push-Pull Model of Yeast Glucose Transporter Regulation

This repository provides Julia and Python implementations for the mathematical model presented in Montaño-Gutierrez et al., describing how budding yeast (*Saccharomyces cerevisiae*) matches the expression of hexose transporter (*HXT*) genes to extracellular glucose concentrations[1].

**Model Overview**

The core of the model is a "push-pull" system involving two pairs of transcriptional repressors. As extracellular glucose levels rise, the activity of repressors Mth1 and Std1 is reduced ('pulled'), while the activity of repressors Mig1 and Mig2 is increased ('pushed'). Conversely, falling glucose levels lead to increased Mth1/Std1 activity and decreased Mig1/Mig2 activity[1]. This dynamic interplay creates an intracellular readout of the external glucose concentration[1].

The model incorporates the glucose sensing pathways involving the Snf3 and Rgt2 sensors and the SNF1 kinase complex, which modulate the activity states of these four key repressors[1].

*HXT* genes encoding transporters with different affinities for glucose (low, medium, high) exhibit distinct expression patterns[1]. The model proposes that this is achieved through differential coupling of their promoters to the two repressor pairs[1]. Low-affinity transporter promoters (e.g., *HXT1*, *HXT3*) are primarily regulated by Mth1/Std1, leading to expression mainly in high glucose when this repression is lifted[1]. High-affinity transporter promoters (e.g., *HXT6*, *HXT7*) are primarily regulated by Mig1/Mig2, restricting their expression to low glucose conditions before Mig1/Mig2 activity increases[1]. Medium-affinity transporter promoters (e.g., *HXT2*, *HXT4*) respond to both pairs, allowing expression predominantly in intermediate glucose concentrations[1].

**Code and Analyses**

This repository contains code in both Julia and Python to:

*   Simulate the system of differential equations describing the regulatory network dynamics under various glucose conditions[1].
*   Perform model selection analysis, allowing comparison of different promoter structures (i.e., which repressors regulate a specific *HXT* gene)[1].
*   Conduct global sensitivity analysis to assess the influence of model parameters on the system's behaviour[1].

**Reference**

The model, its development, and biological validation are detailed in the following publication[1]:

Montaño-Gutierrez, L. F., Sturrock, M., Farquhar, I. L., Correia, K., Shahrezaei, V., & Swain, P. S. (2024). *A push-pull system of repressors matches levels of glucose transporters to extracellular glucose in budding yeast*. (Preprint/Published Paper - *Note: Add full citation details once available*)
