## Push-Pull Model of Yeast Glucose Transporter Regulation

This repository provides Julia and Python implementations for the mathematical model presented in Montaño-Gutierrez et al., describing how budding yeast (*Saccharomyces cerevisiae*) matches the expression of hexose transporter (*HXT*) genes to extracellular glucose concentrations<sup>1</sup>.

**Model Overview**

The core of the model is a "push-pull" system involving two pairs of transcriptional repressors. As extracellular glucose levels rise, the activity of repressors Mth1 and Std1 is reduced ('pulled'), while the activity of repressors Mig1 and Mig2 is increased ('pushed'). Conversely, falling glucose levels lead to increased Mth1/Std1 activity and decreased Mig1/Mig2 activity<sup>1</sup>. This dynamic interplay creates an intracellular readout of the external glucose concentration<sup>1</sup>.

The model incorporates the glucose sensing pathways involving the Snf3 and Rgt2 sensors and the SNF1 kinase complex, which modulate the activity states of these four key repressors<sup>1</sup>.

*HXT* genes encoding transporters with different affinities for glucose (low, medium, high) exhibit distinct expression patterns<sup>1</sup>. The model proposes that this is achieved through differential coupling of their promoters to the two repressor pairs<sup>1</sup>. Low-affinity transporter promoters (e.g., *HXT1*, *HXT3*) are primarily regulated by Mth1/Std1, leading to expression mainly in high glucose when this repression is lifted<sup>1</sup>. High-affinity transporter promoters (e.g., *HXT6*, *HXT7*) are primarily regulated by Mig1/Mig2, restricting their expression to low glucose conditions before Mig1/Mig2 activity increases<sup>1</sup>. Medium-affinity transporter promoters (e.g., *HXT2*, *HXT4*) respond to both pairs, allowing expression predominantly in intermediate glucose concentrations<sup>1</sup>.

**Code and Analyses**

This repository contains code in both Julia and Python to:

*   Simulate the system of differential equations describing the regulatory network dynamics under various glucose conditions<sup>1</sup>.
*   Perform model selection analysis, allowing comparison of different promoter structures (i.e., which repressors regulate a specific *HXT* gene)<sup>1</sup>.
*   Conduct global sensitivity analysis to assess the influence of model parameters on the system's behaviour<sup>1</sup>.

**Reference**

1.  The model, its development, and biological validation are detailed in the following publication<sup>1</sup>:

    Montaño-Gutierrez, L. F., Sturrock, M., Farquhar, I. L., Correia, K., Shahrezaei, V., & Swain, P. S. (2024). *A push-pull system of repressors matches levels of glucose transporters to extracellular glucose in budding yeast*. (preprint)

