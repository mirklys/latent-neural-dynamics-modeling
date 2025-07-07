### **RQ 1**

**To what extent can computational neural modeling frameworks predict neural activity across different recording modalities?**

This assesses the efficacy of Preferential Subspace Identification (PSID) [3], a linear state-space modeling framework, and Dissociative Prioritized Analysis of Dynamics (DPAD) [4], a nonlinear recurrent neural network (RNN) framework, as tools for cross-modality neural signal prediction. The primary objective is to train these models to predict Electrocorticography (ECoG) signals from Local Field Potentials (LFPs) in Parkinson's disease patients during the DBS OFF state [1] and addtionally fit stereoelectroencephalography (sEEG) recordings to infer ECoG activity of epilepsy patients during resting-state [2]. This initial phase will also serve to validate the computational pipelines for PSID and DPAD, ensuring their robust functionality for following RQs.

### **RQ 2**

**How accurately can latent dynamical models classify discrete brain states from neural signals, and what do their learned representations reveal about the underlying neurophysiology?**

I will evaluate the performance of PSID and DPAD in classifying DBS ON and OFF states. Using simultaneously recorded LFP and ECoG signals from Parkinson's disease patients performing tasks in the CopyDraw experiments, the models will be adapted for binary classification. Both AUC and cross-validated correlation coefficient (CC) will be used to asses the classification performance. A key component of this part is to analyze the interpretable latent state representations learned by each model, aiming to elucidate the distinct neural activity patterns that characterize the DBS ON and OFF conditions.

### **RQ 3**

**How effectively can linear and nonlinear dynamical models decode continuous motor behaviour from neural activity, and what do their latent dynamics reveal about the neural control of movement?**

I will aim to predict a continuous behavioural variable—tracing speed—from LFP activity recorded during the CopyDraw task by training both PSID and DPAD. A central part will involve examining the temporal dynamics of the low-dimensional latent states extracted by each model. Specifically, this will focus on identifying and characterizing state-space changes and correlating these features with continuous variations in tracing speed. Both linear and non-linear models will be compared, and analysis of prediction residuals will also be conducted to assess unexplained variance in the models.

### References

1. Dold, M., Pereira, J., Sajonz, B., Coenen, V.A., Thielen, J., Janssen, M.L.F., Tangermann, M. (2024): LFP and ECoG data during CopyDraw task with deep brain stimulation - Dareplane data for proof of concept paper. Version 1. Radboud University. (dataset).
https://doi.org/10.34973/d214-m342
2. Frauscher B, von Ellenrieder N, Zelmann R, Doležalová I, Minotti L, Olivier A, Hall J, Hoffmann D, Nguyen DK, Kahane P, Dubeau F, Gotman J. Atlas of the normal intracranial electroencephalogram: neurophysiological awake activity in different cortical areas. Brain. 2018 Apr 1;141(4):1130-1144. doi: 10.1093/brain/awy035. PMID: 29506200.
3. Omid G. Sani, Hamidreza Abbaspourazad, Yan T. Wong, Bijan Pesaran, Maryam M. Shanechi. Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification. Nature Neuroscience, 24, 140–149 (2021). https://doi.org/10.1038/s41593-020-00733-0
4. Omid G. Sani, Bijan Pesaran, Maryam M. Shanechi. Dissociative and prioritized modeling of behaviorally relevant neural dynamics using recurrent neural networks. Nature Neuroscience (2024). https://doi.org/10.1038/s41593-024-01731-2
