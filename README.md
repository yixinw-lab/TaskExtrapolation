# TaskExtrapolation
Github for the paper "Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation"

| Paper Section | Experiment Script | Results / Slurm File | Corresponding Figures | Notes / Comments |
| :--- | :--- | :--- | :--- | :--- |
| **Section 1 & App E.1:** Motivating Example (Projectile Motion) |Fig1A.py |N/A |fig1_horizontal_compact.pdf | |
| **Section 3.1:** Parameter Extrapolation (Synthetic) |FuncExtrap_no_label.py |slurm-func-37217034.out |ParameterExtrap |Didn't include every image in paper to save space |
| **Section 3.2:** Length Extrapolation (Synthetic) |LengthExtrapNoLabel.py |slurm-neural-37217567.out |LengthExtrap |Didn't include every image in paper to save space |
| **Section 3.3:** Composition Extrapolation (Synthetic) |CompositionExtrapNoLabel.py |slurm-comp-nl-37217791.out |CompExtrap |Didn't include every image in paper to save space |
| **Section 4.1:** LLM Sparse Parity (Length Extrapolation) | | | |N/A |
| **Section 4.2:** LLM CodeIO (Compositional String Transformations) |CompositionLearner.py |final_results.json |N/A | |
| **Appendix B.1.1:** Multi-Step Parameter Extrapolation |FuncExtrapMultiStep.py |FuncMultiStep.txt.rtf |N/A |Ran without bash, output copy pasted |
| **Appendix B.1.2:** Multi-Step Compositional Extrapolation |CompositionExtrapMultiStep.py |CompMultiStep.txt.rtf |N/A |Ran without bash, output copy pasted |
| **Appendix B.2:** EM Pseudo-labels / Relaxing Meta-Label Assumptions |LatentTraining.py | |N/A | |
| **Appendix H.2:** Latent Space / Manifold Analysis Visualizations |VizLatent.py | |TaskManifolds_Final.png | |
| **Appendix J:** Latent Space CodeIO |CodeioVis.py | | | |
| **Appendix K.1:** Parameter Extrapolation Ablations |FuncExtrap_Ablations.py |slurm-func-abl-37089077.out |N/A | |
| **Appendix K.2:** Length Extrapolation Ablations |LengthExtrap_Ablations.py |slurm-len-abl-37088156.out |N/A | |
| **Appendix K.3:** Composition Extrapolation Ablations |CompositionExtrap_Ablations.py |slurm-comp-abl-37087641.out |N/A | |
