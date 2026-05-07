# TaskExtrapolation
Github for the paper "Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation"

| Paper Section | Experiment Script | Results / Slurm File | Corresponding Figures | Notes / Comments |
| :--- | :--- | :--- | :--- | :--- |
| **Section 1 & App E.1:** Motivating Example (Projectile Motion) |Fig1A.py | | | |
| **Section 3.1:** Parameter Extrapolation (Synthetic) |FuncExtrap_no_label.py |slurm-func-37217034.out | | |
| **Section 3.2:** Length Extrapolation (Synthetic) |LengthExtrapNoLabel.py |slurm-neural-37217567.out | | |
| **Section 3.3:** Composition Extrapolation (Synthetic) |CompositionExtrapNoLabel.py |slurm-comp-nl-37217791.out | | |
| **Section 4.1:** LLM Sparse Parity (Length Extrapolation) | | | | |
| **Section 4.2:** LLM CodeIO (Compositional String Transformations) |CompositionLearner.py |final_results.json | | |
| **Appendix B.1.1:** Multi-Step Parameter Extrapolation | | | | |
| **Appendix B.1.2:** Multi-Step Compositional Extrapolation | | | | |
| **Appendix B.2:** EM Pseudo-labels / Relaxing Meta-Label Assumptions |LatentTraining.py | | | |
| **Appendix H.2:** Latent Space / Manifold Analysis Visualizations |CodeioVis.py | | | |
| **Appendix K.1:** Parameter Extrapolation Ablations | | | | |
| **Appendix K.2:** Length Extrapolation Ablations | | | | |
| **Appendix K.3:** Composition Extrapolation Ablations | | | | |
