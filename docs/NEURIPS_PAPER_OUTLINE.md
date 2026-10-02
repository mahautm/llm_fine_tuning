# Mechanistic Interpretability via Dataset-Conditional Marchenko-Pastur Thresholding

## 1. Core Thesis & Theoretical Grounding
Mechanistic interpretability currently relies on arbitrary sparsity penalties (e.g., L1 coefficients) or heuristic activation thresholds (e.g., frequency > $10^{-5}$) to separate true latent features from the "noise substrate" of transformer residual streams (Bricken et al., 2023; Cunningham et al., 2023). We propose that Random Matrix Theory, specifically the Marchenko-Pastur (MP) distribution (Marchenko & Pastur, 1967), provides a mathematically rigorous, un-parameterized boundary for this threshold. Furthermore, by conditioning $\lambda_{max}(D)$ on specific data distributions ($D_{general}$ vs. $D_{domain}$), we can systematically isolate latent features that govern domain-specific behavior.

## 2. Established Empirical Successes
*   **The MP Noise Floor is Real (H1/H2):** Pruning 1,473 "substrate" features below the MP noise threshold in a Pythia-70m SAE resulted in negligible reconstruction loss ($MSE \to 0.1207$). Top-K architectures (Gao et al., 2024) natively enforce this by avoiding feature spread below the MP bulk.
*   **Domain Separation & The Illusion of Interpretability (H3/H4):** Contrasting The Pile ($D_{general}$) with IMDb ($D_{domain}$) revealed 297 text features that broke the MP threshold *only* in the target domain. This effect holds across modalities (Vision: CIFAR-100 vs CIFAR-10 yielding 208 visual latents). Automated LLM-as-a-judge annotation (Llama-3-8B) confirmed high semantic coherence (0.92 confidence).
*   **Causal Efficacy Against Placebos (H5):** Feature steering along MP-isolated vectors shifts behavior deterministically (e.g., smoothly scaling Label Flip Ratios from 0% to 100% in Vision models). Importantly, this causal effect is statistically significant when compared against orthonormal placebo vector injections over the same decoder norm. 
*   **Frontier Scaling Laws (H6):** The extraction and MP-thresholding framework successfully scales to large-parameter regimes, validated on `Mistral-7B-v0.3` (Layer 16, $d_{model}=4096$, Dictionary Size=65,536, MSE=0.0015).

## 3. Gap Analysis & Vulnerabilities (What reviewer 2 will attack)
While the pipeline demonstrates strong baseline results, writing out the narrative highlights several epistemological and experimental gaps:

*   **Gap 1: The "Dead Feature" Confound:** 
    *   *Intuition:* Are features below the MP threshold truly "noise", or are they simply ultra-rare (monosemantic but low-frequency) features that the MP bulk absorbs due to sample size?
    *   *Missing Experiment:* We need to explicitly measure if any features pruned below the MP threshold possess high LLM-annotator confidence or causal efficacy. If pruning drops *interpretable* features, then MP is too aggressive.
*   **Gap 2: Cross-Layer Dynamics (Depth Dependency):**
    *   *Intuition:* We sampled Pythia Layer 3 and Mistral Layer 16. However, residual stream geometry changes drastically from early to late layers (Alain & Bengio, 2014). The MP bulk might perfectly separate noise in middle layers but fail at the unembedding bottleneck.
    *   *Missing Experiment:* A full-depth ablation. Calculate $\lambda_{max}(D)$ across all layers of a small model (e.g., Pythia-70m) and track the ratio of features surviving the MP threshold as depth increases.
*   **Gap 3: Polysemanticity inside the "True" Signal:**
    *   *Intuition:* We claim features above the threshold are "true" latents. But do they suffer from polysemanticity? Just because a feature breaks the MP threshold in a domain doesn't guarantee it represents a single, understandable concept.
    *   *Missing Experiment:* Run neighborhood overlap or feature-splitting tests (e.g., via Batch Top-K) on the 297 isolated IMDb features to ensure they are monosemantic, rather than densely entangled domain-vectors.

## 4. Next Experimental Steps
To reinforce the paper for a top-tier venue, we should execute the following probes:
1.  **Inverse Causal Steering:** Take 5 features *below* the MP threshold, annotate them, and steer them. Prove they act like the placebo vectors (i.e., zero KL shift). 
2.  **Layer-wise MP Spectrum:** Generate a plot of the MP threshold vs. Layer Depth for Pythia-70m.

## References
1. Alain, G., & Bengio, Y. (2014). Understanding intermediate layers using linear classifier probes. *ICLR*.
2. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Anthropic*.
3. Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *arXiv*.
4. Gao, L., et al. (2024). Scaling and Evaluating Sparse Autoencoders. *OpenAI*.
5. Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Math. USSR-Sbornik*.
