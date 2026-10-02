# Direction 1: Automatic Partitioning — "SAE on Steroids"

## Objective
Evaluate whether a dataset-conditional Marchenko-Pastur (MP) threshold provides a more principled, data-dependent sparsity criterion for Sparse Autoencoders (SAEs) compared to standard L1/ReLU thresholds, thereby better isolating true monosemantic features from noise.

## Core Hypothesis
Superposition mixes rare concepts into the MP bulk. Standard SAEs use a uniform L1 penalty, which fails to distinguish a true domain-specific feature from noise. By replacing the static threshold with an MP bulk edge $\lambda_{max}(D)$ conditional on a dataset $D$, we can dynamically separate signals (features) from the noise floor, categorizing features strictly into:
1. **Universal features:** Above MP threshold for all datasets (e.g., syntax, basic token patterns).
2. **Domain features:** Above threshold only for specific distributions (e.g., agricultural robotics).
3. **Latent/dormant features:** Below threshold during general pretraining, but emergently above threshold in finetuning.
4. **Dead features:** Below threshold for all datasets (uncommitted capacity).

---

## Actionable Step-by-Step Experiment

### Step 1: Data and Model Setup
1. **Select a Target Model:** Choose a well-understood open-source model (e.g., Pythia-70M or Llama-3-8B) to extract activations.
2. **Prepare Datasets ($D$):** 
   - $D_{general}$: A general pretraining corpus sample (e.g., The Pile).
   - $D_{domain\_A}$, $D_{domain\_B}$: Highly specific domain corpora (e.g., biomedical text, agricultural robotics).

### Step 2: Baseline SAE Training
1. **Train an overcomplete SAE** (e.g., $d \rightarrow 16d$) on $D_{general}$ using standard methods (MSE reconstruction + L1 sparsity penalty).
2. **Extract Activation Matrices:** Run forward passes of $D_{general}$, $D_{domain\_A}$, and $D_{domain\_B}$ through the SAE to collect feature $\times$ token activation matrices.

### Step 3: Compute Marchenko-Pastur Thresholds
1. For each dataset $D$, compute the covariance matrix of the SAE feature activations.
2. Calculate the theoretical MP bulk edge $\lambda_{max}(D)$ based on the aspect ratio of the activation matrix.
3. Map features: Identify which feature directions possess variance strictly greater than $\lambda_{max}(D)$.

### Step 4: Baseline Feature Selection Comparisons
To prove MP-thresholding is strictly stronger, compare it against the following standard feature selection/pruning baselines:
*   **Activation Energy / Frequency (The Status Quo):** Filtering features based on purely endogenous metrics, such as activating above a fixed threshold $>\epsilon$ or firing on $>10^{-5}$ of tokens.
*   **Dead Feature Heuristics:** The baseline assumption that features are only "noise" if they literally never fire (or fire almost never) across millions of tokens.
*   **Auto-Annotation (LLM-as-a-Judge):** Prompting a frontier LLM (e.g., GPT-4) with max-activating texts for a feature and evaluating if the LLM can confidently assign a coherent label. Features with low confidence or hallucinated labels are treated as "noise".
*   **Human Annotation:** Blind human interpretability scoring (e.g., using a platform like Neuronpedia) to grade whether a feature represents a distinct, understandable concept.

### Step 5: Evaluation & Metrics
1. **Interpretability Yield:** Do the features selected by the MP-threshold have higher Auto-Annotation and Human Annotation scores than those selected by standard Activation Energy?
2. **Domain Separation:** When computing MP thresholds on $D_{domain\_A}$, do the newly emerging features (Latent $\rightarrow$ Domain) correspond strictly to domain-specific jargon/concepts when auto-annotated, compared to random baseline thresholding?
3. **Reconstruction vs. Sparsity Pareto:** Does zeroing out features below the MP noise floor preserve downstream model performance (KL divergence of logits) better than zeroing out the bottom $N\%$ of features sorted by activation magnitude?
4. **Combating the Illusion of Interpretability:** As highlighted in recent SAE literature, relying solely on top-activating examples often creates an "illusion" of monosemanticity, where human pattern-matching (or LLM auto-annotation) hallucinates a clean concept that fails to generalize across the activation distribution or lacks causal effect.
   - *Activation Prediction (Generality):* Can an LLM provided with the feature's label accurately predict the feature's activation strength on *unseen* text across the full stratified distribution (not just the top-k)?
   - *Causal Intervention (Steering/Ablation):* Does artificially clamping a domain feature to its mean activation effectively ablate the model's ability to process that specific domain concept, proving it is a causal mechanism and not a correlated epiphenomenon?
   - *Long-Tail Consistency:* Are the features rigorously consistent in the lower/middle quantiles of their activation distribution, or does the interpretable concept break down?

### Step 6: Analysis of "The Substrate"
*   Investigate the features trapped *inside* the MP bulk (the noise). Are these truly unformed concepts in superposition? Provide qualitative examples showing that directions below the MP threshold are polysemantic "soup", while those breaking $\lambda_{max}(D)$ are crisp, monosemantic concepts.