# Applying Marchenko-Pastur (MP) Thresholding to Image Sparse Autoencoders (Vision SAEs)

As Sparse Autoencoders (SAEs) are increasingly applied to Vision Transformers (ViTs) and CNNs to interpret visual representations (e.g., CLIP, DINO, or stable diffusion components), distinguishing true visual latents from reconstructive noise remains a critical challenge.

The Marchenko-Pastur (MP) thresholding technique can be adapted from text (token-wise) to images (patch-wise or pixel-wise) to rigorously identify and prune non-semantic noise features from the SAE dictionary.

## 1. The Core Analogy: Tokens to Patches
In text models, SAEs reconstruct the activations of discrete tokens in a sequence context. In Vision Transformers, SAEs reconstruct the activations of continuous **image patches** (e.g., $16 \times 16$ spatial regions). 
- **Text**: $N$ tokens $\times$ $d_{model}$
- **Vision**: $N$ patches (or spatial tokens) $\times$ $d_{model}$

When we project these patch activations into the SAE dictionary space, we generate a high-dimensional feature activation matrix where MP principles apply.

## 2. Step-by-Step Application for Image SAEs

### Step A: Collect Patch Activations
Sample a diverse distribution of images (e.g., ImageNet, LAION, or COCO) and run them through the vision model. Extract the layer activations $X \in \mathbb{R}^{N \times d_{model}}$, where $N$ is the total number of flattened patches across all images in the sample.

### Step B: Train the Vision SAE
Train the SAE (Standard L1 or Top-K) to reconstruct $X$. Let $f(x)$ be the encoder. 
The encoded features for the dataset form the Feature Activation Matrix $A \in \mathbb{R}^{N \times D_{dict}}$, where $D_{dict}$ is the number of learned visual features.

### Step C: Construct the Covariance Matrix
Center the matrix $A$ along the patch dimension (subtract the mean activation for each feature). 
Compute the covariance matrix of the feature activations:
$$ C = \frac{1}{N} A^T A $$

### Step D: Calculate the MP Eigenvalue Spectrum
Compute the eigenvalues of $C$. Under the assumption that noisy, non-meaningful features resemble random uncorrelated uniform noise, their eigenvalues will fall within the Marchenko-Pastur theoretical bulk:
$$ \lambda_{\pm} = \sigma^2 \left(1 \pm \sqrt{\frac{D_{dict}}{N}}\right)^2 $$
*(Note: $\sigma^2$ is estimated from the variance of the lowest eigenvalues).*

### Step E: Thresholding and Segregation
- **Sub-Threshold Features (The Bulk):** Eigenvalues $\lambda \le \lambda_+$. These correspond to vectors that merely absorb local spatial variations, high-frequency compression artifacts, or $L_1$ shrinkage noise. They are **visually "dead"** and have no semantic meaning (e.g., they will not consistently activate for "dog snouts" or "curved edges").
- **Supra-Threshold Features (The Signal):** Eigenvalues $\lambda > \lambda_+$. These break away from the random matrix prediction. They represent robust, monosemantic visual concepts that reliably generalize across different images.

## 3. Unique Challenges in Vision Application

### Challenge 1: Spatial Correlation (Violating IID)
Unlike text tokens which have complex linguistic dependencies, adjacent image patches are *highly* correlated (e.g., a blue sky spans many patches). This violates the Independent and Identically Distributed (IID) assumption of random matrix theory.
**Solution:** Apply spatial subsampling. Instead of using all patches from every image to construct matrix $A$, randomly sample only a few non-adjacent patches per image to force statistical independence before computing the MP bounds.

### Challenge 2: Background vs. Object Foreground
Vision SAE models often allocate massive capacity to background colors or ubiquitous textures (flat walls, gradients). The MP distribution might shift if the null hypothesis contains heavy-tailed variance from pervasive background textures. 
**Solution:** Domain-specific MP thresholding. Calculate separate MP thresholds for heavily textured datasets vs. object-centric datasets to see which structural features are universal limits versus domain-specific artifacts.

## 4. Why This Matters for Vision SAEs
Vision SAEs are notorious for producing features that look like "pointillist noise" or slight variations of edge detectors when visualized via maximal activation cropping. 

By applying MP thresholding, researchers can:
1. **Mathematically prove** which specific SAE features are pure reconstruction noise.
2. Filter out hyper-specific texture-matching artifacts before running human-in-the-loop annotations or automated CLIP-based feature scoring.
3. Optimize the dictionary size ($D_{dict}$) and Sparsity ($K$) dynamically by monitoring the mass of the eigenvalue spectrum that escapes the MP bounded bulk.
