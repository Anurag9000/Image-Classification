# The Supreme Codex of Mastery
## Ultimate Research & Implementation Architecture

This document is the **definitive, textbook-level authority** on the Image Classification & Representation Learning Repository. It deconstructs the codebase into its atomic mathematical and theoretical components, providing the intuition, implementation details, and optimal configuration strategies for every single lever in the system.

---

# Part I: The Hybrid Backbone Architecture
**File:** `pipeline/backbone.py` | **Class:** `HybridBackbone`

The backbone is the "eye" of the system. It is designed to solve the **Inductive Bias vs. Global Context** trade-off by fusing a Convolutional Neural Network (CNN) with a Vision Transformer (ViT).

## 1. The Dual-Stream Logic
*   **CNN Stream (`cnn_backbone`)**:
    *   **Role**: Extracts high-frequency, local features (edges, textures, shapes).
    *   **Math**: $F_{cnn} = \phi_{cnn}(X) \in \mathbb{R}^{B \times C_c \times H/32 \times W/32}$.
    *   **Implementation**: Utilizes `timm.create_model`. Common choice: `convnextv2_base`.
    *   **Inductive Bias**: Translation Equivariance & Locality.
*   **ViT Stream (`vit_backbone`)**:
    *   **Role**: Extracts low-frequency, global semantic features (object relationships, scene context).
    *   **Math**: $F_{vit} = \phi_{vit}(X) \in \mathbb{R}^{B \times C_v}$.
    *   **Implementation**: Utilizes `timm.create_model`. Common choice: `swinv2_base_window12_192_22k`.
    *   **Inductive Bias**: Permutation Invariance (weak), Long-range dependency modeling.

## 2. The Fusion Mechanism
*   **Dynamic Head Attention (DHA)**:
    *   **Problem**: Simply concatenating CNN and ViT features treats them equally, even if one stream is noisy.
    *   **Solution**: A learnable gating mechanism that independently re-weights each channel of the combined feature vector.
    *   **Math**:
        $$ Z = [F_{cnn} || F_{vit}] $$
        $$ \alpha = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot Z)) $$
        $$ Z_{refined} = Z \odot \alpha $$
    *   **Code**: `DynamicHeadAttention` class.
    *   **Intuition**: The model learns to "turn off" the CNN stream if the texture is misleading, or "turn off" the ViT stream if the global context is irrelevant.

## 3. Attention Modules (The "Focus" Mechanisms)
### CBAM: Convolutional Block Attention Module
*   **Switch**: `BackboneConfig(use_cbam=True)`
*   **Theory**: Enhances the CNN stream by explicitly modeling "What" (Channel) and "Where" (Spatial) is important.
    1.  **Channel Attention**:
        $$ M_c(F) = \sigma(MLP(\text{AvgPool}(F)) + MLP(\text{MaxPool}(F))) $$
        *Intuition*: "Is the 'Fur' channel more important than the 'Background' channel?"
    2.  **Spatial Attention**:
        $$ M_s(F) = \sigma(\text{Conv}^{7\times7}([\text{AvgPool}(F); \text{MaxPool}(F)])) $$
        *Intuition*: "Where is the object located in the feature map?"
    3.  **Refinement**: $F' = F \odot M_c(F) \odot M_s(F \odot M_c(F))$.

### TokenLearner
*   **Switch**: `BackboneConfig(token_learner_tokens=N)`
*   **Problem**: ViTs differ from CNNs in that compute scales quadratically with tokens $O(N^2)$.
*   **Solution**: Learn $K \ll N$ "summary tokens" that represent the entire image.
*   **Math**:
    $$ \alpha_i = \text{Softmax}(MLP(X_i)) \in \mathbb{R}^{N \times K} $$
    $$ Z = \alpha^T X \in \mathbb{R}^{K \times C} $$
*   **Impact**: Reduces generic $14\times14=196$ tokens to just $K=8$ or $16$ semantic clusters, enabling massive throughput gains.

## 4. Adaptation & Efficiency
### LoRA (Low-Rank Adaptation)
*   **Switch**: `BackboneConfig(lora_rank=r)`
*   **Theory**: Updates to weight matrices have a low "intrinsic rank".
*   **Math**: $\Delta W = \frac{\alpha}{r} B A$, where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$.
*   **Mechanism**:
    *   Freeze $W_{frozen}$.
    *   Train only $A$ and $B$.
    *   Forward: $h = W_{frozen}x + \frac{\alpha}{r}BAx$.
*   **Benefit**: Fine-tune a 600M parameter model by updating only 2M parameters.

### Token Merging (ToMe)
*   **Switch**: `BackboneConfig(token_merging_ratio=r)`
*   **Algorithm**: Bipartite Soft Matching.
    1.  Partition tokens into two sets $A, B$.
    2.  Draw edges between most similar tokens.
    3.  Merge $k$ edges by averaging features.
*   **Benefit**: Speeds up inference by reducing token count layer-by-layer without retraining.

---

# Part II: Training Dynamics & Loss Functions
**Files:** `losses.py`, `train_*.py`

## 1. Metric Learning: AdaFace
*   **Class**: `AdaFace` (Adaptive Margin ArcFace)
*   **Goal**: Maximize intra-class compactness and inter-class separability on the hypersphere.
*   **The Math**:
    $$ L = -\log \frac{e^{s(\cos(\theta_{y}) - m)}}{e^{s(\cos(\theta_{y}) - m)} + \sum_{j \neq y} e^{s \cos(\theta_{j})}} $$
*   **The "Ada" Twist**: The margin $m$ is not fixed. It adapts to image quality (Norm $|z|$).
    $$ m(\hat{x}) \propto \text{Norm}(\hat{x}) $$
    *   **High Norm (Clear Image)** $\rightarrow$ High Margin (Harder constraint).
    *   **Low Norm (Blurry/Occluded)** $\rightarrow$ Low Margin (Softer constraint).
*   **Why**: Prevents the model from trying to force "impossible" noisy samples into tight clusters, which causes gradient explosion and overfitting.

## 2. Uncertainty: Evidential Deep Learning (EDL)
*   **Class**: `EvidentialLoss`
*   **Theory**: Subjective Logic.
    *   Standard Classifier: Outputs a probability point estimate $P(y|x)$.
    *   EDL Classifier: Outputs the parameters $\alpha$ of a Dirichlet distribution $D(p|\alpha)$.
*   **Math**:
    $$ \alpha_k = e^{z_k} + 1 $$
    $$ S = \sum \alpha_k $$
    $$ \text{Uncertainty } u = \frac{K}{S} $$
*   **Impact**: The model yields a measure of **Epistemic Uncertainty** ("I have never seen this before") vs **Aleatoric Uncertainty** ("This image is too blurry").

## 3. Pre-training: Supervised Contrastive (SupCon)
*   **Class**: `SupConLoss`
*   **Theory**: Contrastive Learning generalized to arbitrary numbers of positives per anchor.
*   **Mechanism**:
    1.  Create 2 augmented views of every image in batch.
    2.  Pull together all views of the *same class*.
    3.  Push apart all views of *different classes*.
*   **Impact**: Pre-aligns the feature space so that the final classifier learning (Phase 2) is basically just drawing linear boundaries around already-formed clusters.

---

# Part III: Optimization Engineering
**Files:** `optimizers/*.py`

## 1. Sharpness-Aware Minimization (SAM)
*   **Philosophy**: "Flat minima generalize better than sharp minima."
*   **Algorithm**:
    1.  **Ascent**: Compute $\nabla w$. Move $w \rightarrow w + \epsilon$ to the point of *highest* loss in the immediate neighborhood.
        $$ \epsilon = \rho \frac{\nabla w}{||\nabla w||} $$
    2.  **Descent**: Compute gradient at that worst-case point $\nabla w|_{w+\epsilon}$.
    3.  **Update**: Move original $w$ using that gradient.
*   **Result**: The model doesn't just find a low loss; it finds a "valley" of low loss robust to weight perturbations.

## 2. Lookahead Optimizer
*   **Philosophy**: "Fast Exploration, Slow Synchronization."
*   **Algorithm**:
    *   **Fast Weights** ($\theta$): Updated by standard AdamW/SGD for $k=5$ steps. They explore strictly stochastic paths.
    *   **Slow Weights** ($\phi$): Updated once every $k$ steps.
        $$ \phi \leftarrow \phi + \alpha(\theta - \phi) $$
*   **Benefit**: The slow weights act as a stable "anchor," reducing variance and allowing the fast weights to explore without diverging.

## 3. Gradient Centralization (GC)
*   **Math**:
    $$ \nabla w_{GC} = \nabla w - \text{mean}(\nabla w) $$
*   **Intuition**: Constrains the optimization to occur on a hyperplane perpendicular to the unit vector.
*   **Impact**: Acts as a uniquely powerful regularizer (smoother than Weight Decay) and allows for higher learning rates.

---

# Part IV: Configuration & Hyperparameters
**File:** `configs/*.yaml`

## 1. Backbone Configs
*   `fusion_dim`: (Default: 1024). The info-bottleneck size. Lower = more compression/generalization; Higher = more capacity.
*   `drop_path_rate`: (Default: `0.0`, `0.2` for ViT). Probability of dropping a residual block. Acts as "Depth Regularization."

## 2. ArcFace Configs
*   `margin` ($m$): (Default: 0.5). Stepsize of angular separation. 0.5 radians $\approx$ 28 degrees.
*   `scale` ($s$): (Default: 64.0). Temperature scaling of the softmax. Essential for convergence on Hypersphere manifolds.
*   `rho` ($\rho$): (Default: 0.05). Neighborhood size for SAM. Larger = Flatter (but harder to train).

## 3. Distillation Configs
*   `distill_weight`: (Default: 0.3). Balance between "Imitate Teacher" (Soft target) and "Solve Task" (Hard target).
*   `temperature`: (Default: 2.0). Softens the teacher's probability distribution to reveal dark knowledge (inter-class relationships).

---

# Mastery Checklist
To claim total mastery, you must understand:
1.  [ ] How **DHA** re-weights the CNN vs ViT streams dynamically.
2.  [ ] How **AdaFace** uses batch statistics to estimate image quality.
3.  [ ] The min-max game played by **SAM** in the loss landscape.
4.  [ ] The rank-decomposition math of **LoRA**.
5.  [ ] The difference between **Epistemic** and **Aleatoric** uncertainty in EDL.

This codebase is a laboratory of modern SOTA distinctions. Every line of code exists to solve a specific theoretical failure mode of standard Deep Learning.
