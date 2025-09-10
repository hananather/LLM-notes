**Core Idea:**
- You have a large set of unlabeled covariates X, but only a small sample of observed outcomes Y
- Train a predictive model $f(X)$ on labeled data, then use it to estimate a parameter (e.g., mean, proportion) over the entire population using unlabeled data.
- Valid inference is achieved by _correcting_ the bias of the predictor via a labeled test set.

Prediction Powered inference allows us to "impute" outcomes via ML to boost effective sample size, then use those labeled data to "rectify" those imputations so that all the resulting 

**Why can’t we simply plug in ML-imputed labels into a standard confidence‐interval procedure and be done?**
1. **Bias & Invalid Coverage**: When you replace the true $Y_i$ with $f(X_i)$, your estimator $\tilde\theta_f$ will typically be **biased** (unless f is perfect), and as a result your CI will **under- or over-cover** the true parameter $\theta^*$
2. Prediction-Powered Inference (PPI) fixes this by introducing a **rectifier** $\Delta$ that captures the systematic error from using $f(X)$ in place of Y.  Concretely, for the simple case of estimating a mean, one defines$$
    \Delta ;=; \mathbb{E}\bigl[Y - f(X)\bigr],$$and then uses the labeled data to build a **confidence set** $R$ for $\Delta$.  Finally, one **“rectifies”** the naive estimator $\tilde\theta_f$ by each candidate in $R$ to obtain a valid CI for $\theta^*$
 Only **after** you ensure **validity** via the rectifier, without it, you’d simply be averaging biased predictions under the false pretence they’re “real” data.

---

In prediction-powered inference (PPI), the “rectifier set” $R$ is **not** the single-point estimate $\hat\Delta$ but rather a **confidence set** containing all plausible values of the rectifier $\Delta$ given your labeled data.  Because $\Delta$ is itself estimated with sampling noise, we construct $R$ (typically an interval) to capture that uncertainty.  In the mean‐estimation warm‐up, $\Delta = \mathbb{E}[\,f(X)-Y\,]$ is estimated by the empirical average $\hat\Delta = \frac1n\sum_i\bigl(f(X_i)-Y_i\bigr)$; the rectifier set is then
$$
R = \bigl[\hat\Delta - z_{1-\delta/2}\,\mathrm{SE}(\hat\Delta),\;\hat\Delta + z_{1-\delta/2}\,\mathrm{SE}(\hat\Delta)\bigr],
$$
an **infinite** continuum of values rather than a single number
# Conformal Predition
**Core Idea:**
- CP wraps around _any_ predictive model (black-box or otherwise)
- Given a trained model and a calibration set, it outputs _set-valued predictions_ that contain the true label (or value) with probability $≥ 1 − α$.
- It gives **finite-sample, distribution-free guarantees**.
**Ideas:**
**Quality Assurance in Imputation:** Use CP to generate _interval estimates_ when imputing missing values (e.g., in income, education level).
**Outlier Detection**: Conformal methods (like Conformal Anomaly Detection) could flag units whose predicted behavior doesn’t match the calibration distribution.

- CP works by computing nonconformity scores on previous unlabled data, and using these to create prediction sets on a new (unlabled) test data
- requires a user specified significance level for which the algorithm should produce its predictions (predicton sets)
    - this significance level restricts the frequency of errors that the algorithm is allowed to make;
    - for example, significance of 0.1 means that the algorithm can make at more 10% erroneous predictons 
- To meet this requirement, the output is a predictions set, instead of a point prediction produced by standard supervised models
    - for classification, this means that predictions are not a single class, for example `cat`, but instead, a set `{‘cat’, ‘dog'}`
- depending on how good this underlying model is (how well it can discen between classes) and the specified significance level, these sets can be smaller or larger. 
- for regression tasks, the output is prediction intervals
 --- 
 C is a _set-valued_ function that maps each test point $X_{\text{test}}$ to a subset of the K possible labels.  Concretely, after calibration you compute
$$C(X_{\text{test}}) \;=\; \{\,y\in\{1,\dots,K\}: \hat f(X_{\text{test}})_y \;\ge\;1 - \hat q\},$$
Let:
- $\hat f(x)_y$ be your model’s softmax “probability” for class $y$.
- $\alpha\in(0,1)$ be your target miscoverage level (e.g. $\alpha=0.1$ for 90 % coverage)
- A calibration set $\{(X_i,Y_i)\}_{i=1}^n$
To construct $C$ from $\hat f$ and the calibration data:
1. Nonconformity Scores
$$s_i \;=\; 1 - \hat f(X_i)_{Y_i} \quad\text{for }i=1,\dots,n$$
A large $s_i$ means the model gave low probability to the true label.  
2. **Threshold via empirical quantile.**
Order the $s_i’s$ from smallest to largest and let
$\hat q_{1-\alpha} \;=\; \text{the }\lceil (n+1)(1-\alpha)\rceil\text{-th smallest }s_i.$
Equivalently, $\hat q_{1-\alpha}$ is the smallest number such that at least $(1-\alpha)\times100\%$ of the calibration scores satisfy $s_i \le \hat q_{1-\alpha}$.
3. **Prediction sets.**
For a
4. we set the conformal scores si = 1-f(Xi) to be (one minus the softmax output of the true class). The score is high when the model is baldy wrong.
5. Define $\hat q$ to be the empirical quantile 
6. For a new test data set create a prediction set such that:
$$C(X_{\text{test}}) \;=\; \{\,y\in\{1,\dots,K\}: \hat f(X_{\text{test}})_y \;\ge\;1 - \hat q\},$$
	that includes all classes with a high enough softmax output. 

--- 
## **Cross-Prediction-Powered Inference (CrossPPI)**
### **Core Idea**
CrossPPI addresses the overfitting and bias that arise when the **same** data are used to train $f$ and to estimate the rectifier.  It partitions the labeled dataset into K folds: for each fold, the model is trained on the other K-1 folds and used to predict on the held-out fold; this “cross-prediction” mimics an independent test set and avoids overly optimistic bias estimates  .
### **Debiasing Mechanism**

1. **Fold-wise rectifiers** $\hat\Delta^{(k)}$ are computed on each held-out fold k.
2. These are pooled to form an overall confidence set R for the bias $\Delta$.
3. The final **prediction-powered set**
    
$$C_{PP} \;=\;\bigl\{\;\tilde\theta_f + d : d\in R\bigr\}$$
    
    retains exact coverage while leveraging all labeled data without reuse bias  .
CrossPPI requires only that predictions be measurable functions of the training folds—**no consistency** or **i.i.d. guarantees** on f are needed for validity.  Empirically, it often tightens intervals compared to classical PPI when the model generalizes well across folds


