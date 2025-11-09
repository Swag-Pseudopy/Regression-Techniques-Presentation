# **Regression Techniques Demonstration: PCR, LASSO, Group LASSO, and Fused LASSO**

### **Author**

Supratim Das, Swagato Das  
**Roll No(s):** MB2532, MB2534  
**M.Stat. I**, Indian Statistical Institute, Kolkata  
**Course:** Regression Techniques — Presentation Project (2025)


---

## **Overview**

This project presents a unified R implementation of four major regularization and regression methods—**Principal Component Regression (PCR)**, **LASSO**, **Group LASSO**, and **Fused LASSO**—demonstrated on simulated data.
The work aims to illustrate the common theme of **regularization through penalization**, showing how different penalties influence the structure and smoothness of estimated coefficients.

All implementations use reliable R packages (`pls`, `glmnet`, `gglasso`, `genlasso`) with concise, reproducible examples suitable for presentation and demonstration.

---

## **Methods**

### **1. Principal Component Regression (PCR)**

Performs regression after projecting predictors onto a reduced set of orthogonal principal components.
Implemented using `pls::pcr()` with cross-validation to select the optimal number of components.

### **2. LASSO Regression**

Applies **L1 regularization**, which drives some coefficients exactly to zero, yielding sparse solutions and automatic variable selection.
Implemented via `glmnet::cv.glmnet()`.

### **3. Group LASSO**

Extends LASSO to grouped predictors, applying the penalty at the group level rather than to individual variables.
Encourages inclusion or exclusion of entire feature groups.
Implemented using `gglasso::cv.gglasso()`.

### **4. Fused LASSO**

Penalizes both coefficient magnitudes and their pairwise differences, promoting **piecewise constant solutions**.
Implemented using `genlasso::fusedlasso()` on a simulated noisy image, with an option to import real grayscale images.
This formulation connects directly to **Total Variation (TV) regularization**.

---

## **What is Total Variation–Style Smoothing?**

**Total Variation (TV) smoothing** is a regularization technique commonly used in image processing and spatial regression.
Instead of penalizing large coefficients (as in LASSO), it penalizes **large differences between neighboring coefficients**—in 1D between consecutive data points, and in 2D between adjacent pixels.

Mathematically, for an image ( x ), TV regularization minimizes:

$$\min_{x}  \frac{1}{2} \|y - x\|_2^2 + \lambda \sum_{i,j} \Big( |x_{i,j} - x_{i+1,j}| + |x_{i,j} - x_{i,j+1}| \Big)$$

This encourages **spatial smoothness** while **preserving sharp edges**, unlike classical Gaussian smoothing which blurs boundaries.
In essence:

* Flat regions (constant areas) are smoothed.
* Edges (sudden changes) are preserved.

This property makes **Fused LASSO** a discrete analogue of TV denoising—ideal for recovering structured signals or images corrupted by noise.

---

## **Dependencies**

```r
install.packages(c("glmnet", "gglasso", "genlasso", "pls", "Matrix", "imager"))
```

---

## **Usage**

```r
source("PCR_LASSO_GroupFused_R.R")

example_pcr()          # Principal Component Regression
example_lasso()        # LASSO Regression
example_group_lasso()  # Group LASSO
example_fused_lasso(simulate = TRUE)  # Fused LASSO (image denoising)
```

To test on a real image:

```r
example_fused_lasso(simulate = FALSE, img_path = "image.jpg", nr = 128, nc = 128)
```

---

## **Educational Value**

This implementation provides a concise comparative view of:

* **Dimensionality reduction (PCR)**
* **Sparsity (LASSO)**
* **Structured sparsity (Group LASSO)**
* **Spatial smoothness (Fused LASSO / Total Variation)**

Together, these illustrate the evolution of regression from coefficient shrinkage to spatially aware regularization, integrating statistical, geometric, and computational perspectives.

---

## **References**

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
2. Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*.
3. Zou, H., & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net*.
4. Yuan, M., & Lin, Y. (2006). *Model Selection and Estimation in Regression with Grouped Variables*.
5. Tibshirani, R. et al. (2011). *Solution Paths for Generalized LASSO Problems*.
6. Arnold, T. B. et al. (2011). *Efficient Implementations of the Generalized Lasso Dual Path Algorithm*.
