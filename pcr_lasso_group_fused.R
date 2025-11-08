# PCR_LASSO_GroupFused_R.R
# -------------------------------------------------------------
# Title: PCR, LASSO (glmnet), Group LASSO (gglasso), and 2D Fused LASSO (genlasso)
# Author: Generated for user — revised to use established packages where appropriate
# Date: 2025-11-08
# Purpose: Self-contained R script demonstrating PCR, LASSO, Group LASSO,
#          and 2D fused-lasso (image denoising) on simulated data.
#          Uses widely used CRAN packages: glmnet, gglasso, genlasso, pls, imager.
# -------------------------------------------------------------
# Usage:
# - Run the whole script in an R environment (R kernel in Colab, RStudio, etc.)
# - Install required packages if missing (see section below).
# - Follow the examples at the bottom to test each method.
#
# Notes:
# - LASSO & cross-validated lambda use glmnet::cv.glmnet.
# - Group LASSO uses gglasso::gglasso (block penalty by groups).
# - PCR uses pls::pcr with built-in cross-validation.
# - Fused LASSO (2D image denoising) uses genlasso::fusedlasso with a 2D
#   finite-difference operator assembled as a sparse matrix.
# - This script focuses on clarity and reproducibility for use in Colab.
#
# ------------------------------------------------------------------
# Dependencies (install if missing):
# install.packages(c('glmnet','gglasso','genlasso','pls','Matrix','imager'))
# ------------------------------------------------------------------

# ---------------------- Utilities ---------------------------------
mse <- function(y, yhat) mean((y - yhat)^2)

# ------------------- Simulate data --------------------------------
simulate_regression <- function(n = 200, p = 50, s = 5, sigma = 1, seed = 1) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- rep(0, p)
  beta[1:s] <- seq(from = 2, length.out = s, by = -0.3)
  y <- X %*% beta + rnorm(n, sd = sigma)
  list(X = X, y = as.numeric(y), beta = beta)
}

# -------------------- Principal Component Regression --------------
# Uses pls::pcr which has internal cross-validation
pcr_example <- function(X, y, ncomp = 10, validation = "CV") {
  if (!requireNamespace('pls', quietly = TRUE)) stop('Install package pls')
  library(pls)
  # center & scale performed inside pcr if scale = TRUE
  fit <- pcr(y ~ X, ncomp = ncomp, data = as.data.frame(X), scale = TRUE, validation = validation)
  # choose ncomp by CV
  cv <- RMSEP(fit)
  # pick minimum CV (ignoring "adj" components if present)
  ncomp_opt <- which.min(cv$val[1,1, ]) - 1 # RMSEP returns with 0 comps at index 1
  if (ncomp_opt < 1) ncomp_opt <- 1
  list(fit = fit, ncomp_opt = ncomp_opt)
}

pcr_predict <- function(pcr_obj, Xnew) {
  predict(pcr_obj$fit, newdata = as.data.frame(Xnew), ncomp = pcr_obj$ncomp_opt)
}

# -------------------- LASSO via glmnet -----------------------------
# Uses glmnet and cv.glmnet to select lambda
lasso_glmnet <- function(X, y, nfolds = 5, alpha = 1) {
  if (!requireNamespace('glmnet', quietly = TRUE)) stop('Install package glmnet')
  library(glmnet)
  # glmnet expects x as matrix, y as vector
  cvfit <- cv.glmnet(x = X, y = y, alpha = alpha, nfolds = nfolds, standardize = TRUE)
  lambda_min <- cvfit$lambda.min
  coef_vec <- as.numeric(coef(cvfit, s = "lambda.min")) # includes intercept at index 1
  intercept <- coef_vec[1]
  beta <- coef_vec[-1]
  list(cvfit = cvfit, lambda_min = lambda_min, intercept = intercept, beta = beta)
}

predict_lasso_glmnet <- function(fit, Xnew) {
  predict(fit$cvfit, newx = Xnew, s = "lambda.min")
}

# -------------------- Group LASSO via gglasso ----------------------
# groups: integer vector of length p assigning each variable to a group
group_lasso_gglasso <- function(X, y, groups, nfolds = 5) {
  if (!requireNamespace('gglasso', quietly = TRUE)) stop('Install package gglasso')
  library(gglasso)
  stopifnot(length(groups) == ncol(X))
  # gglasso wants X, y, group, loss = "ls"
  cvfit <- cv.gglasso(x = X, y = y, group = groups, loss = "ls", nfolds = nfolds)
  lambda_min <- cvfit$lambda.min
  beta_full <- coef(cvfit, s = "lambda.min")
  intercept <- as.numeric(beta_full[1])
  beta <- as.numeric(beta_full[-1])
  list(cvfit = cvfit, lambda_min = lambda_min, intercept = intercept, beta = beta)
}

predict_group_gglasso <- function(fit, Xnew) {
  predict(fit$cvfit, newx = Xnew, s = "lambda.min")
}

# -------------------- Fused LASSO on 2D image using genlasso -------
# We will use genlasso::fusedlasso with a difference matrix D constructed for 2D.
# This solves: minimize 0.5||y - x||_2^2 + lambda ||D x||_1

build_diff_ops_sparse <- function(nr, nc) {
  if (!requireNamespace('Matrix', quietly = TRUE)) stop('Install package Matrix')
  library(Matrix)
  N <- nr * nc
  # horizontal differences
  rows_h <- c(); cols_h <- c(); vals_h <- c(); rcount <- 1
  for (i in 1:nr) {
    for (j in 1:(nc - 1)) {
      idx <- (j - 1) * nr + i
      idx2 <- j * nr + i
      rows_h <- c(rows_h, rcount, rcount)
      cols_h <- c(cols_h, idx, idx2)
      vals_h <- c(vals_h, -1, 1)
      rcount <- rcount + 1
    }
  }
  Dh <- sparseMatrix(i = rows_h, j = cols_h, x = vals_h, dims = c(nr * (nc - 1), N))
  # vertical differences
  rows_v <- c(); cols_v <- c(); vals_v <- c(); rcount <- 1
  for (j in 1:nc) {
    for (i in 1:(nr - 1)) {
      idx <- (j - 1) * nr + i
      idx2 <- (j - 1) * nr + i + 1
      rows_v <- c(rows_v, rcount, rcount)
      cols_v <- c(cols_v, idx, idx2)
      vals_v <- c(vals_v, -1, 1)
      rcount <- rcount + 1
    }
  }
  Dv <- sparseMatrix(i = rows_v, j = cols_v, x = vals_v, dims = c((nr - 1) * nc, N))
  D <- rbind(Dh, Dv)
  D
}

fused_lasso_genlasso <- function(y_image, lambda = NULL, nlambda = 50) {
  if (!requireNamespace('genlasso', quietly = TRUE)) stop('Install package genlasso')
  library(genlasso)
  nr <- nrow(y_image); nc <- ncol(y_image); N <- nr * nc
  y_vec <- as.numeric(y_image)
  D <- build_diff_ops_sparse(nr, nc)
  # genlasso::fusedlasso accepts y and D (and optionally X)
  # We will compute solution path and let user pick lambda or pick by heuristic
  fit <- genlasso::fusedlasso(y = y_vec, D = D, minlam = 0)
  # fusedlasso returns path; if lambda provided, interpolate; else pick moderate index
  if (!is.null(lambda)) {
    # find index with closest lambda
    idx <- which.min(abs(fit$lambda - lambda))
  } else {
    # pick median lambda index (heuristic)
    idx <- ceiling(length(fit$lambda) / 3)
  }
  x_hat <- fit$beta[, idx]
  x_img <- matrix(x_hat, nrow = nr, ncol = nc)
  list(fit = fit, idx = idx, x = x_hat, x_img = x_img, lambda_used = fit$lambda[idx])
}

# -------------------- Image import helper -------------------------
import_image_gray <- function(path, target_dim = NULL) {
  if (!requireNamespace('imager', quietly = TRUE)) stop('Please install package imager')
  library(imager)
  img <- load.image(path)
  imgg <- grayscale(img)
  # resize if requested
  if (!is.null(target_dim)) imgg <- imager::imresize(imgg, size = target_dim)
  mat <- t(as.matrix(imgg))
  mat
}

# -------------------- Examples & Demonstration --------------------
# Example 1: PCR on simulated regression
example_pcr <- function() {
  dat <- simulate_regression(n = 200, p = 50, s = 5, sigma = 1, seed = 123)
  res <- pcr_example(dat$X, dat$y, ncomp = 15)
  yhat <- pcr_predict(res, dat$X)
  cat('PCR chosen ncomp:', res$ncomp_opt, '
')
  cat('PCR MSE:', mse(dat$y, yhat), '
')
  invisible(list(dat = dat, res = res, yhat = yhat))
}

# Example 2: LASSO via glmnet
example_lasso <- function() {
  dat <- simulate_regression(n = 200, p = 50, s = 5, sigma = 1, seed = 42)
  res <- lasso_glmnet(dat$X, dat$y, nfolds = 5)
  yhat <- as.numeric(predict_lasso_glmnet(res, dat$X))
  cat('glmnet lambda.min:', res$lambda_min, '
')
  cat('LASSO MSE:', mse(dat$y, yhat), '
')
  cat('Nonzero coefficients:', sum(abs(res$beta) > 1e-8), '
')
  invisible(list(dat = dat, res = res, yhat = yhat))
}

# Example 3: Group LASSO via gglasso
example_group_lasso <- function() {
  dat <- simulate_regression(n = 200, p = 60, s = 6, sigma = 1, seed = 7)
  groups <- rep(1:(60/3), each = 3)
  res <- group_lasso_gglasso(dat$X, dat$y, groups = groups, nfolds = 5)
  yhat <- as.numeric(predict_group_gglasso(res, dat$X))
  cat('gglasso lambda.min:', res$lambda_min, '
')
  cat('Group LASSO MSE:', mse(dat$y, yhat), '
')
  cat('Nonzero groups:', length(unique(groups)[sapply(unique(groups), function(g) any(abs(res$beta[groups==g])>1e-8))]), '
')
  invisible(list(dat = dat, res = res, yhat = yhat))
}

# Example 4: Fused LASSO on simulated image + optional import
example_fused_lasso <- function(simulate = TRUE, img_path = NULL, nr = 64, nc = 64, lambda = NULL) {
  if (simulate) {
    # synthetic piecewise constant image + noise
    img <- matrix(0, nr, nc)
    img[16:40, 16:40] <- 1
    img[10:20, 45:60] <- 0.7
    noisy <- img + matrix(rnorm(nr * nc, sd = 0.2), nr, nc)
    y <- noisy
  } else {
    if (is.null(img_path)) stop('Provide img_path when simulate = FALSE')
    y <- import_image_gray(img_path, target_dim = c(nr, nc))
  }
  res <- fused_lasso_genlasso(y, lambda = lambda)
  # display if imager available
  if (requireNamespace('imager', quietly = TRUE)) {
    library(imager)
    plot(as.cimg(t(y)), main = 'Noisy image (input)')
    plot(as.cimg(t(res$x_img)), main = sprintf('Fused LASSO denoised (lambda=%.4g)', res$lambda_used))
  }
  invisible(list(y = y, denoised = res$x_img, res = res))
}

# -------------------- Run examples in Colab ------------------------
# Example usage in Colab (R kernel):
# source('PCR_LASSO_GroupFused_R.R')
example_pcr()
example_lasso()
example_group_lasso()
example_fused_lasso(simulate = TRUE)
# To denoise your own image (after uploading or mounting Drive):
# example_fused_lasso(simulate = FALSE, img_path = 'your_image.jpg', nr = 128, nc = 128)

# Notes & suggestions:
# - glmnet and gglasso perform standardization internally but you can pre-scale if desired.
# - genlasso may produce many lambda values; choose the index or lambda based on visual inspection
#   or by cross-validated heuristics (not shown here due to complexity of CV for fused lasso).
# - For very large images, building the 2D D matrix can be memory intensive; consider processing
#   patches or using specialized TV denoising libraries.

# End of file
