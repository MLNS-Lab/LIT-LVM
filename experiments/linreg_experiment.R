#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(hierNet)
  library(data.table)
  library(farff)
  library(jsonlite)
  library(tools)
})

args <- commandArgs(trailingOnly = TRUE)
argval <- function(name, default = NULL) {
  hit <- grepl(paste0("^--", name, "="), args)
  if (!any(hit)) return(default)
  sub(paste0("^--", name, "="), "", args[which(hit)[1]])
}
dataset_name <- argval("dataset", default = NULL)
out_dir      <- argval("out_dir", default = "results")
n_runs       <- as.integer(argval("runs", default = "5"))
train_frac   <- as.numeric(argval("train_frac", default = "0.5"))
val_frac     <- as.numeric(argval("val_frac", default = "0.1"))

if (is.null(dataset_name)) {
  stop("Usage: Rscript linreg_experiment.R --dataset=boston.arff [--out_dir=results --runs=5 --train_frac=0.5 --val_frac=0.1]")
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
base_noext <- file_path_sans_ext(dataset_name)
results_csv <- file.path(out_dir, sprintf("%s_hiernet_results_%s.csv", base_noext, stamp))
summary_json <- file.path(out_dir, sprintf("%s_hiernet_summary_%s.json", base_noext, stamp))

to_dt <- function(x) as.data.table(x)
rmse_score <- function(y_true, y_pred) {
  if (is.list(y_pred)) {
    y_pred <- unlist(y_pred)
  }
  y_pred <- as.numeric(y_pred)
  y_true <- as.numeric(y_true)
  
  if (length(y_pred) != length(y_true)) {
    stop(sprintf("Length mismatch: y_pred (%d) vs y_true (%d)", 
                 length(y_pred), length(y_true)))
  }
  
  if (any(is.na(y_pred)) || any(is.na(y_true))) {
    warning("NA values detected in RMSE calculation")
    return(Inf)
  }
  
  sqrt(mean((y_true - y_pred)^2))
}

make_splits <- function(n, seed, train_frac, val_frac) {
  set.seed(seed)
  idx <- seq_len(n)
  n_train <- floor(train_frac * n)
  n_val   <- floor(val_frac   * n)
  train_idx <- sample(idx, size = n_train, replace = FALSE)
  remaining <- setdiff(idx, train_idx)
  set.seed(seed)
  val_idx <- sample(remaining, size = n_val, replace = FALSE)
  test_idx <- setdiff(remaining, val_idx)
  list(train = train_idx, val = val_idx, test = test_idx)
}

one_hot <- function(df) {
  mm <- model.matrix(~ . - 1, data = df)
  mm <- as.matrix(mm)
  storage.mode(mm) <- "double"
  mm
}

load_regression <- function(dataset_name) {
  file_path <- file.path("/home/mxn447/Datasets/regression", dataset_name)
  if (!file.exists(file_path)) stop(sprintf("ARFF file not found: %s", file_path))
  cat(sprintf("[INFO] Loading ARFF: %s\n", file_path))
  df <- to_dt(readARFF(file_path))

  choose_target_and_X <- function(df, target_name, drops = NULL) {
    if (!is.null(drops)) df <- df[, !..drops]
    y <- df[[target_name]]
    X <- df[, !..target_name]
    list(X = X, y = y)
  }

  X <- NULL; y <- NULL

  if (dataset_name == "boston.arff") {
    lst <- choose_target_and_X(df, "MEDV")
  } else if (dataset_name == "space_ga.arff") {
    lst <- choose_target_and_X(df, "ln(VOTES/POP)")
  } else if (dataset_name == "socmob.arff") {
    lst <- choose_target_and_X(df, "counts_for_sons_current_occupation")
  } else if (dataset_name == "diamonds.arff") {
    lst <- choose_target_and_X(df, "price")
  } else if (dataset_name == "bike_sharing.arff") {
    lst <- choose_target_and_X(df, "count")
  } else if (dataset_name == "medical_charges.arff") {
    lst <- choose_target_and_X(df, "AverageTotalPayments")
  } else if (dataset_name == "superconduct.arff") {
    lst <- choose_target_and_X(df, "criticaltemp")
  } else if (dataset_name == "wind.arff") {
    lst <- choose_target_and_X(df, "MAL")
  } else if (dataset_name == "fri_c4_500_10.arff") {
    lst <- choose_target_and_X(df, "oz11")
  } else {
    y <- df[[ncol(df)]]
    X <- df[, 1:(ncol(df)-1)]
    lst <- list(X = X, y = y)
  }

  X <- to_dt(lst$X)
  y <- as.numeric(lst$y)

  for (nm in names(X)) {
    if (is.factor(X[[nm]])) X[[nm]] <- as.character(X[[nm]])
  }

  X_oh <- one_hot(X)

  list(X = X_oh, y = y)
}

cat(sprintf("[INFO] Dataset Name: %s\n", dataset_name))
dl <- load_regression(dataset_name)
X_full <- dl$X
y      <- dl$y
n <- nrow(X_full)
p <- ncol(X_full)
cat(sprintf("[INFO] n=%d, p=%d\n", n, p))

hn_strong   <- FALSE
hn_diagonal <- FALSE
hn_center   <- FALSE
hn_stand_m  <- TRUE
hn_stand_i  <- TRUE
lamlist     <- c(0.01, 0.1, 1, 10, 100)
hn_rho      <- NULL
hn_niter    <- 100
hn_sym_eps  <- 1e-3
hn_tol      <- 1e-5
hn_trace    <- 0

per_run <- vector("list", n_runs)
cat(sprintf("[INFO] Running %d repeats with seeds 42..%d\n", n_runs, 42 + n_runs - 1))

for (i in seq_len(n_runs)) {
  seed <- 42 + (i - 1)
  cat(sprintf("\n[RUN %d/%d] seed=%d\n", i, n_runs, seed))
  sp <- make_splits(n, seed, train_frac, val_frac)
  tr_idx <- sp$train; va_idx <- sp$val; te_idx <- sp$test

  X_tr <- X_full[tr_idx, , drop = FALSE]
  y_tr <- y[tr_idx]
  X_va <- X_full[va_idx, , drop = FALSE]
  y_va <- y[va_idx]
  X_te <- X_full[te_idx, , drop = FALSE]
  y_te <- y[te_idx]

  fit_path <- hierNet.path(
    x = X_tr, y = y_tr,
    strong   = hn_strong,
    diagonal = hn_diagonal,
    lamlist  = lamlist,
    delta    = 1e-8,
    stand.main = hn_stand_m,
    stand.int  = hn_stand_i,
    rho = nrow(X_tr),
    niter = hn_niter,
    sym.eps = hn_sym_eps,
    step = 1, maxiter = 2000, backtrack = 0.2, tol = hn_tol, trace = hn_trace
  )

  rmses <- numeric(length(lamlist))
  
  cat(sprintf("Evaluating %d lambda values...\n", length(lamlist)))
  
  for (j in seq_along(lamlist)) {
    fit_j <- hierNet(
      x = X_tr, y = y_tr,
      lam = lamlist[j],
      strong = hn_strong,
      diagonal = hn_diagonal,
      delta = 1e-8,
      stand.main = hn_stand_m,
      stand.int = hn_stand_i,
      rho = nrow(X_tr),
      niter = hn_niter,
      sym.eps = hn_sym_eps,
      step = 1, maxiter = 2000, backtrack = 0.2, tol = hn_tol, trace = 0
    )
    
    p_val_j <- predict(fit_j, X_va)
    
    cat(sprintf("Lambda %d: pred length=%d, y_va length=%d\n", j, length(p_val_j), length(y_va)))
    
    if (is.list(p_val_j)) p_val_j <- unlist(p_val_j)
    p_val_j <- as.numeric(p_val_j)
    
    if (length(p_val_j) != length(y_va)) {
      cat(sprintf("Warning: Prediction length mismatch for lambda %g. Expected %d, got %d\n", 
                  lamlist[j], length(y_va), length(p_val_j)))
      if (length(p_val_j) > length(y_va)) {
        p_val_j <- p_val_j[1:length(y_va)]
      } else {
        rmses[j] <- Inf
        next
      }
    }
    
    rmses[j] <- tryCatch({
      rmse_score(y_va, p_val_j)
    }, error = function(e) {
      cat(sprintf("RMSE error for lambda %g: %s\n", lamlist[j], e$message))
      Inf
    })
  }
  
  best_j <- which.min(rmses)
  lam_sel <- lamlist[best_j]
  best_val_rmse <- rmses[best_j]

  X_trva <- rbind(X_tr, X_va)
  y_trva <- c(y_tr, y_va)
  fit_final <- hierNet(
    x = X_trva, y = y_trva,
    lam      = lam_sel,
    strong   = hn_strong,
    diagonal = hn_diagonal,
    delta    = 1e-8,
    stand.main = hn_stand_m,
    stand.int  = hn_stand_i,
    rho = nrow(X_trva),
    niter = hn_niter,
    sym.eps = hn_sym_eps,
    step = 1, maxiter = 2000, backtrack = 0.2, tol = hn_tol, trace = 0
  )

  p_test <- predict(fit_final, X_te)
  
  cat(sprintf("Test: pred length=%d, y_te length=%d\n", length(p_test), length(y_te)))
  
  if (is.list(p_test)) {
    p_test <- unlist(p_test)
  }
  p_test <- as.numeric(p_test)
  
  if (length(p_test) != length(y_te)) {
    cat(sprintf("Warning: Test prediction length mismatch. Expected %d, got %d\n", 
                length(y_te), length(p_test)))
    if (length(p_test) > length(y_te)) {
      p_test <- p_test[1:length(y_te)]
    } else {
      stop(sprintf("Test prediction length (%d) is less than y_te length (%d)", 
                   length(p_test), length(y_te)))
    }
  }
  
  rmse_te <- rmse_score(y_te, p_test)

  main_coefs <- as.numeric(fit_final$bp - fit_final$bn)
  nnz_main   <- sum(main_coefs != 0)
  Theta <- (fit_final$th + t(fit_final$th)) / 2
  nnz_inter <- {
    ut <- upper.tri(Theta, diag = FALSE)
    sum(Theta[ut] != 0)
  }

  per_run[[i]] <- data.table(
    dataset = dataset_name,
    run = i,
    seed = seed,
    n = n,
    n_train = length(tr_idx),
    n_val   = length(va_idx),
    n_test  = length(te_idx),
    lambda  = lam_sel,
    val_rmse = best_val_rmse,
    test_rmse = rmse_te,
    nnz_main = nnz_main,
    nnz_inter = nnz_inter
  )

  cat(sprintf("[RUN %d] val RMSE=%.4f | test RMSE=%.4f | lambda=%.6g | nnz(main)=%d nnz(int)=%d\n",
              i, best_val_rmse, rmse_te, lam_sel, nnz_main, nnz_inter))
}

res_dt <- rbindlist(per_run)
fwrite(res_dt, results_csv)
mean_rmse <- mean(res_dt$test_rmse)
se_rmse   <- sd(res_dt$test_rmse) / sqrt(nrow(res_dt))

params <- list(
  dataset_name = dataset_name,
  n_runs = n_runs,
  seed_base = 42,
  split = list(train_frac = train_frac, val_frac = val_frac,
               test_frac = 1 - train_frac - val_frac),
  hiernet = list(
    family = "gaussian",
    strong = hn_strong,
    diagonal = hn_diagonal,
    center = hn_center,
    stand_main = hn_stand_m,
    stand_int  = hn_stand_i,
    nlam = hn_nlam,
    flmin = hn_flmin,
    niter = hn_niter,
    sym_eps = hn_sym_eps,
    tol = hn_tol
  )
)

summary <- list(
  params = params,
  per_run = res_dt,
  aggregate = list(
    mean_test_rmse = mean_rmse,
    se_test_rmse   = se_rmse
  )
)

summary$aggregate$mean_test_rmse <- mean_rmse
summary$aggregate$se_test_rmse <- se_rmse

writeLines(toJSON(summary, pretty = TRUE, auto_unbox = TRUE), summary_json)

cat("\n==================== SUMMARY ====================\n")
cat(sprintf("Results CSV : %s\n", results_csv))
cat(sprintf("Summary JSON: %s\n", summary_json))
cat(sprintf("TEST RMSE mean=%.4f  SE=%.4f  (n=%d runs)\n", mean_rmse, se_rmse, nrow(res_dt)))
cat("=================================================\n")
