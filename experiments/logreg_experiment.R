#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(hierNet)
  library(data.table)
  library(farff)
  library(pROC)
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
  stop("Usage: Rscript logreg_experiment.R --dataset=tecator.arff [--out_dir=results --runs=5 --train_frac=0.5 --val_frac=0.1]")
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
base_noext <- file_path_sans_ext(dataset_name)
results_csv <- file.path(out_dir, sprintf("%s_hiernet_results_%s.csv", base_noext, stamp))
summary_json <- file.path(out_dir, sprintf("%s_hiernet_summary_%s.json", base_noext, stamp))

to_dt <- function(x) as.data.table(x)
auc_score <- function(y_true, p_hat) {
  if (is.list(p_hat)) {
    p_hat <- unlist(p_hat)
  }
  p_hat <- as.numeric(p_hat)
  y_true <- as.numeric(y_true)
  
  if (length(p_hat) != length(y_true)) {
    stop(sprintf("Length mismatch: p_hat (%d) vs y_true (%d)", 
                 length(p_hat), length(y_true)))
  }
  
  if (any(is.na(p_hat)) || any(is.na(y_true))) {
    warning("NA values detected in AUC calculation")
    return(0.5)
  }
  
  if (length(unique(y_true)) < 2) {
    warning("y_true has no variation (all same class)")
    return(0.5)
  }
  
  as.numeric(pROC::auc(y_true, p_hat, quiet = TRUE))
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

load_classification <- function(dataset_name) {
  if (dataset_name == "mri") {
    file_name <- "/home/mxn447/Datasets/MRI"
    thickness_data <- fread(file.path(file_name, "thickness_data.csv"))
    X <- fread(file.path(file_name, "X.csv"))
    y <- fread(file.path(file_name, "y.csv"))[[1]]
    X <- cbind(X, thickness_data)
    y <- as.integer(y)
    return(list(X = to_dt(X), y = y))
  }

  file_path <- file.path("/home/mxn447/Datasets/classification", dataset_name)
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

  if (dataset_name %in% c("madeline.arff","philippine.arff","jasmine.arff")) {
    lst <- choose_target_and_X(df, "class")
  } else if (dataset_name %in% c("clean1.arff","clean2.arff")) {
    drops <- if (dataset_name == "clean1.arff" || dataset_name == "clean2.arff")
      c("conformation_name","molecule_name") else NULL
    lst <- choose_target_and_X(df, "class", drops)
  } else if (dataset_name %in% c("fri_c4_1000_100.arff","fri_c4_500_100.arff","tecator.arff","pol_class.arff")) {
    lst <- choose_target_and_X(df, "binaryClass")
  } else if (dataset_name == "speech.arff") {
    lst <- choose_target_and_X(df, "Target")
  } else if (dataset_name == "nomao.arff") {
    lst <- choose_target_and_X(df, "Class")
  } else if (dataset_name == "musk.arff") {
    lst <- choose_target_and_X(df, "class", drops = "ID")
  } else if (dataset_name == "scene.arff") {
    lst <- choose_target_and_X(df, "Urban")
  } else if (dataset_name %in% c("hill_valley.arff","hill_valley_noiseless.arff")) {
    lst <- choose_target_and_X(df, "Class")
  } else if (dataset_name == "bioresponse.arff") {
    lst <- choose_target_and_X(df, "target")
  } else if (dataset_name == "eye_movement.arff") {
    lst <- choose_target_and_X(df, "label")
  } else if (dataset_name == "jannis.arff") {
    lst <- choose_target_and_X(df, "class")
  } else if (dataset_name == "miniboone.arff") {
    lst <- choose_target_and_X(df, "signal")
  } else if (dataset_name == "australian.arff") {
    lst <- choose_target_and_X(df, "A15")
  } else if (dataset_name %in% c("autoUniv-au1-1000.arff","climate-model-simulation-crashes.arff")) {
    lst <- choose_target_and_X(df, "Class")
    if (dataset_name == "climate-model-simulation-crashes.arff") {
      lst$X <- lst$X[, !.(V1, V2), with = FALSE]
    }
  } else if (dataset_name == "coil2000.arff") {
    lst <- choose_target_and_X(df, "CARAVAN")
  } else if (dataset_name == "credit-approval.arff") {
    df <- df[complete.cases(df)]
    df[df == "?"] <- NA
    df <- df[complete.cases(df)]
    lst <- choose_target_and_X(df, "class")
  } else if (dataset_name %in% c("credit-g.arff")) {
    lst <- choose_target_and_X(df, "class")
  } else if (dataset_name %in% c("ilpd.arff")) {
    lst <- choose_target_and_X(df, "Class")
  } else if (dataset_name %in% c("kc1.arff")) {
    lst <- choose_target_and_X(df, "defects")
  } else if (dataset_name %in% c("kc2.arff")) {
    lst <- choose_target_and_X(df, "problems")
  } else if (dataset_name %in% c("ozone_level.arff")) {
    lst <- choose_target_and_X(df, "Class")
  } else if (dataset_name %in% c("pc1.arff")) {
    lst <- choose_target_and_X(df, "defects")
  } else if (dataset_name %in% c("pc3.arff","pc4.arff","svmguide3.arff","w4a.arff","spambase.arff")) {
    lst <- choose_target_and_X(df, "class")
  } else if (dataset_name %in% c("qsar-biodeg.arff","steel-plates-fault.arff")) {
    lst <- choose_target_and_X(df, "Class")
  } else if (dataset_name == "satellite.arff") {
    lst <- choose_target_and_X(df, "Target")
  } else {
    y <- df[[ncol(df)]]
    X <- df[, 1:(ncol(df)-1)]
    lst <- list(X = X, y = y)
  }

  X <- to_dt(lst$X)
  y <- lst$y

  for (nm in names(X)) {
    if (is.factor(X[[nm]])) X[[nm]] <- as.character(X[[nm]])
  }

  map_y <- function(y, mapping) {
    y_chr <- if (is.factor(y)) as.character(y) else as.character(y)
    if (!all(y_chr %in% names(mapping))) {
      stop("Label values do not match expected mapping for dataset: ",
           paste(setdiff(unique(y_chr), names(mapping)), collapse = ", "))
    }
    as.integer(unname(mapping[y_chr]))
  }

  if (dataset_name %in% c("pol_class.arff","fri_c4_1000_100.arff","fri_c4_500_100.arff","tecator.arff")) {
    y <- map_y(y, c("P"=1, "N"=0))
  } else if (dataset_name == "speech.arff") {
    y <- map_y(y, c("Anomaly"=1, "Normal"=0))
  } else if (dataset_name == "nomao.arff") {
    y <- map_y(y, c("2"=1, "1"=0))
  } else if (dataset_name == "musk.arff") {
    y <- map_y(y, c("1"=1, "0"=0))
  } else if (dataset_name %in% c("scene.arff","hill_valley.arff","hill_valley_noiseless.arff",
                                 "australian.arff","bioresponse.arff","clean1.arff","clean2.arff",
                                 "jannis.arff","jasmine.arff","madeline.arff","eye_movement.arff",
                                 "spambase.arff")) {
    y <- map_y(y, c("1"=1, "0"=0))
  } else if (dataset_name == "autoUniv-au1-1000.arff") {
    y <- map_y(y, c("class2"=1, "class1"=0))
  } else if (dataset_name == "climate-model-simulation-crashes.arff") {
    y <- map_y(y, c("2"=1, "1"=0))
  } else if (dataset_name == "credit-approval.arff") {
    y <- map_y(y, c("+"=1, "-"=0))
  } else if (dataset_name == "credit-g.arff") {
    y <- map_y(y, c("good"=1, "bad"=0))
  } else if (dataset_name == "ilpd.arff") {
    y <- map_y(y, c("2"=1, "1"=0))
  } else if (dataset_name == "kc1.arff") {
    y <- map_y(y, c("true"=1, "false"=0))
  } else if (dataset_name == "kc2.arff") {
    y <- map_y(y, c("yes"=1, "no"=0))
  } else if (dataset_name == "pc1.arff") {
    y <- map_y(y, c("true"=1, "false"=0))
  } else if (dataset_name == "pc3.arff") {
    y <- map_y(y, c("TRUE"=1, "FALSE"=0))
  } else if (dataset_name == "pc4.arff") {
    y <- map_y(y, c("TRUE"=1, "FALSE"=0))
  } else if (dataset_name == "qsar-biodeg.arff") {
    y <- map_y(y, c("2"=1, "1"=0))
  } else if (dataset_name == "satellite.arff") {
    y <- map_y(y, c("Anomaly"=1, "Normal"=0))
  } else if (dataset_name == "steel-plates-fault.arff") {
    y <- map_y(y, c("2"=1, "1"=0))
  } else if (dataset_name %in% c("madeline.arff","jasmine.arff")) {
    y <- as.integer(as.character(y))
  } else {
    if (is.factor(y)) y <- as.character(y)
    if (is.character(y) && all(unique(y) %in% c("0","1"))) {
      y <- as.integer(y)
    } else if (is.numeric(y) && all(unique(y) %in% c(0,1))) {
      y <- as.integer(y)
    } else {
      stop("Unhandled label mapping for dataset: ", dataset_name)
    }
  }

  X_oh <- one_hot(X)

  list(X = X_oh, y = y)
}

cat(sprintf("[INFO] Dataset Name: %s\n", dataset_name))
dl <- load_classification(dataset_name)
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

fit_path <- hierNet.logistic.path(
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

  aucs <- numeric(length(lamlist))
  
  cat(sprintf("Evaluating %d lambda values...\n", length(lamlist)))
  
  for (j in seq_along(lamlist)) {
    fit_j <- hierNet.logistic(
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
        aucs[j] <- 0.5
        next
      }
    }
    
    aucs[j] <- tryCatch({
      auc_score(y_va, p_val_j)
    }, error = function(e) {
      cat(sprintf("AUC error for lambda %g: %s\n", lamlist[j], e$message))
      0.5
    })
  }
  
  best_j <- which.max(aucs)
  lam_sel <- lamlist[best_j]
  best_val_auc <- aucs[best_j]


  # 3) Refit at selected lambda on TRAIN+VAL
  X_trva <- rbind(X_tr, X_va)
  y_trva <- c(y_tr, y_va)
  fit_final <- hierNet.logistic(
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

  # 4) Test AUC
  p_test <- predict(fit_final, X_te)
  
  # Debug: check test prediction dimensions
  cat(sprintf("Test: pred length=%d, y_te length=%d\n", length(p_test), length(y_te)))
  
  # Ensure p_test is numeric and correct length
  if (is.list(p_test)) {
    p_test <- unlist(p_test)
  }
  p_test <- as.numeric(p_test)
  
  # Handle prediction length mismatch
  if (length(p_test) != length(y_te)) {
    cat(sprintf("Warning: Test prediction length mismatch. Expected %d, got %d\n", 
                length(y_te), length(p_test)))
    # Try to take only the needed predictions (in case it's returning too many)
    if (length(p_test) > length(y_te)) {
      p_test <- p_test[1:length(y_te)]
    } else {
      stop(sprintf("Test prediction length (%d) is less than y_te length (%d)", 
                   length(p_test), length(y_te)))
    }
  }
  
  auc_te <- auc_score(y_te, p_test)

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
    val_auc = best_val_auc,
    test_auc = auc_te,
    nnz_main = nnz_main,
    nnz_inter = nnz_inter
  )

  cat(sprintf("[RUN %d] val AUC=%.4f | test AUC=%.4f | lambda=%.6g | nnz(main)=%d nnz(int)=%d\n",
              i, best_val_auc, auc_te, lam_sel, nnz_main, nnz_inter))
}

res_dt <- rbindlist(per_run)
fwrite(res_dt, results_csv)
mean_auc <- mean(res_dt$test_auc)
se_auc   <- sd(res_dt$test_auc) / sqrt(nrow(res_dt))

params <- list(
  dataset_name = dataset_name,
  n_runs = n_runs,
  seed_base = 42,
  split = list(train_frac = train_frac, val_frac = val_frac,
               test_frac = 1 - train_frac - val_frac),
  hiernet = list(
    family = "logistic",
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
    mean_test_auc = mean_auc,
    se_test_auc   = se_auc
  )
)

summary$aggregate$mean_test_auc <- mean_auc
summary$aggregate$se_test_auc <- se_auc

writeLines(toJSON(summary, pretty = TRUE, auto_unbox = TRUE), summary_json)

cat("\n==================== SUMMARY ====================\n")
cat(sprintf("Results CSV : %s\n", results_csv))
cat(sprintf("Summary JSON: %s\n", summary_json))
cat(sprintf("TEST AUC mean=%.4f  SE=%.4f  (n=%d runs)\n", mean_auc, se_auc, nrow(res_dt)))
cat("=================================================\n")
