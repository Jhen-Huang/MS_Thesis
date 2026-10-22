#########################################################################
# 03_Simulation_Study.R
#
# Purpose:
# Generate simulated datasets under predefined scenarios and evaluate
# prediction performance of MI prediction and sweeping prediction.
#
#########################################################################

library(MASS)      # Multivariate normal data generation
library(mice)      # Multiple imputation by chained equations
library(pROC)      # ROC curve and AUC calculation
library(mipred)    # Prediction with multiple imputation
library(dplyr)     # Data manipulation and summarization
library(tidyr)     # Data reshaping and tidying
library(ggplot2)   # Data visualization

#----------------------------------------------------------------------#
# Global Simulation Settings

# Select one simulation scenario.
# The same workflow can be repeated by changing scenario_run.
scenario_run <- "B2"

# MI prediction settings (prediction averaging)
m_mi <- 5      # Number of imputations
maxit_mi <- 5  # Number of MICE iterations

# Sweeping prediction settings (MI-based starting model)
m_sw <- 20      # Number of imputations
maxit_sw <- 20  # Number of MICE iterations

# Sample sizes
n_calib <- 5110 # Calibration sample size
n_pred <- 10000 # Prediction sample size

#----------------------------------------------------------------------#
## 1. Scenario Design ####
scenario_table <- data.frame(
  scenario_id = c("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"), 
  
  # Outcome model
  beta1 = c(1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7),     # age_Z
  beta2 = c(0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4),     # hypertension
  beta3 = c(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),     # glucose_z
  beta4 = c(0.01, 0.01, 2.0, 2.0, 0.01, 0.01, 2.0, 2.0), # BMI_z
  beta5 = c(0.01, 0.01, 2.0, 2.0, 0.01, 0.01, 2.0, 2.0), # Smoking
  beta6 = c(0, 0, 0, 0, 2.0, 2.0, 2.0, 2.0),             # hidden variable W -> outcome

  # BMI missingness model
  # Missingness depends on observed predictors and W
  alpha1 = c(0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36),         # gender
  alpha2 = c(0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66),         # hypertension
  alpha3 = c(0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70),         # heart_disease
  alpha4 = c(0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18),         # avg_glucose_level_z (0.004 * 45 ≈ 0.18)
  alpha5 = c(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0),                 # W -> BMI missingness
  
  # Smoking missingness model
  # Missingness depends on observed predictors and optionally on W
  gamma1 = c(-0.76, -0.76, -0.76, -0.76, -0.76, -0.76, -0.76, -0.76), # hypertension
  gamma2 = c(-0.24, -0.24, -0.24, -0.24, -0.24, -0.24, -0.24, -0.24), # ever_married
  gamma3 = c(-0.14, -0.14, -0.14, -0.14, -0.14, -0.14, -0.14, -0.14), # work_type_Govt
  gamma4 = c(0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003), # work_type_Self
  gamma5 = c(3.04, 3.04, 3.04, 3.04, 3.04, 3.04, 3.04, 3.04),         # work_type_children
  gamma6 = c(0, 2.0, 0, 2.0, 0, 2.0, 0, 2.0),                         # W -> smoking missingness
  
  target_prev      = rep(0.0487, 8),
  target_obs_bmi   = rep(0.96, 8),
  target_obs_smoke = rep(0.70, 8),
  
  stringsAsFactors = FALSE
)

# Scenario used for single-scenario checks.
sc <- scenario_table[scenario_table$scenario_id == scenario_run, ]
#----------------------------------------------------------------------#
## 2. Generate Complete Simulated Data ####
generate_full_data <- function(n, sc, seed = NULL){
  if (!is.null(seed)) set.seed(seed)
  
  # Generate binary and categorical predictors using marginal distributions
  # approximated from the empirical stroke dataset.
  gender         <- rbinom(n, 1, 0.41)
  hypertension   <- rbinom(n, 1, 0.098)
  heart_disease  <- rbinom(n, 1, 0.05)
  ever_married   <- rbinom(n, 1, 0.66)
  Residence_type <- rbinom(n, 1, 0.51)
  smoking_status <- rbinom(n, 1, 0.44)
  
  work_type <- sample(x = c("private", "govt_job", "self_employed", "children"), 
                      size = n, replace = TRUE, prob = c(0.573, 0.129, 0.164, 0.134))
  work_type <- factor(work_type, 
                      levels = c("private", "govt_job", "self_employed", "children"))
  
  # Continuous variables with correlation structure
  mu <- c(43.2, 106.1, 28.9)
  sd <- c(22.6, 45.3, 7.9)
  cor_mat <- matrix(c(1.000, 0.238, 0.332, 
                      0.238, 1.000, 0.178, 
                      0.332, 0.178, 1.000), nrow = 3, byrow = TRUE)
  
  # Convert correlation matrix into covariance matrix
  # Covariance matrix = D %*% Cor %*% D
  sd_mat <- diag(sd)
  cov_mat <- sd_mat %*% cor_mat %*% sd_mat
  
  cont_vars <- MASS::mvrnorm(n =n, mu = mu, Sigma = cov_mat)
  
  age <- cont_vars[, 1]
  avg_glucose_level <- cont_vars[, 2]
  bmi <- cont_vars[, 3]
  
  # Standardize continuous variables for outcome model
  age_z_pop <- as.numeric(scale(age))
  avg_glucose_level_z_pop <- as.numeric(scale(avg_glucose_level))
  bmi_z_pop <- as.numeric(scale(bmi))
  
  # Generate latent variable W 
  W <- rnorm(n, mean = 0, sd = 1)
  
  # Generate the linear predictor for the true outcome model
  lp_y <- sc$beta1*age_z_pop +
          sc$beta2*hypertension +
          sc$beta3*avg_glucose_level_z_pop +
          sc$beta4*bmi_z_pop +
          sc$beta5*smoking_status +
          sc$beta6*W
  
  # Calibrate the intercept to match the target outcome prevalence
  f_intercept <- function(beta0) {
    mean(plogis(beta0 + lp_y)) - sc$target_prev
  }
  beta0 <- uniroot(f_intercept, interval = c(-10, 10))$root
  
  # Generate true event probabilities and observed binary outcomes
  p_y <- plogis(beta0 + lp_y)
  Y <- rbinom(n, 1, p_y)
  
  # Store the complete simulated dataset before imposing missingness
  sim_data_full <- data.frame(
    gender = gender,
    age = age,
    age_z_pop = age_z_pop,
    hypertension = hypertension,
    heart_disease = heart_disease,
    ever_married = ever_married,
    work_type = work_type,
    Residence_type = Residence_type,
    avg_glucose_level = avg_glucose_level,
    avg_glucose_level_z_pop = avg_glucose_level_z_pop,
    bmi = bmi,
    bmi_z_pop = bmi_z_pop,
    smoking_status = smoking_status,
    W = W,
    stroke = Y, 
    true_p = p_y
  )
  
  return(sim_data_full)
}
#----------------------------------------------------------------------#
## 3. Impose Missingness on BMI and Smoking Status ####
impose_missingness <- function(sim_data_full, sc, seed = NULL){
  if (!is.null(seed)) set.seed(seed)
  
  n <- nrow(sim_data_full)
  X_work <- model.matrix(~ work_type, data = sim_data_full)
  
  # BMI missingness model: probability of BMI being missing
  lp_bmi_miss <- sc$alpha1 * sim_data_full$gender + 
                 sc$alpha2 * sim_data_full$hypertension + 
                 sc$alpha3 * sim_data_full$heart_disease + 
                 sc$alpha4 * sim_data_full$avg_glucose_level_z_pop + 
                 sc$alpha5 * sim_data_full$W
 
  f_bmi <- function(a0) {
    mean(plogis(a0 + lp_bmi_miss)) - (1 - sc$target_obs_bmi)
  }

  # Calibrate alpha_0 to obtain the target BMI observation rate
  alpha0 <- uniroot(f_bmi, interval = c(-20, 20))$root
  p_miss_bmi <- plogis(alpha0 + lp_bmi_miss)
  
  # Missingness indicator: 1 = missing, 0 = observed
  R_bmi_miss <- rbinom(n, 1, p_miss_bmi)
  
  
  # Smoking missingness model: probability of smoking being missing
  lp_smoke_miss <- sc$gamma1 * sim_data_full$hypertension +
                   sc$gamma2 * sim_data_full$ever_married + 
                   sc$gamma3 * X_work[, "work_typegovt_job"] +
                   sc$gamma4 * X_work[, "work_typeself_employed"] +
                   sc$gamma5 * X_work[, "work_typechildren"] + 
                   sc$gamma6 * sim_data_full$W
  
  f_smoke <- function(g0){
    mean(plogis(g0 + lp_smoke_miss)) - (1 - sc$target_obs_smoke)
  }

  # Calibrate gamma_0 to obtain the target smoking observation rate
  gamma0 <- uniroot(f_smoke, interval = c(-20, 20))$root
  p_miss_smoke <- plogis(gamma0 + lp_smoke_miss)
  
  # Missingness indicator: 1 = missing, 0 = observed
  R_smoke_miss <- rbinom(n, 1, p_miss_smoke)
  
  # Impose missingness
  sim_data_miss <- sim_data_full 
  
  # Keep true values for diagnostics only.
  # These variables are not used for model fitting or prediction.
  sim_data_miss$bmi_true <- sim_data_full$bmi
  sim_data_miss$bmi_z_pop_true <- sim_data_full$bmi_z_pop
  sim_data_miss$smoking_status_true <- sim_data_full$smoking_status
  
  # Impose missingness on BMI and smoking status
  sim_data_miss$bmi[R_bmi_miss == 1] <- NA
  sim_data_miss$smoking_status[R_smoke_miss == 1] <- NA
  
  # Define missing-data pattern:
  # p00 = Complete, p10 = BMI missing, p01 = Smoking missing, p11 = Both missing
  sim_data_miss$missing_pattern <- ifelse(
    R_bmi_miss == 0 & R_smoke_miss == 0, "p00_complete",
    ifelse(
      R_bmi_miss == 0 & R_smoke_miss == 1, "p01_smoking_missing",
      ifelse(R_bmi_miss == 1 & R_smoke_miss == 0, "p10_bmi_missing",
             "p11_both_missing")
    )
  )
  
  sim_data_miss$p_miss_bmi <- p_miss_bmi
  sim_data_miss$p_miss_smoke <- p_miss_smoke
  
  sim_data_miss$R_bmi_miss <- R_bmi_miss
  sim_data_miss$R_smoke_miss <- R_smoke_miss
  
  return(sim_data_miss)
}
#----------------------------------------------------------------------#
## 4. Generate Independent Calibration and Prediction Data ####
generate_calib_pred_data <- function(sc, n_calib, n_pred,
                                     seed_calib = 123, seed_pred = 456) {
  
  calib_data_full <- generate_full_data(n = n_calib, sc = sc, seed = seed_calib)
  pred_data_full  <- generate_full_data(n = n_pred,  sc = sc, seed = seed_pred)
  
  calib_data_miss <- impose_missingness(calib_data_full, sc = sc, seed = seed_calib)
  pred_data_miss  <- impose_missingness(pred_data_full,  sc = sc, seed = seed_pred)
  
  # Calculate standardization parameters from the calibration data only
  # This avoids using information from the prediction data during model development.
  age_mean <- mean(calib_data_miss$age, na.rm = TRUE)
  age_sd   <- sd(calib_data_miss$age, na.rm = TRUE)
  
  glucose_mean <- mean(calib_data_miss$avg_glucose_level, na.rm = TRUE)
  glucose_sd   <- sd(calib_data_miss$avg_glucose_level, na.rm = TRUE)
  
  bmi_mean <- mean(calib_data_miss$bmi, na.rm = TRUE)
  bmi_sd   <- sd(calib_data_miss$bmi, na.rm = TRUE)
  
  # Apply calibration-based scaling to both calibration and prediction data.
  calib_data_miss$age_z <- (calib_data_miss$age - age_mean) / age_sd
  pred_data_miss$age_z  <- (pred_data_miss$age - age_mean) / age_sd
  
  calib_data_miss$avg_glucose_level_z <- (calib_data_miss$avg_glucose_level - glucose_mean) / glucose_sd
  pred_data_miss$avg_glucose_level_z <- (pred_data_miss$avg_glucose_level - glucose_mean) / glucose_sd
  
  calib_data_miss$bmi_z <- (calib_data_miss$bmi - bmi_mean) / bmi_sd
  pred_data_miss$bmi_z  <- (pred_data_miss$bmi - bmi_mean) / bmi_sd
  
  return(list(calibration = calib_data_miss, prediction = pred_data_miss))
}
#----------------------------------------------------------------------#
#### 4.1 Calibration and Prediction Data Summary ####
summarise_calib_pred <- function(data, scenario_id, dataset_name){
  
  data %>%
    dplyr::group_by(missing_pattern) %>%
    dplyr::summarise(
      Scenario = scenario_id,
      Dataset = dataset_name,
      N = dplyr::n(),
      Percent = round(100 * dplyr::n() / nrow(data), 2),
      Events = sum(stroke == 1),
      Event_rate = round(mean(stroke), 4),
      Mean_true_p = round(mean(true_p), 4),
      Mean_W = round(mean(W), 4),
      Q95_true_p = round(as.numeric(quantile(true_p, 0.95)), 4),
      .groups = "drop"
    )
}

# Generate calibration and prediction data for the selected scenario
dat_s <- generate_calib_pred_data(sc = sc, n_calib = n_calib, n_pred = n_pred,
                                  seed_calib = 123, seed_pred = 456)

calib_sum <- summarise_calib_pred(data = dat_s$calibration,
                                  scenario_id = scenario_run,
                                  dataset_name = "Calibration")

pred_sum <- summarise_calib_pred(data = dat_s$prediction,
                                 scenario_id = scenario_run,
                                 dataset_name = "Prediction"
)

calib_pred_summary_table <- dplyr::bind_rows(calib_sum, pred_sum)

print(calib_pred_summary_table)
#----------------------------------------------------------------------#
## 5. Prediction Methods ####
#### 5.1 MI Prediction ####
fit_mi_prediction <- function(calib_data_miss, pred_data_miss, m = m_mi, maxit = maxit_mi, seed = NULL){

  if (!is.null(seed)) set.seed(seed)
  
  calib_data_mi <- calib_data_miss[, c("stroke", "gender", "age_z", "hypertension", "heart_disease",
                                       "ever_married", "work_type", "avg_glucose_level_z", "bmi_z", "smoking_status")]
  pred_data_mi_input <- pred_data_miss[, c("gender", "age_z", "hypertension", "heart_disease",
                                           "ever_married", "work_type", "avg_glucose_level_z", "bmi_z", "smoking_status")]
  
  # Generate MI predictions using the prediction averaging approach.
  fit_mi <- mipred(stroke ~ age_z + hypertension + avg_glucose_level_z + bmi_z + smoking_status, 
                   family = binomial, data = calib_data_mi,
                   newdata = pred_data_mi_input, nimp = m, method = "averaging",
                   mice.options = list(maxit = maxit, printFlag = FALSE))
  
  # Average predicted probabilities across imputations
  pred_mi <- rowMeans(fit_mi$pred, na.rm = TRUE)
  
  pred_data_mi <- pred_data_miss
  pred_data_mi$pred_mi <- pred_mi
  
  # Evaluate prediction performance
  brier_mi <- mean((pred_data_mi$stroke - pred_data_mi$pred_mi)^2, na.rm = TRUE)
  auc_mi <- as.numeric(roc(pred_data_mi$stroke, pred_data_mi$pred_mi, quiet = TRUE)$auc)
  
  results_mi <- data.frame(Method = "MI",
                           N = sum(!is.na(pred_data_mi$pred_mi)),
                           Brier = brier_mi,
                           AUC = auc_mi
  )
  
  return(list(results = results_mi,
              pred_data = pred_data_mi[, c("stroke", "true_p", "W", "bmi_z",
                                           "smoking_status", "smoking_status_true",
                                           "pred_mi"), drop = FALSE],
              fit = fit_mi
  ))
}
#----------------------------------------------------------------------#
#### 5.2 Sweeping Prediction ####
# Helper functions for sweeping prediction

# Derive reduced submodel coefficients using the one-step sweep operator
derive_swept_coef <- function(beta_hat, V_hat, remove_terms) {
  all_terms <- names(beta_hat)
  keep_terms <- setdiff(all_terms, remove_terms)
  
  if (length(remove_terms) == 0) {
    return(beta_hat[keep_terms])
  }
  
  beta1 <- beta_hat[keep_terms]
  beta2 <- beta_hat[remove_terms]
  
  V12 <- V_hat[keep_terms, remove_terms, drop = FALSE]
  V22 <- V_hat[remove_terms, remove_terms, drop = FALSE]
  
  beta_reduced <- beta1 - V12 %*% solve(V22) %*% beta2
  beta_reduced <- as.numeric(beta_reduced)
  names(beta_reduced) <- keep_terms
  
  return(beta_reduced)
}

# Generate predictions using the pattern-specific reduced coefficients.
predict_from_coef <- function(coefs, newdata) {
  rhs_terms <- names(coefs)[names(coefs) != "(Intercept)"]
  
  if (length(rhs_terms) == 0) {
    X <- matrix(1, nrow = nrow(newdata), ncol = 1)
    colnames(X) <- "(Intercept)"
  } else {
    X <- model.matrix(
      as.formula(paste("~", paste(rhs_terms, collapse = " + "))),
      data = newdata
    )
    
    if ("(Intercept)" %in% names(coefs)) {
      X <- cbind("(Intercept)" = 1, X[, colnames(X) != "(Intercept)", drop = FALSE])
    }
  }
  
  X <- X[, names(coefs), drop = FALSE]
  lp <- as.numeric(X %*% coefs)
  p <- plogis(lp)
  
  return(p)
}

fit_sw_prediction <- function(calib_data_miss, pred_data_miss, m = m_sw, maxit = maxit_sw, seed = NULL){
  
  calib_data_sw <- calib_data_miss[, c("stroke", "gender", "age_z", "hypertension", "heart_disease",
                                       "ever_married", "work_type", "avg_glucose_level_z", "bmi_z", "smoking_status")]

  # Convert smoking status to a factor for logistic imputation in mice.
  calib_data_sw$smoking_status <- factor(calib_data_sw$smoking_status, levels = c(0, 1))
  
  if(!is.null(seed)) set.seed(seed)
  
  # Initialize the multiple imputation model.
  init <- mice(calib_data_sw, maxit = 0, printFlag = FALSE)
  meth <- init$method
  pred <- init$predictorMatrix
  
  # Impute missing predictors
  meth[] <- ""
  meth["bmi_z"] <- "pmm"
  meth["smoking_status"] <- "logreg"
  meth["stroke"] <- ""
  
  pred[,] <- 1
  diag(pred) <- 0
  pred["stroke", ] <- 0 # stroke is fully observed and is not imputed
  
  # Perform multiple imputation on the calibration dataset
  imp_calib <- mice(calib_data_sw, m = m, method = meth,
                    predictorMatrix = pred, maxit = maxit, printFlag = FALSE)
  
  coef_mat <- NULL
  vcov_list <- vector("list", m)
  fit_list <- vector("list", m)
  
  for (k in 1:m) {
    calib_k <- complete(imp_calib, action = k)
    
    calib_k$smoking_status <- as.numeric(as.character(calib_k$smoking_status))
    
    fit_k <- glm(stroke ~ age_z + hypertension + avg_glucose_level_z + bmi_z + smoking_status,
                 data = calib_k, family = binomial())
    
    fit_list[[k]] <- fit_k
    vcov_list[[k]] <- vcov(fit_k)
    
    coef_k <- coef(fit_k)
    
    if (is.null(coef_mat)) {
      coef_mat <- matrix(NA, nrow = m, ncol = length(coef_k))
      colnames(coef_mat) <- names(coef_k)
    }
    coef_mat[k, ] <- coef_k
  } 
  
  # Pool regression coefficients across imputations.
  beta_pool <- colMeans(coef_mat)
  beta_pool <- as.numeric(beta_pool)
  names(beta_pool) <- colnames(coef_mat)
  
  # Pool covariance matrices using Rubin's rules
  U_bar <- Reduce("+", vcov_list) / m  # Within-imputation covariance
  B_mat <- cov(coef_mat)               # Between-imputation covariance
  V_pool <- U_bar + (1 + 1/m) * B_mat  # Total pooled covariance
  
  # Derive one reduced submodel for each missing-data pattern.
  coef_p00 <- derive_swept_coef(beta_pool, V_pool, remove_terms = character(0))
  coef_p10 <- derive_swept_coef(beta_pool, V_pool, remove_terms = c("bmi_z"))
  coef_p01 <- derive_swept_coef(beta_pool, V_pool, remove_terms = c("smoking_status"))
  coef_p11 <- derive_swept_coef(beta_pool, V_pool, remove_terms = c("bmi_z", "smoking_status"))
  
  # Identify missing-data patterns in the prediction dataset
  pat_bmi_miss   <- is.na(pred_data_miss$bmi_z)
  pat_smoke_miss <- is.na(pred_data_miss$smoking_status)
  
  idx_p00 <- which(!pat_bmi_miss & !pat_smoke_miss)
  idx_p10 <- which( pat_bmi_miss & !pat_smoke_miss)
  idx_p01 <- which(!pat_bmi_miss &  pat_smoke_miss)
  idx_p11 <- which( pat_bmi_miss &  pat_smoke_miss)
  
  # Generate predictions using the corresponding reduced submodel.
  pred_sw <- rep(NA, nrow(pred_data_miss))
  
  # Pattern 00: Complete
  if (length(idx_p00) > 0) pred_sw[idx_p00] <- predict_from_coef(coefs = coef_p00, newdata = pred_data_miss[idx_p00, ])
  # Pattern 10: BMI missing only
  if (length(idx_p10) > 0) pred_sw[idx_p10] <- predict_from_coef(coefs = coef_p10, newdata = pred_data_miss[idx_p10, ])
  # Pattern 01: Smoking missing only
  if (length(idx_p01) > 0) pred_sw[idx_p01] <- predict_from_coef(coefs = coef_p01, newdata = pred_data_miss[idx_p01, ])
  # Pattern 11: Both missing
  if (length(idx_p11) > 0) pred_sw[idx_p11] <- predict_from_coef(coefs = coef_p11, newdata = pred_data_miss[idx_p11, ])
  
  pred_data_sw <- pred_data_miss
  pred_data_sw$pred_sw <- pred_sw
  
  brier_sw <- mean((pred_data_sw$stroke - pred_data_sw$pred_sw)^2, na.rm = TRUE)
  auc_sw <- as.numeric(roc(pred_data_sw$stroke, pred_data_sw$pred_sw, quiet = TRUE)$auc)
  
  results_sw <- data.frame(Method = "Sweeping",
                           N = sum(!is.na(pred_data_sw$pred_sw)),
                           Brier = brier_sw,
                           AUC = auc_sw)
  
  return(list(results = results_sw,
              pred_data = pred_data_sw[, c("stroke", "true_p", "W", "bmi_z",
                                           "smoking_status", "smoking_status_true",
                                           "pred_sw"), drop = FALSE],
              beta_pool = beta_pool,
              V_pool = V_pool,
              coef_pattern = list(p00 = coef_p00,
                                  p10 = coef_p10,
                                  p01 = coef_p01,
                                  p11 = coef_p11),
              fit_list = fit_list))
}
#----------------------------------------------------------------------#
## 6. Performance Evaluation ####
evaluate_prediction <- function(calib_data_miss, pred_data_miss, 
                                m_mi = m_mi, m_sw = m_sw, 
                                maxit_mi = maxit_mi, maxit_sw = maxit_sw, 
                                seed = NULL){
  
  res_mi <- fit_mi_prediction(calib_data_miss, pred_data_miss, 
                              m = m_mi, maxit = maxit_mi, seed = seed)
  
  res_sw <- fit_sw_prediction(calib_data_miss, pred_data_miss, 
                              m = m_sw, maxit = maxit_sw, seed = seed)
  
  # Combine overall performance results
  results_table <- base::rbind(res_mi$results, res_sw$results)
  
  return(list(results = results_table, 
              mi = res_mi, 
              sw = res_sw))
}

# Pattern-specific performance evaluation
evaluate_by_pattern <- function(data, pred_var, method_name = NULL){
  
  pat_bmi_miss   <- is.na(data$bmi_z)
  pat_smoke_miss <- is.na(data$smoking_status)
  
  data$pattern <- ifelse(!pat_bmi_miss & !pat_smoke_miss, "p00_complete",
                         ifelse(!pat_bmi_miss & pat_smoke_miss, "p01_smoking_missing",
                                ifelse(pat_bmi_miss & !pat_smoke_miss, "p10_bmi_missing",
                                       "p11_both_missing")))
  
  pattern_levels <- c("p00_complete", "p01_smoking_missing",
                      "p10_bmi_missing", "p11_both_missing")
  
  results <- data.frame()
  
  for (p in pattern_levels) {
    
    sub_data <- data[data$pattern == p, , drop = FALSE]
    n_total <- nrow(sub_data)
    
    event_rate <- if (n_total > 0) {
      mean(sub_data$stroke == 1, na.rm = TRUE)
    } else {
      NA_real_
    }
    
    sub_pred <- sub_data[!is.na(sub_data[[pred_var]]), , drop = FALSE]
    n_pred <- nrow(sub_pred)
    
    if (n_pred == 0) {
      results <- rbind(results, data.frame(
        method = method_name,
        pattern = p,
        n_total = n_total,
        n_pred = 0,
        event_rate = event_rate,
        mean_pred = NA_real_,
        mean_true_p = if ("true_p" %in% names(sub_data)) mean(sub_data$true_p, na.rm = TRUE) else NA_real_,
        mean_W = if ("W" %in% names(sub_data)) mean(sub_data$W, na.rm = TRUE) else NA_real_,
        cor_true_pred = NA_real_,
        brier = NA_real_,
        auc = NA_real_
      ))
      next
    }
    
    brier <- mean((sub_pred$stroke - sub_pred[[pred_var]])^2)
    
    auc <- if (length(unique(sub_pred$stroke)) < 2) {
      NA_real_
    } else {
      as.numeric(pROC::roc(sub_pred$stroke, sub_pred[[pred_var]], quiet = TRUE)$auc)
    }
    
    cor_true_pred <- if ("true_p" %in% names(sub_pred)) {
      cor(sub_pred$true_p, sub_pred[[pred_var]], use = "complete.obs")
    } else {
      NA_real_
    }
    
    results <- rbind(results, data.frame(
      method = method_name,
      pattern = p,
      n_total = n_total,
      n_pred = n_pred,
      event_rate = event_rate,
      mean_pred = mean(sub_pred[[pred_var]], na.rm = TRUE),
      mean_true_p = if ("true_p" %in% names(sub_data)) mean(sub_data$true_p, na.rm = TRUE) else NA_real_,
      mean_W = if ("W" %in% names(sub_data)) mean(sub_data$W, na.rm = TRUE) else NA_real_,
      cor_true_pred = cor_true_pred,
      brier = brier,
      auc = auc
    ))
  }
  
  return(results)
}
#----------------------------------------------------------------------#
## 7. Single Simulation Run ####
# Run Scenario
data_obj <- generate_calib_pred_data(sc = sc, n_calib = n_calib, n_pred = n_pred,
                                seed_calib = 123, seed_pred = 456)

calib_data_miss <- data_obj$calibration
pred_data_miss  <- data_obj$prediction

# Evaluate MI prediction and sweeping prediction
out_methods <- evaluate_prediction(calib_data_miss = calib_data_miss,
                                   pred_data_miss  = pred_data_miss,
                                   m_mi = m_mi, m_sw = m_sw,
                                   maxit_mi = maxit_mi, maxit_sw = maxit_sw, 
                                   seed = 123
)

print(out_methods$results %>%
        dplyr::mutate(across(where(is.numeric), ~ round(.x, 6))))

#----------------------------------------------------------------------#
#### 7.1 Calibration Model Diagnostics ####
# Compare fitted and true regression coefficients
# Assess potential coefficient shrinkage

# True regression coefficients from the data-generating outcome model
true_beta <- c(age_z = sc$beta1,
               hypertension = sc$beta2,
               avg_glucose_level_z = sc$beta3,
               bmi_z = sc$beta4,
               smoking_status = sc$beta5)

# Estimated regression coefficients from the calibrated
# MI-based starting model used for sweeping prediction
fitted_beta <- out_methods$sw$beta_pool[names(true_beta)]

# Compare fitted and true regression coefficients
beta_diagnostics <- data.frame(Variable = names(true_beta),
                               True_beta = as.numeric(true_beta),
                               Fitted_beta = as.numeric(fitted_beta),
                               Difference = as.numeric(fitted_beta - true_beta),
                               Ratio = as.numeric(fitted_beta / true_beta))

# Round results for reporting
beta_diagnostics <- beta_diagnostics %>%
  dplyr::mutate(
    True_beta = round(True_beta, 3),
    Fitted_beta = round(Fitted_beta, 3),
    Difference = round(Difference, 3),
    Ratio = round(Ratio, 3)
  )

print(beta_diagnostics)
#----------------------------------------------------------------------#
## 8. Pattern-Specific Diagnostic Analysis ####
# MI prediction
pattern_mi <- evaluate_by_pattern(data = out_methods$mi$pred_data,
                                  pred_var = "pred_mi",
                                  method_name = "MI")

# Sweeping prediction
pattern_sw <- evaluate_by_pattern(data = out_methods$sw$pred_data,
                                  pred_var = "pred_sw",
                                  method_name = "Sweeping")

pattern_results <- rbind(pattern_mi, pattern_sw) %>%
  dplyr::mutate(
    pred_minus_true = mean_pred - mean_true_p,
    abs_pred_minus_true = abs(pred_minus_true)
  )

print(pattern_results)

# Diagnostic Plots 
# Plot Setting
pattern_levels <- c("p00_complete", "p01_smoking_missing",
                    "p10_bmi_missing", "p11_both_missing")

method_colors <- c("MI" = "#F8766D",
                   "Sweeping" = "#00BFC4") 

theme_thesis <- theme_minimal(base_size= 16) + 
  theme(panel.background = element_rect(fill = "white", color = NA), 
        plot.background  = element_rect(fill = "white", color = NA), 
        panel.grid.minor = element_blank(), 
        
        axis.title = element_text(size = 17),
        axis.text = element_text(size = 14),
        axis.text.x = element_text(angle = 15, hjust = 1),
        
        strip.text = element_text(size = 16, face = "bold"),
        
        legend.title = element_text(size = 15),
        legend.text = element_text(size = 14),
        legend.position = "bottom")

# Prepare prediction data for plotting
plot_data <- data.frame(true_p = out_methods$mi$pred_data$true_p,
                        stroke = out_methods$mi$pred_data$stroke,
                        W = out_methods$mi$pred_data$W,
                        
                        pred_mi = out_methods$mi$pred_data$pred_mi,
                        pred_sw = out_methods$sw$pred_data$pred_sw,
                        
                        bmi_miss = is.na(out_methods$mi$pred_data$bmi_z),
                        smoke_miss = is.na(out_methods$mi$pred_data$smoking_status))

# Define missing-data patterns
plot_data$Pattern <- ifelse(
  !plot_data$bmi_miss & !plot_data$smoke_miss, "p00_complete",
  ifelse(
    plot_data$bmi_miss & !plot_data$smoke_miss, "p10_bmi_missing",
    ifelse(
      !plot_data$bmi_miss & plot_data$smoke_miss, "p01_smoking_missing",
      "p11_both_missing"
    )
  )
)

# Reshape prediction data to long format
plot_long <- plot_data %>%
  tidyr::pivot_longer(cols = c(pred_mi, pred_sw),
                      names_to = "Method",
                      values_to = "Pred") %>%
  dplyr::mutate(Method  = ifelse(Method == "pred_mi", "MI", "Sweeping"), 
                Method  = factor(Method, levels = c("MI", "Sweeping")), 
                Pattern = factor(Pattern, levels = pattern_levels))

# Plot Prediction Versus True Probability by Missing Pattern
ggplot(plot_long, aes(x = true_p, y = Pred, color = Method)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.8) +
  
  facet_wrap(~ Pattern, ncol = 2) + 
  scale_color_manual(values = method_colors) +
  
  labs(x = "True probability",
       y = "Predicted probability",
       color = "Method") +
  theme_thesis

#------------------------------------------------------------#
#### 8.1 B2 Diagnostic Investigation ####
# Investigate the mechanisms underlying prediction performance
# in Scenario B2. 

if (scenario_run == "B2") {
  cat("\nRunning B2 diagnostic investigation...\n")
 
  # Prepare diagnostic data 
  plot_data_b2 <- plot_data %>%
    dplyr::mutate(bmi_z_true = pred_data_miss$bmi_z_pop_true,
                  smoking_status_true = pred_data_miss$smoking_status_true,
                  Pattern = factor(Pattern, levels = pattern_levels))
  
  # Combine MI and sweeping predictions into a common format
  plot_long_b2 <- plot_data_b2 %>%
    dplyr::select(true_p, stroke, Pattern, 
                  bmi_z_true, smoking_status_true,
                  pred_mi, pred_sw) %>%
    tidyr::pivot_longer(cols = c(pred_mi, pred_sw),
                        names_to = "Method",
                        values_to = "Pred") %>%
    dplyr::mutate(
      Method = dplyr::recode(Method, pred_mi = "MI", pred_sw = "Sweeping"),
      Method = factor(Method, levels = c("MI", "Sweeping")))
  #------------------------------------------------------------#
  ## 1. p01 Smoking-Missing Diagnostic 
  # Summarize prediction performance according to the
  # true smoking status within the p01 pattern.
  b2_p01_smoking_table <- plot_long_b2 %>%
    dplyr::filter(Pattern == "p01_smoking_missing") %>%
    dplyr::group_by(Method, smoking_status_true) %>%
    dplyr::summarise(
      N = dplyr::n(),
      mean_true_p = mean(true_p, na.rm = TRUE),
      mean_pred = mean(Pred, na.rm = TRUE),
      pred_bias = mean(Pred - true_p, na.rm = TRUE),
      .groups = "drop"
    )

  print(b2_p01_smoking_table)
  
  # Scatterplot of predicted versus true probabilities,
  # stratified by true smoking status
  p_b2_p01_smoking <- plot_long_b2 %>%
    dplyr::filter(Pattern == "p01_smoking_missing") %>%
    ggplot(aes(x = true_p,
               y = Pred,
               color = factor(smoking_status_true))) +
    geom_point(alpha = 0.65, size = 2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    facet_wrap(~ Method) +
    scale_color_manual(values = c("0" = "grey60","1" = "#2E8B57"),
                       labels = c("Non-smoker", "Smoker")) +
    labs(x = "True probability",
         y = "Predicted probability",
         color = "True smoking") +
    theme_thesis +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
  
  print(p_b2_p01_smoking)

  #------------------------------------------------------------#
  ## 2. p10 BMI-Missing Diagnostic 
  # Summarize prediction performance across
  # true BMI quartiles within the p10 pattern.
  b2_p10_bmi_table <- plot_long_b2 %>%
    dplyr::filter(Pattern == "p10_bmi_missing") %>%
    dplyr::mutate(
      bmi_group = dplyr::ntile(bmi_z_true, 4),
      bmi_group = factor(
        bmi_group,
        levels = 1:4,
        labels = c("Q1 lowest BMI", "Q2", "Q3", "Q4 highest BMI")
      )
    ) %>%
    dplyr::group_by(Method, bmi_group) %>%
    dplyr::summarise(
      N = dplyr::n(),
      mean_bmi = mean(bmi_z_true, na.rm = TRUE),
      mean_true_p = mean(true_p, na.rm = TRUE),
      mean_pred = mean(Pred, na.rm = TRUE),
      pred_bias = mean(Pred - true_p, na.rm = TRUE),
      .groups = "drop"
    )
  
  print(b2_p10_bmi_table)
}

#----------------------------------------------------------------------#
#### 8.2 D2 Diagnostic Investigation ####
if (scenario_run == "D2") {
  cat("\nRunning D2 hidden high-risk diagnostic investigation...\n")
  
  # Prepare diagnostic data
  plot_data_d2 <- plot_data %>%
    dplyr::mutate(Pattern = factor(Pattern, levels = pattern_levels))
  
  # Combine MI and sweeping predictions into a common format
  plot_long_d2 <- plot_data_d2 %>%
    dplyr::select(true_p, W, Pattern, pred_mi, pred_sw) %>%
    tidyr::pivot_longer(cols = c(pred_mi, pred_sw),
                        names_to = "Method",
                        values_to = "Pred") %>%
    dplyr::mutate(
      Method = dplyr::recode(Method, pred_mi = "MI", pred_sw = "Sweeping"),
      Method = factor(Method, levels = c("MI", "Sweeping")),
      error = Pred - true_p
    )
  
  #------------------------------------------------------------#
  ## 1. W Distribution by Missing-Data Pattern
  # Summarize the distribution of the hidden variable W
  # across missing-data patterns.
  d2_w_pattern_table <- plot_data_d2 %>%
    dplyr::group_by(Pattern) %>%
    dplyr::summarise(
      N = dplyr::n(), 
      mean_W = mean(W, na.rm = TRUE),
      mean_true_p = mean(true_p, na.rm = TRUE),
      .groups = "drop"
    )
  
  print(d2_w_pattern_table)
  
  #------------------------------------------------------------#
  ## 2. Prediction Error versus W
  # Plot prediction error against the hidden variable W.
  p_d2_error_w <- plot_long_d2 %>%
    ggplot(aes(x = W, y = error, color = Method)) +
    geom_point(alpha = 0.55, size = 1.5) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_wrap(~ Pattern, ncol = 2) +
    scale_color_manual(values = method_colors) +
    labs(x = "Hidden variable W",
         y = "Prediction error",
         color = "Method") +
    theme_thesis +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
  
  print(p_d2_error_w)

  #------------------------------------------------------------#
  ## 3. p11 Hidden High-Risk Subgroup
  # Summarize prediction performance across W tertiles
  # within the p11 pattern.
  d2_p11_w_table <- plot_long_d2 %>%
    dplyr::filter(Pattern == "p11_both_missing") %>%
    dplyr::mutate(W_group = dplyr::ntile(W, 3),
                  W_group = factor(W_group,
                                   levels = 1:3,
                                   labels = c("Low W", "Middle W", "High W"))) %>%
    dplyr::group_by(Method, W_group) %>%
    dplyr::summarise(N = dplyr::n(),
                     mean_W = mean(W, na.rm = TRUE),
                     mean_true_p = mean(true_p, na.rm = TRUE),
                     mean_pred = mean(Pred, na.rm = TRUE),
                     pred_bias = mean(Pred - true_p, na.rm = TRUE),
                     .groups = "drop")
  
  print(d2_p11_w_table)
}

