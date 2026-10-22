#########################################################################
# 03_Real_Data_Analysis.R
#
# Purpose:
# Evaluate prediction performance of MI prediction and sweeping
# prediction using 10-fold cross-validation.
#
# Stroke Prediction Dataset:
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
#########################################################################

library(mice)
library(mipred)
library(pROC)
library(dplyr)
library(ggplot2)
stroke <- read.csv("Stroke/healthcare-dataset-stroke-data.csv",  stringsAsFactors = FALSE)

#----------------------------------------------------------------------#
## 1. Data Preparation ####
# Prepare the dataset for analysis.

# Remove identifier
stroke$id <- NULL

# Recode unavailable values as missing
stroke$bmi[stroke$bmi == "N/A"] <- NA
stroke$smoking_status[stroke$smoking_status == "Unknown"] <- NA

# Convert BMI to numeric
stroke$bmi <- as.numeric(stroke$bmi)

# Merge sparse categories
stroke$gender[stroke$gender == "Other"] <- "Male"
stroke$work_type[stroke$work_type == "Never_worked"] <- "Self-employed"

# Recode smoking status into a binary variable
# Smoking: 0 = never smoked, 1 = ever smoked
stroke$smoking_status <- ifelse(is.na(stroke$smoking_status), NA,
                                ifelse(stroke$smoking_status == "never smoked", 0, 1))

# Set variable types and reference categories
stroke$gender         <- factor(stroke$gender, levels = c("Female", "Male"))
stroke$hypertension   <- factor(stroke$hypertension, levels = c(0, 1))
stroke$heart_disease  <- factor(stroke$heart_disease, levels = c(0, 1))
stroke$ever_married   <- factor(stroke$ever_married, levels = c("No", "Yes"))
stroke$work_type      <- factor(stroke$work_type,
                                levels = c("Private", "Govt_job", 
                                           "Self-employed", "children"))
stroke$Residence_type <- factor(stroke$Residence_type, levels = c("Rural", "Urban"))
stroke$smoking_status <- factor(stroke$smoking_status, levels = c(0, 1))

# Outcome: convert to numeric (0/1) for prediction modelling
# and performance evaluation (Brier score, AUC, calibration).
stroke$stroke <- as.numeric(as.character(stroke$stroke))

#----------------------------------------------------------------------#
## 2. Cross-Validation Framework ####
# Ten-fold cross-validation was used to evaluate prediction performance.
# Within each fold:
#   1. Calibrate the prediction model using the training fold.
#   2. Apply the fitted model to the held-out fold.
#   3. Store predictions for performance evaluation.

set.seed(123)

K <- 10
fold_id <- sample(rep(1:K, length.out = nrow(stroke)))

stroke$pred_mi_cv <- NA_real_
stroke$pred_sw_cv <- NA_real_
stroke$fold <- fold_id

# MI prediction settings (prediction averaging)
m_mi <- 5      # Number of imputations
maxit_mi <- 5  # Number of MICE iterations

# Sweeping prediction settings (MI-based starting model)
m_sw <- 20      # Number of imputations
maxit_sw <- 20  # Number of MICE iterations

for(k in 1:K){
  
  cat("Running fold", k, "\n")
  
  train <- stroke[fold_id != k, ]
  test  <- stroke[fold_id == k, ]
  test_index <- which(fold_id == k)
  
  # Standardize continuous predictors using training-fold statistics
  age_mean <- mean(train$age, na.rm = TRUE)
  age_sd   <- sd(train$age, na.rm = TRUE)

  glucose_mean <- mean(train$avg_glucose_level, na.rm = TRUE)
  glucose_sd   <- sd(train$avg_glucose_level, na.rm = TRUE)

  bmi_mean <- mean(train$bmi, na.rm = TRUE)
  bmi_sd   <- sd(train$bmi, na.rm = TRUE)

  train$age_z <- (train$age - age_mean) / age_sd
  test$age_z  <- (test$age - age_mean) / age_sd

  train$avg_glucose_level_z <- (train$avg_glucose_level - glucose_mean) / glucose_sd
  test$avg_glucose_level_z  <- (test$avg_glucose_level - glucose_mean) / glucose_sd

  train$bmi_z <- (train$bmi - bmi_mean) / bmi_sd
  test$bmi_z  <- (test$bmi - bmi_mean) / bmi_sd
  #----------------------------------------------------------------------#
 
  #### 2.1 MI Prediction ####
  # MI prediction is implemented using mipred with prediction averaging.
  # The model is calibrated on the training fold and applied to the held-out fold.
  cat("  MI prediction\n")
  
  train_mi <- train[, c("stroke", "gender", "age_z", "hypertension", "heart_disease", "ever_married", 
                        "work_type", "Residence_type", "avg_glucose_level_z", "bmi_z", "smoking_status")]
  test_mi <- test[, c("gender", "age_z", "hypertension", "heart_disease", "ever_married", 
                      "work_type", "Residence_type", "avg_glucose_level_z", "bmi_z", "smoking_status")]

  fit_mi_pred <- mipred(stroke ~ gender + age_z + hypertension + heart_disease + ever_married + work_type + 
                                 Residence_type + avg_glucose_level_z + bmi_z + smoking_status,
                        family = binomial, data = train_mi, newdata = test_mi,
                        nimp = m_mi, method = "averaging",
                        mice.options = list(maxit = maxit_mi, printFlag = FALSE)
  )

  pred_mi <- rowMeans(fit_mi_pred$pred)
  stroke$pred_mi_cv[test_index] <- pred_mi
  #----------------------------------------------------------------------#
  ## 2.2 Sweeping Prediction ####
  cat("  Sweeping prediction\n")

  train_sw <- train[, c("stroke", "gender", "age_z", "hypertension", "heart_disease",
                        "ever_married", "work_type", "Residence_type",
                           "avg_glucose_level_z", "bmi_z", "smoking_status")]
  
  test_sw <- test[, c("stroke", "gender", "age_z", "hypertension", "heart_disease",
                      "ever_married", "work_type", "Residence_type",
                      "avg_glucose_level_z", "bmi_z", "smoking_status")]

  train_sw$smoking_status <- as.numeric(as.character(train_sw$smoking_status))
  test_sw$smoking_status  <- as.numeric(as.character(test_sw$smoking_status))

  ### Step 1: Construct MI-based starting model
  # Impute missing predictors in the training fold only
  init <- mice(train_sw, maxit = 0, printFlag = FALSE)
  meth <- init$method
  pred <- init$predictorMatrix
  
  meth[] <- ""
  meth["bmi_z"] <- "pmm"
  meth["smoking_status"] <- "logreg"
  
  pred[,] <- 1
  diag(pred) <- 0
  pred["stroke", ] <- 0 # Outcome is fully observed and is not imputed
  
  imp_train_sw <- mice(train_sw, m = m_sw, method = meth, predictorMatrix = pred, 
                       maxit = maxit_sw, printFlag = FALSE)
  
  # Fit full model in each imputed dataset
  fit_sw_full <- with(imp_train_sw, {
    glm(stroke ~ gender + age_z + hypertension + heart_disease + ever_married +
                 work_type + Residence_type + avg_glucose_level_z + bmi_z + smoking_status,
        family = binomial()
    )
  })

  ### Step 2: Pooled regression coefficients and covariance matrix
  pool_glm_with_vcov <- function(fit_mira){
    # Extract fitted models from all imputed datasets
    fits <- fit_mira$analyses
    m <- length(fits)
    
    coef_names <- names(coef(fits[[1]]))
    p <- length(coef_names)
  
    # Beta matrix
    beta_mat <- matrix(NA_real_, nrow = m, ncol = p, 
                       dimnames = list(paste0("imp", 1:m), coef_names))
  
    # Variance-covariance matrix
    U_list <- vector("list", m)
  
    for (k in seq_len(m)) {
      beta_k <- coef(fits[[k]])
      U_k <- vcov(fits[[k]])
    
      beta_k <- beta_k[coef_names]
      U_k <- U_k[coef_names, coef_names]
      
      beta_mat[k, ] <- beta_k
      U_list[[k]] <- U_k
    }
  
    # Pooled coefficient
    beta_bar <- colMeans(beta_mat)
    # Within-imputation variance
    U_bar <- Reduce(`+`, U_list) / m
    # Between-imputation variance  
    B <- matrix(0, nrow = p, ncol = p,
                dimnames = list(coef_names, coef_names))
  
    for (k in seq_len(m)) {
      d <- matrix(beta_mat[k, ] - beta_bar, ncol = 1)
      B <- B + d %*% t(d)
    }
  
    B <- B / (m - 1)
  
    # Rubin total variance
    Tmat <- U_bar + (1 + 1 / m) * B
    # Pooled standard errors
    pooled_se <- sqrt(diag(Tmat))
  
    list(pooled_beta = beta_bar,
         pooled_vcov = Tmat,
         within_vcov = U_bar,
         between_vcov = B,
         pooled_se = pooled_se,
         beta_each_imp = beta_mat,
         m = m
    )
  }

  pooled_sw <- pool_glm_with_vcov(fit_sw_full)
  pooled_beta <- pooled_sw$pooled_beta
  pooled_vcov <- pooled_sw$pooled_vcov

  ### Step 3: Derive reduced prediction models
  full_formula <- stroke ~ gender + age_z + hypertension + heart_disease + ever_married + 
                           work_type + Residence_type + avg_glucose_level_z + bmi_z + smoking_status

  # Logistic inverse link function
  expit <- function(x) 1 / (1 + exp(-x))

  # Sweeping formula: beta_red = beta_O - V_OM %*% solve(V_MM) %*% beta_M
  # beta: pooled full model coefficients
  # V: pooled covariance matrix
  derive_reduced_model <- function(beta, V, remove_terms) {
    all_terms <- names(beta)
  
    remove_terms <- intersect(remove_terms, all_terms)
    # Define keep terms
    keep_terms <- setdiff(all_terms, remove_terms)
  
    # If no predictors are removed, the reduced model is identical to the full model.
    if (length(remove_terms) == 0) {
      return(list(beta_red = beta[keep_terms],
                  vcov_red = V[keep_terms, keep_terms, drop = FALSE],
                  keep_terms = keep_terms,
                  remove_terms = remove_terms))
    }
  
    beta_O <- beta[keep_terms]
    beta_M <- beta[remove_terms]
  
    # O = observed / keep terms
    # M = missing / remove terms
    V_OO <- V[keep_terms, keep_terms, drop = FALSE]
    V_OM <- V[keep_terms, remove_terms, drop = FALSE]
    V_MO <- V[remove_terms, keep_terms, drop = FALSE]
    V_MM <- V[remove_terms, remove_terms, drop = FALSE]
  
    V_MM_inv <- solve(V_MM)
  
    # Derive reduced regression coefficients
    beta_red <- beta_O - V_OM %*% V_MM_inv %*% beta_M
    # Derive reduced covariance matrix
    vcov_red <- V_OO - V_OM %*% V_MM_inv %*% V_MO
  
    beta_red <- as.numeric(beta_red)
    names(beta_red) <- keep_terms
  
    list(beta_red = beta_red,
         vcov_red = vcov_red,
         keep_terms = keep_terms,
         remove_terms = remove_terms
    )
  }
  
  # Use reduced beta to predict newdata 
  predict_reduced_model <- function(newdata, formula, beta_red) {
    newdata2 <- newdata
  
    # To avoid model.matrix dropping rows because of NA
    if (!("bmi_z" %in% names(beta_red)) && "bmi_z" %in% names(newdata2)) {
      newdata2$bmi_z[is.na(newdata2$bmi_z)] <- 0
    }
  
    if (!("smoking_status" %in% names(beta_red)) && "smoking_status" %in% names(newdata2)) {
     newdata2$smoking_status[is.na(newdata2$smoking_status)] <- 0
    }
  
    rhs_terms <- delete.response(terms(formula))
    mm <- model.matrix(rhs_terms, data = newdata2)
  
    X <- matrix(0, nrow = nrow(mm), ncol = length(beta_red))
    colnames(X) <- names(beta_red)
  
    common_terms <- intersect(colnames(mm), names(beta_red))
    if (length(common_terms) > 0) {
      X[, common_terms] <- mm[, common_terms, drop = FALSE]
    }
  
    # Linear predictor
    eta <- as.vector(X %*% beta_red)
    expit(eta)
  }

  ### Step 4: Build reduced submodels
  # Define missing Patterns in test dataset
  test_sw$pattern <- ifelse(
    !is.na(test_sw$bmi_z) & !is.na(test_sw$smoking_status), "p00_complete",
    ifelse(
      is.na(test_sw$bmi_z) & !is.na(test_sw$smoking_status), "p10_bmi_missing",
      ifelse(
        !is.na(test_sw$bmi_z) & is.na(test_sw$smoking_status), "p01_smoking_missing", 
        "p11_both_missing")
    )
  )

  pattern_remove_terms <- list(p00_complete = character(0),
                               p10_bmi_missing = c("bmi_z"),
                               p01_smoking_missing = c("smoking_status"),
                               p11_both_missing = c("bmi_z", "smoking_status")
  )
  reduced_models <- list()

  for (pat in names(pattern_remove_terms)) {
    reduced_models[[pat]] <- derive_reduced_model(beta = pooled_beta,
                                                  V = pooled_vcov,
                                                  remove_terms = pattern_remove_terms[[pat]])
  }
 
  ### Step 5: Generate sweeping predictions
  pred_sw <- rep(NA_real_, nrow(test_sw))
  
  for (pat in names(reduced_models)) {
    idx <- which(test_sw$pattern == pat)
    if (length(idx) == 0) next
  
    beta_red <- reduced_models[[pat]]$beta_red
  
    pred_sw[idx] <- predict_reduced_model(newdata = test_sw[idx, , drop = FALSE],
                                          formula = full_formula,
                                          beta_red = beta_red)
  }

  # Save cross-validated predictions back to the full dataset
  test_sw$pred_sw <- pred_sw
  stroke$pred_sw_cv[test_index] <- pred_sw
}
#----------------------------------------------------------------------#
## 3. Overall Evaluation ####
brier_score <- function(y, p) mean((y - p)^2, na.rm = TRUE)

auc_score <- function(y, p) {
  idx <- !is.na(p)
  
  if (length(unique(y[idx])) < 2) {
    return(NA_real_)
  }
  
  roc_obj <- roc(response = y[idx], predictor = p[idx], quiet = TRUE)
  as.numeric(auc(roc_obj))
}

evaluate_predictions <- function(y, p) {
  idx <- !is.na(p)
  
  data.frame(N = sum(idx),
             Brier = brier_score(y[idx], p[idx]),
             AUC = auc_score(y, p)
  )
}

# Overall evaluation
y_test_all <- stroke$stroke

eval_mi <- evaluate_predictions(y_test_all, stroke$pred_mi_cv)
eval_mi$Method <- "MI Prediction"

eval_sw <- evaluate_predictions(y_test_all, stroke$pred_sw_cv)
eval_sw$Method <- "Sweeping Prediction"

eval_overall <- dplyr::bind_rows( eval_mi, eval_sw) %>%
  dplyr::select(Method, N, Brier, AUC)

print(eval_overall)
#----------------------------------------------------------------------#
## 4. Pattern-Specific Evaluation ####

# Define missing-data patterns based on BMI and smoking status
define_missing_pattern <- function(data){
  
  bmi_missing <- is.na(data$bmi)
  smoke_missing <- is.na(data$smoking_status)
  
  data$missing_pattern <- ifelse(
    !bmi_missing & !smoke_missing, "p00_complete",
    ifelse(
      bmi_missing & !smoke_missing, "p10_bmi_missing",
      ifelse(
        !bmi_missing & smoke_missing, "p01_smoking_missing",
        "p11_both_missing"
      )
    )
  )
  
  return(data)
}

stroke <- define_missing_pattern(stroke)

# Missing-data pattern summary
pattern_summary <- stroke %>%
  group_by(missing_pattern) %>%
  summarise(N = n(),
            Event_rate = mean(stroke),
            Mean_MI = mean(pred_mi_cv, na.rm = TRUE),
            Mean_Sweeping = mean(pred_sw_cv, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(Event_rate = round(Event_rate, 4), 
         Mean_MI = round(Mean_MI, 4), 
         Mean_Sweeping = round(Mean_Sweeping, 4))

print(pattern_summary)

# Pattern-specific prediction performance
evaluate_by_pattern <- function(data, pred_var, method_name) {
  
  pattern_levels <- c("p00_complete", "p01_smoking_missing",
                      "p10_bmi_missing", "p11_both_missing")
  
  res_list <- lapply(pattern_levels, function(pat) {
    
    subdat <- data[data$missing_pattern == pat, ]
    p <- subdat[[pred_var]]
    y <- subdat$stroke
    
    n_pred <- sum(!is.na(p))
    
    if (nrow(subdat) == 0 || n_pred == 0) {
      return(data.frame(Method = method_name,
                        Missing_pattern = pat,
                        N_total = nrow(subdat),
                        N_pred = n_pred,
                        Brier = NA_real_,
                        AUC = NA_real_)
      )
    }
    
    auc_val <- NA_real_
    if (length(unique(y[!is.na(p)])) == 2) {
      auc_val <- as.numeric(roc(y[!is.na(p)], p[!is.na(p)], quiet = TRUE)$auc)
    }
    
    data.frame(Method = method_name,
               Missing_pattern = pat,
               N_total = nrow(subdat),
               N_pred = n_pred,
               Brier = mean((y[!is.na(p)] - p[!is.na(p)])^2),
               AUC = auc_val)
  })
  
  bind_rows(res_list)
}

# Pattern-specific evaluation for MI prediction and sweeping prediction
sub_mi_data <- stroke
sub_mi_data$pred <- stroke$pred_mi_cv

sub_sw_data <- stroke
sub_sw_data$pred <- stroke$pred_sw_cv

sub_eval_mi <- evaluate_by_pattern(sub_mi_data, pred_var = "pred",
                                   method_name = "MI Prediction")
sub_eval_sw <- evaluate_by_pattern(sub_sw_data, pred_var = "pred", 
                                   method_name = "Sweeping Prediction")

sub_pattern_results <- bind_rows(sub_eval_mi, sub_eval_sw) %>%
  mutate(Brier = round(Brier, 4),
         AUC = round(AUC, 4)
  )

print(sub_pattern_results)
