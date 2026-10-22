#########################################################################
# 01_Data_Preparation.R
#
# Prepare the empirical dataset and derive quantities used for
# the simulation study.
#
# Stroke Prediction Dataset:
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
#########################################################################

library(mice)
stroke <- read.csv("Stroke/healthcare-dataset-stroke-data.csv", stringsAsFactors = FALSE)

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

# Binary outcome
stroke$stroke         <- factor(stroke$stroke)
#----------------------------------------------------------------------#
## 2. Dataset and Missing-Data Description ####
# Summarize the outcome and missing-data structure.

# Outcome (Stroke) distribution
stroke_table <- table(stroke$stroke)
stroke_prop <- round(prop.table(stroke_table)*100, 2)
stroke_res <- cbind(Count = stroke_table, Percent = stroke_prop)

cat("Outcome (stroke) Distribution\n")
print(stroke_res)

# Missing rate per variable
cat("BMI missing rate:", round(mean(is.na(stroke$bmi))*100, 2),"%", 
    "\nSmoking status missing rate:", round(mean(is.na(stroke$smoking_status))*100, 2), "%")

# Missing pattern for BMI and smoking_status
cat("\nMissing Pattern Distribution\n")
pattern_df <- data.frame(
  Pattern = c("Complete", "Smoking missing", "BMI missing", "Both missing"), 
  Count = c(sum(!is.na(stroke$bmi) & !is.na(stroke$smoking_status)), 
            sum(!is.na(stroke$bmi) &  is.na(stroke$smoking_status)),
            sum( is.na(stroke$bmi) & !is.na(stroke$smoking_status)),
            sum( is.na(stroke$bmi) &  is.na(stroke$smoking_status))
  )
)
pattern_df$Percent <- round(pattern_df$Count/sum(pattern_df$Count)*100, 2)
print(pattern_df)
#----------------------------------------------------------------------#
## 3. Missing Mechanism Exploration ####
# Explore predictors associated with missingness.

# Define missingness indicators: 1 = missing, 0 = observed
stroke$R_bmi_miss <- ifelse(is.na(stroke$bmi),1,0)
stroke$R_smoke_miss <- ifelse(is.na(stroke$smoking_status),1,0)

# Logistic regression model for BMI missingness
bmi_miss_fit <- glm(R_bmi_miss ~ gender + age + hypertension + heart_disease + ever_married +
                                 work_type + Residence_type + avg_glucose_level + stroke,
                    data = stroke, family = binomial())
summary(bmi_miss_fit)

# Overall likelihood ratio test for work_type
bmi_no_work_miss_fit <- glm(R_bmi_miss ~ gender + age + hypertension + heart_disease + ever_married +
                                         Residence_type + avg_glucose_level + stroke,
                            data = stroke, family = binomial())

anova(bmi_miss_fit, bmi_no_work_miss_fit, test = "Chisq")


# Logistic regression model for smoking-status missingness
smoke_miss_fit <- glm(R_smoke_miss ~ gender + age + hypertension + heart_disease + ever_married + 
                                     work_type + Residence_type + avg_glucose_level + stroke,
                      data = stroke, family = binomial())

summary(smoke_miss_fit)

# Overall likelihood ratio test for work_type
smoke_no_work_miss_fit <- glm(R_smoke_miss ~ gender + age + hypertension + heart_disease + ever_married +
                                             Residence_type + avg_glucose_level + stroke,
                              data = stroke, family = binomial())

anova(smoke_miss_fit, smoke_no_work_miss_fit, test = "Chisq")
#----------------------------------------------------------------------#
## 4. Multiple Imputation for Empirical Model Summaries ####
# Impute missing predictors to summarize the empirical outcome model

set.seed(123)

m_mi_exploration <- 20
maxit_mi_exploration <- 20

# Initialize mice
init <- mice(stroke, maxit = 0, printFlag = FALSE)
meth <- init$method
pred <- init$predictorMatrix

meth[] <- ""
meth["bmi"] <- "pmm"
meth["smoking_status"] <- "logreg"

pred[, ] <- 1
diag(pred) <- 0
pred["stroke", ] <- 0 # Stroke outcome is not imputed

imp <- mice(data = stroke, m = m_mi_exploration, method = meth, predictorMatrix = pred, 
            maxit = maxit_mi_exploration, printFlag = FALSE)

# Fit pooled MI model
fit_mi <- with(imp,{
  
  # Standardized continuous variables within each imputed dataset
  age_z <- as.numeric(scale(age))
  bmi_z <- as.numeric(scale(bmi))
  avg_glucose_level_z <- as.numeric(scale(avg_glucose_level))
  
  # Logistic regression model
  glm(stroke ~ gender + age_z + hypertension + heart_disease + ever_married + 
        work_type + Residence_type + avg_glucose_level_z + bmi_z + smoking_status,
      family = binomial())
})
pool_mi <- pool(fit_mi)
summary(pool_mi)
#----------------------------------------------------------------------#
## 5. Empirical Quantities for Simulation Design ####
# Derive quantities used in the simulation study.
#----------------------------------------------------------------------#
### 5.1 Predictor Distributions ####
# Summarize predictor distributions.

imp_list <- complete(imp, action = "all")
m <- length(imp_list)

# Continuous predictors
con_vars <- c("age", "avg_glucose_level", "bmi")

mean_list <- lapply(imp_list, function(dat){
  sapply(dat[con_vars], mean, na.rm = TRUE)
})

sd_list <- lapply(imp_list, function(dat){
  sapply(dat[con_vars], sd, na.rm = TRUE)
})

mean_avg <- Reduce("+", mean_list) / m
sd_avg <- Reduce("+", sd_list) / m

con_sum <- data.frame(variable = con_vars, 
                      mean = round(mean_avg, 3), 
                      sd = round(sd_avg, 3))
print(con_sum)

# Categorical variables
cat_vars <- c("gender", "hypertension", "heart_disease", "ever_married", 
              "work_type", "Residence_type", "smoking_status")

cat_sum_list <- lapply(cat_vars, function(v) {
  # Calculate category proportions within each imputed dataset
  prop_list <- lapply(imp_list, function(dat) {
    prop.table(table(dat[[v]]))
  })
  
  # Collect all possible levels across imputations
  all_levels <- unique(unlist(lapply(prop_list, names)))
  
  # Proportion matrix: rows = categories, cols = imputations
  prop_matrix <- matrix(0, nrow = length(all_levels), ncol = length(prop_list))
  rownames(prop_matrix) <- all_levels
  
  for (j in seq_along(prop_list)) {
    p <- prop_list[[j]]
    prop_matrix[names(p), j] <- as.numeric(p)
  }
  
  # Average proportion across imputations
  prop_avg <- rowMeans(prop_matrix)
  
  data.frame(variable = v,
             category = all_levels,
             proportion = round(prop_avg, 4),
             row.names = NULL
  )
})

cat_sum_df <- do.call(rbind, cat_sum_list)
print(cat_sum_df)
#----------------------------------------------------------------------#
### 5.2 Correlation Structure ####
# Estimate correlations among continuous predictors.

cor_list <- lapply(imp_list, function(dat) {
  cor(dat[, c("age", "avg_glucose_level", "bmi")])
})

cor_avg <- Reduce("+", cor_list) / length(cor_list)
round(cor_avg, 3)

