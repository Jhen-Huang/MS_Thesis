# =============================================================================
# 00_setup.R
# Install / load required packages and define project paths.
# Run this once before the analysis scripts. All scripts assume the working
# directory is the repository root (open `sweeping-vs-imputation.Rproj`).
# =============================================================================

required_packages <- c(
  "mice",    # Multiple imputation by chained equations
  "mipred",  # Prediction with multiple imputation (prediction averaging)
  "pROC",    # ROC curves and AUC
  "MASS",    # Multivariate normal data generation (simulation)
  "dplyr",   # Data manipulation
  "tidyr",   # Reshaping
  "ggplot2"  # Figures
)

missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

# Project paths
DATA_FILE   <- file.path("data", "healthcare-dataset-stroke-data.csv")
FIG_DIR     <- "figures"
RESULTS_DIR <- "results"

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(DATA_FILE)) {
  stop(
    "Dataset not found at '", DATA_FILE, "'.\n",
    "Download it from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset ",
    "and place the CSV in the data/ folder (see data/README.md)."
  )
}
