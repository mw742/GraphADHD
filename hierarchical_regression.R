#!/usr/bin/env Rscript

library(dplyr)

# Load datasets
mh_p_asr       <- read.csv("mh_p_asr.csv", header=TRUE, stringsAsFactors=FALSE)
mh_p_abcl      <- read.csv("mh_p_abcl.csv", header=TRUE, stringsAsFactors=FALSE)
adhd_phenotype <- read.csv("mh_p_cbcl.csv", header=TRUE, stringsAsFactors=FALSE)

prs_data <- read.table("/gpfs/group/ehlers/mwei/ABCD/QC_results_round3/PRS_cauc_only_results.best",
                       header=TRUE, sep=" ", stringsAsFactors=FALSE)

sex_df <- read.table("sexphenotypes_original.txt",
                     header=FALSE, stringsAsFactors=FALSE)
colnames(sex_df) <- c("FID","IID","sex")

event_of_interest <- "2_year_follow_up_y_arm_1"

# Define the columns you need from mh_p_abcl
available_abcl_cols <- colnames(mh_p_abcl)
abcl_vars <- intersect(
  available_abcl_cols, 
  c("src_subject_id", "eventname",
    "abcl_scr_sub_use_tobacco_t",
    "abcl_scr_sub_use_alcohol_t")
)

# Filter mh_p_abcl
mh_p_abcl_filtered <- mh_p_abcl %>%
  filter(eventname == event_of_interest) %>%
  select(all_of(abcl_vars))

# Filter adhd_phenotype --> adhd_filtered
adhd_filtered <- adhd_phenotype %>%
  filter(eventname == event_of_interest) %>%
  select(src_subject_id, eventname, cbcl_scr_dsm5_adhd_t)

# Merge datasets
merged_df <- mh_p_abcl_filtered %>%
  inner_join(adhd_filtered, by=c("src_subject_id","eventname"))

prs_data_cleaned <- prs_data %>%
  select(IID, PRS) %>%
  rename(src_subject_id = IID)

merged_df <- merged_df %>%
  inner_join(prs_data_cleaned, by="src_subject_id")

sex_df <- sex_df %>% 
  select(-FID) %>%
  rename(src_subject_id = IID)

merged_df <- merged_df %>%
  inner_join(sex_df, by="src_subject_id")

# Rename variable for clarity
merged_df <- merged_df %>%
  rename(ADHDScore = cbcl_scr_dsm5_adhd_t)

# Check if Parent Tobacco Use column exists before filtering
if("abcl_scr_sub_use_tobacco_t" %in% colnames(merged_df)) {
    merged_df$ParentTob_z <- scale(merged_df$abcl_scr_sub_use_tobacco_t)[,1]
} else {
    stop("Error: Parent Tobacco Use variable (abcl_scr_sub_use_tobacco_t) is missing from dataset.")
}

# Ensure all required columns exist before filtering out missing values
all_cols <- c("abcl_scr_sub_use_tobacco_t",
              "abcl_scr_sub_use_alcohol_t",
              "PRS",
              "ADHDScore",
              "sex")

merged_df <- merged_df %>%
  filter(!if_any(all_of(all_cols), is.na))

cat("Shape after dropna:", nrow(merged_df), "rows\n")

# Standardize variables
merged_df$PRS_z         <- scale(merged_df$PRS)[,1]

# NOTE: This was originally "Parent ADHD" but references the alcohol column.
# If you really have a separate ADHD measure for the parent, replace with that variable here.
merged_df$ParentADHD_z  <- scale(merged_df$abcl_scr_sub_use_alcohol_t)[,1]  

# We'll define the parent alcohol usage for the interaction term
merged_df$ParentAlc_z   <- scale(merged_df$abcl_scr_sub_use_alcohol_t)[,1]

##################################################################
# RUN HIERARCHICAL REGRESSION MODELS
##################################################################

# Model 1: PRS only
model1 <- lm(ADHDScore ~ ParentADHD_z, data = merged_df)

# Model 2: Add "Parent ADHD" placeholder
model2 <- lm(ADHDScore ~ PRS_z + ParentADHD_z, data = merged_df)

# Model 3: Add sex
model3 <- lm(ADHDScore ~ PRS_z + ParentADHD_z + sex, data = merged_df)

# Model 4: Add Parent Tobacco Use
model4 <- lm(ADHDScore ~ PRS_z + ParentADHD_z + sex + ParentTob_z, data = merged_df)

# Model 5: Add ONLY the interaction of PRS & ParentAlc_z (no main effect for ParentAlc_z)
#   ~ PRS_z + ParentADHD_z + sex + ParentTob_z + (PRS_z * ParentAlc_z)
model5 <- lm(ADHDScore ~ PRS_z + ParentADHD_z + sex + ParentTob_z + PRS_z:ParentAlc_z,
             data = merged_df)

##################################################################
# COLLECT R-SQUARED VALUES
##################################################################
r2_model1 <- summary(model1)$r.squared
r2_model2 <- summary(model2)$r.squared
r2_model3 <- summary(model3)$r.squared
r2_model4 <- summary(model4)$r.squared
r2_model5 <- summary(model5)$r.squared

##################################################################
# PRINT R-SQUARED VALUES AND CHANGES
##################################################################
cat("\n==================================================\n")
cat("R-Squared for Each Model\n")
cat("==================================================\n")
cat("Model 1 (ParentADHD only):                     ", r2_model1, "\n")
cat("Model 2 (+ PRS_z):               ", r2_model2, "\n")
cat("Model 3 (+ sex):                        ", r2_model3, "\n")
cat("Model 4 (+ ParentTob_z):                ", r2_model4, "\n")
cat("Model 5 (+ PRS_x_ParentAlc_z):          ", r2_model5, "\n")

cat("\n==================================================\n")
cat("R-Squared Changes (Cumulative)\n")
cat("==================================================\n")
cat("Change from Model 1 to 2:", (r2_model2 - r2_model1), "\n")
cat("Change from Model 2 to 3:", (r2_model3 - r2_model2), "\n")
cat("Change from Model 3 to 4:", (r2_model4 - r2_model3), "\n")
cat("Change from Model 4 to 5:", (r2_model5 - r2_model4), "\n")

##################################################################
# ANOVA MODEL COMPARISONS
##################################################################
anova_1_2 <- anova(model1, model2)  # Model 1 vs 2
anova_2_3 <- anova(model2, model3)  # Model 2 vs 3
anova_3_4 <- anova(model3, model4)  # Model 3 vs 4
anova_4_5 <- anova(model4, model5)  # Model 4 vs 5

cat("\n==================================================\n")
cat("ANOVA Comparisons\n")
cat("==================================================\n")
cat("Model 1 vs Model 2:\n")
print(anova_1_2)

cat("\nModel 2 vs Model 3:\n")
print(anova_2_3)

cat("\nModel 3 vs Model 4:\n")
print(anova_3_4)

cat("\nModel 4 vs Model 5:\n")
print(anova_4_5)
cat("\n")

