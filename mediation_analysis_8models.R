#!/usr/bin/env Rscript

#----------------------------------------------------
# 1. Libraries
#----------------------------------------------------
library(dplyr)
library(lavaan)

#----------------------------------------------------
# 2. Read Slope Data
#----------------------------------------------------
slope_df <- read.csv("final_slopes.csv", header = TRUE, stringsAsFactors = FALSE)

#----------------------------------------------------
# 3. Read & Merge Sex Data
#----------------------------------------------------
sex_df <- read.table("sexphenotypes_original.txt", 
                     header = FALSE, 
                     stringsAsFactors = FALSE)
colnames(sex_df) <- c("FID", "IID", "sex")

sex_df <- sex_df %>%
  select(-FID) %>%
  rename(src_subject_id = IID)

merged_df <- slope_df %>%
  inner_join(sex_df, by = "src_subject_id")

cat("Merged shape with sex data:", nrow(merged_df), "rows\n")

#----------------------------------------------------
# 4. PCA on Parent Substance Use
#----------------------------------------------------
substance_cols <- c(
  "slope_asr_q06_p", "slope_asr_q90_p",
  "slope_abcl_q06_p", "slope_abcl_q90_p", 
  "slope_abcl_q124_p",
  "slope_abcl_scr_sub_use_tobacco_t",
  "slope_abcl_scr_sub_use_alcohol_t",
  "slope_abcl_scr_sub_use_t_mean"
)

sub_data <- merged_df %>% select(all_of(substance_cols))
sub_data_scaled <- scale(sub_data)

pca_fit <- prcomp(sub_data_scaled, center=FALSE, scale.=FALSE)
cat("PCA explained variance ratio (first 3 PCs):\n")
print(summary(pca_fit)$importance[2, 1:3])

merged_df$PC1_substance <- pca_fit$x[,1]

#----------------------------------------------------
# 5. Rename Key Variables
#----------------------------------------------------
merged_df <- merged_df %>%
  rename(
    parentADHD = slope_asr_scr_adhd_t,        # Parent's ADHD
    parentSUB  = PC1_substance,              # Parent's Substance Use (PC1)
    childADHD  = slope_cbcl_scr_dsm5_adhd_t  # Child's ADHD
    # PRS is already "PRS"
    # sex is already "sex"
  )

# Optionally scale PRS
merged_df$PRS_z <- as.numeric(scale(merged_df$PRS))

#----------------------------------------------------
# 6. Drop NAs in Key Variables
#----------------------------------------------------
key_vars <- c("parentADHD", "parentSUB", "childADHD", "PRS_z", "sex")
merged_df <- merged_df %>%
  filter(!if_any(all_of(key_vars), is.na))

cat("Final shape after dropping NAs:", nrow(merged_df), "rows\n")
cat("Number of rows in analysis dataset:", nrow(merged_df), "\n")

############################################################################
#                      1) SERIAL MEDIATION MODELS
#            PRS -> parentSUB -> parentADHD -> childADHD
############################################################################

#----------------------------------------------------
# MODEL 1a: SERIAL (FULL)
#   We INCLUDE direct path from PRS to childADHD,
#   but we do NOT allow a direct path from parentSUB -> childADHD 
#   (the chain is "full": SUB->ADHD->child only).
#
#   childADHD ~ b*parentADHD + d*PRS + sex
#   parentADHD ~ a2*parentSUB + a3*PRS + sex
#   parentSUB ~ a1*PRS + sex
#
#   Indirect effect = a1 * a2 * b
#   Direct effect of PRS->child = d
#----------------------------------------------------
model_1a <- "
  parentSUB  ~ a1*PRS_z + c1*sex
  parentADHD ~ a2*parentSUB + c2*PRS_z + c3*sex
  childADHD  ~ b*parentADHD + d*PRS_z + c4*sex

  indirect := a1*a2*b
"
fit_1a <- sem(model_1a, data=merged_df, estimator="ML")
summary_1a <- summary(fit_1a, fit.measures=TRUE, standardized=TRUE)
cat("\n--- MODEL 1a: SERIAL (FULL) ---\n")
print(summary_1a)


#----------------------------------------------------
# MODEL 1b: SERIAL (PARTIAL)
#   Same as above, but we ADD a direct path from parentSUB->childADHD.
#   That means the chain is partial: parent's SUB influences child both 
#   through parentADHD *and* directly.
#   We still keep PRS->childADHD in the model.
#----------------------------------------------------
model_1b <- "
  parentSUB  ~ a1*PRS_z + c1*sex
  parentADHD ~ a2*parentSUB + c2*PRS_z + c3*sex
  childADHD  ~ b*parentADHD + c*parentSUB + d*PRS_z + c4*sex

  indirect := a1*a2*b
"
fit_1b <- sem(model_1b, data=merged_df, estimator="ML")
summary_1b <- summary(fit_1b, fit.measures=TRUE, standardized=TRUE)
cat("\n--- MODEL 1b: SERIAL (PARTIAL) ---\n")
print(summary_1b)


############################################################################
#               2) SINGLE MEDIATOR = PARENT'S ADHD
#                PRS -> parentADHD -> childADHD
############################################################################

#----------------------------------------------------
# MODEL 2a: SINGLE MEDIATOR (FULL)
#   No direct path from PRS->childADHD
#   So the only route is PRS->parentADHD->childADHD
#----------------------------------------------------
model_2a <- "
  parentADHD ~ a*PRS_z + c1*sex
  childADHD  ~ b*parentADHD + c2*sex

  # Indirect effect
  ab := a*b
"
fit_2a <- sem(model_2a, data=merged_df, estimator="ML")
summary_2a <- summary(fit_2a, fit.measures=TRUE, standardized=TRUE)
cat("\n--- MODEL 2a: SINGLE MEDIATOR ADHD (FULL) ---\n")
print(summary_2a)


#----------------------------------------------------
# MODEL 2b: SINGLE MEDIATOR (PARTIAL)
#   We include a direct path PRS->childADHD
#   plus the mediation path PRS->parentADHD->childADHD
#----------------------------------------------------
model_2b <- "
  parentADHD ~ a*PRS_z + c1*sex
  childADHD  ~ b*parentADHD + d*PRS_z + c2*sex

  # Indirect effect
  ab := a*b
  # Direct = d
  # Total = ab + d
"
fit_2b <- sem(model_2b, data=merged_df, estimator="ML")
summary_2b <- summary(fit_2b, fit.measures=TRUE, standardized=TRUE)
cat("\n--- MODEL 2b: SINGLE MEDIATOR ADHD (PARTIAL) ---\n")
print(summary_2b)


############################################################################
#       3) PARALLEL MEDIATION: parentADHD + parentSUB
#        PRS -> {parentADHD, parentSUB} -> childADHD
############################################################################

#----------------------------------------------------
# MODEL 3a: PARALLEL (FULL)
#   No direct path from PRS->childADHD
#   We only allow the indirect paths:
#     PRS->parentADHD->childADHD
#     PRS->parentSUB->childADHD
#   sex is covariate in mediators + outcome
#----------------------------------------------------
model_3a <- "
  parentADHD ~ a1*PRS_z + c1*sex
  parentSUB  ~ a2*PRS_z + c2*sex

  # No direct path from PRS->childADHD
  childADHD ~ b1*parentADHD + b2*parentSUB + c3*sex

  # Indirects
  ind_ADHD := a1*b1
  ind_SUB  := a2*b2
  ind_total := ind_ADHD + ind_SUB
"
fit_3a <- sem(model_3a, data=merged_df, estimator="ML")
summary_3a <- summary(fit_3a, fit.measures=TRUE, standardized=TRUE)
cat("\n--- MODEL 3a: PARALLEL (FULL) ---\n")
print(summary_3a)


#----------------------------------------------------
# MODEL 3b: PARALLEL (PARTIAL)
#   Add a direct path from PRS->childADHD
#   Everything else is the same as 3a
#----------------------------------------------------
model_3b <- "
  parentADHD ~ a1*PRS_z + c1*sex
  parentSUB  ~ a2*PRS_z + c2*sex

  # Direct path from PRS->childADHD
  childADHD ~ b1*parentADHD + b2*parentSUB + d*PRS_z + c3*sex

  ind_ADHD := a1*b1
  ind_SUB  := a2*b2
  ind_total := ind_ADHD + ind_SUB
"
fit_3b <- sem(model_3b, data=merged_df, estimator="ML")
summary_3b <- summary(fit_3b, fit.measures=TRUE, standardized=TRUE)
cat("\n--- MODEL 3b: PARALLEL (PARTIAL) ---\n")
print(summary_3b)


############################################################################
#           4) SINGLE MEDIATOR = PARENT'S SUBSTANCE USE
#                PRS -> parentSUB -> childADHD
############################################################################

#----------------------------------------------------
# MODEL 4a: SINGLE MEDIATOR SUB (FULL)
#   No direct path from PRS->childADHD
#----------------------------------------------------
model_4a <- "
  parentSUB ~ a*PRS_z + c1*sex
  childADHD ~ b*parentSUB + c2*sex

  ab := a*b
"
fit_4a <- sem(model_4a, data=merged_df, estimator="ML")
summary_4a <- summary(fit_4a, fit.measures=TRUE, standardized=TRUE)
cat("\n--- MODEL 4a: SINGLE MEDIATOR SUB (FULL) ---\n")
print(summary_4a)


#----------------------------------------------------
# MODEL 4b: SINGLE MEDIATOR SUB (PARTIAL)
#   Direct path from PRS->childADHD included
#----------------------------------------------------
model_4b <- "
  parentSUB ~ a*PRS_z + c1*sex
  childADHD ~ b*parentSUB + d*PRS_z + c2*sex

  ab := a*b
"
fit_4b <- sem(model_4b, data=merged_df, estimator="ML")
summary_4b <- summary(fit_4b, fit.measures=TRUE, standardized=TRUE)
cat("\n--- MODEL 4b: SINGLE MEDIATOR SUB (PARTIAL) ---\n")
print(summary_4b)


cat("\nAll eight models have been fit. Summaries are shown above.\n")

