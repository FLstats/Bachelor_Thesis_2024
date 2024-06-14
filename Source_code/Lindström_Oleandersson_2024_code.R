setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

library(tidyverse)
library(dplyr)
library(rlang)
library(gridExtra)
library(cowplot)


rm(list = ls())
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                           Read raw data                                     #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# 1. Save the csv file in the same folder as this file
# 2. Run the code on line 1
# 3. Run the code below

matrix.raw <- read.csv("raw_data.csv")


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                             Zero values                                     #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# Check proportion of zeroes
sum(matrix.raw == 0) # 15 210
(sum(matrix.raw == 0) / (nrow(matrix.raw)*ncol(matrix.raw))) |> round(digits=3) # 0.791

# ----------------------------------------- #
###               Remove zeroes           ###
# ----------------------------------------- #
# Remove columns with < 10 non-zero values
cols.to.remove <- names(matrix.raw)[colSums(matrix.raw != 0 ) <= 6]
matrix.m <- matrix.raw[, !(names(matrix.raw) %in% cols.to.remove)]

# Subset matrix.m
glimpse(matrix.m) # "condition" variable is non-numeric
subset.m <- subset(matrix.m, select = -condition)

# Closure operator [Equation 3]
matrix.x <-  subset.m[,] / rowSums(subset.m)
matrix.x[3,] |> sum() # check arbitrary row


# ----------------------------------------- #
###               Impute zeroes           ###
# ----------------------------------------- #
# Multiplicative imputation method [Equation 3]

# Find minimum non-zero value in first column
min.vals.x <- min(matrix.x[matrix.x[,1] > 0, 1])

# Apply to rest of data frame
for (j in 2:ncol(matrix.x)) {
  min.vals.x <- c(min.vals.x, min(matrix.x[matrix.x[,j] > 0, j]))
}

# Define delta (smallest non-zero value across data set)
delta <- min(min.vals.x)


# Create duplicate target matrix
matrix.r <- matrix.x

## Perform imputation
for (i in 1:nrow(matrix.x)) {
  for (j in 1:ncol(matrix.x)) {
    if (matrix.x[i,j] == 0) {
      matrix.r[i,j] <- delta
    }
    
    else {
      sum.term <- sum(delta * ifelse(matrix.x[i,] == 0, 1, 0))
      matrix.r[i,j] <- matrix.x[i,j] * (1-sum.term)
    }
  }
}

# Check arbitrary row for unit-sum constraint
matrix.r[42,] |> sum()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                           Centred Log-Ratios                                #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
library(easyCODA)

# Transform matrix.r to CLR-scale 
matrix.y <- as.data.frame(CLR(matrix.r, weight = FALSE)$LR)



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                           Permanova on CLRs                                 #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
library(vegan)

# ----------------------------------------- #
###             PERMANOVA adonis2         ###
# ----------------------------------------- #
# Add labels to CLR-matrix
matrix.y$condition <- matrix.raw$condition

set.seed(73)
permanova.output <- adonis2(formula = matrix.y[,-ncol(matrix.y)] ~ condition,
                            data = matrix.y,
                            permutations = 999,
                            method = "euc")

permanova.output

# F Reference value
F.ref <- permanova.output$F[1]

# ----------------------------------------- #
###           PERMANOVA manually          ###
# ----------------------------------------- #
# PERMANOVA is performed manually to be able to create the histogram showing
# the distribution of permuted F-statistics.

# Create cores for parallel programming
library(parallel)
library(doParallel)
cores <- makeCluster(detectCores(logical = FALSE)-1)
registerDoParallel(cores)

# Set number of permutations
num.perm <- 999

# Perform PERMANOVA
set.seed(73)
perm.F.stats <- replicate(num.perm, {
  shuffled_data <- matrix.y  # Make a copy of the data
  shuffled_data$condition <- sample(shuffled_data$condition) # Permute label of response
  F.stat <- adonis2(shuffled_data[,-ncol(shuffled_data)] ~ shuffled_data$condition, method = "euc")$F[1]  # Extract F-statistic
  return(F.stat)
})

## [Figure 4]
## Histogram of pseudo F-statistics
# Reference value for F-original is taken from the adonis2-output
ggplot() + 
  geom_histogram(aes(perm.F.stats), bins = 40, color = "black", fill = "gray") +
  theme_classic() + 
  labs(x = "F-statistic", y = "Frequency") +
  theme(panel.border = element_rect(colour = "black", fill = NA)) +
  geom_segment(aes(x = 1.6805, xend = 1.6805, y = 0, yend = 80), linetype = "dashed") +
  theme(axis.title.y = element_text(vjust = + 4))
  

## Permutation-based p-value
df_perm.F.stats <- data.frame(perm.F.stats)
df_perm.F.stats$numerator <- ifelse(df_perm.F.stats$perm.F.stats >= F.ref, 1, 0)
p.value <- sum(df_perm.F.stats$numerator) / num.perm # 0.006006 (999)
p.value


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                   Lasso-penalised logistic regression                       #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
library(glmnet)

# Add labels to CLR-matrix
matrix.y$condition <- matrix.raw$condition

set.seed(73)
n = 59 # Define number of folds

# Specify logistic regression model
cv.lasso <- cv.glmnet(x = as.matrix(matrix.y[,-ncol(matrix.y)]), y = matrix.y[,ncol(matrix.y)],
                    family = "binomial",
                    type.measure = "class", 
                    nfolds = n,
                    alpha = 1-1e-05,
                    nlambda = n,
                    grouped = FALSE,
                    keep = TRUE)

# Results
cv.lasso
cv.lasso$lambda.min
cv.lasso$lambda.1se

# [Figure 5]
plot(cv.lasso)

# Extract coefficients of retained variables
lasso.coef <- coef(cv.lasso, s = cv.lasso$lambda.min)
lasso.coef
coef(cv.lasso, s = cv.lasso$lambda.min)@x


# ----------------------------------------- #
###             Confusion matrix          ###
# ----------------------------------------- # 
# Extract log-odds from model
lasso_log.odds <- cv.lasso$fit.preval[,11]

# Convert log-odds to probabilities
pred_lasso_log.odds <- exp(lasso_log.odds) / (1 + exp(lasso_log.odds))

# Binary classification based on cutoff value of 0.5
pred_lasso_labels <- ifelse(pred_lasso_log.odds > 0.5, 1, 0)

# Create confusion matrix table
base::table(matrix.y$condition, pred_lasso_labels)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                               XGBoost                                       #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
library(xgboost)

# Add labels to CLR-matrix
matrix.y$condition <- matrix.raw$condition

# Make labels binary
matrix.y$condition <- ifelse(matrix.y$condition == "parkinson", 1, 0)

# Create dmatrix to be used in xgboost model
Dmatrix.y <- xgb.DMatrix(data = (as.matrix(matrix.y[,-134])), label = (as.matrix(matrix.y[,134])))

# Define parameters in xgboost model
prms <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  eta = 0.05,
  max_depth = 5,
  alpha = 3,
  lambda = 2,
  subsample = 0.75,
  colsample_bytree = 0.65,
  gamma = 0,
  min_child_weight = 0.6
  
)

# Specify xgboost model
set.seed(73)
xgb.cv.model <- xgb.cv(data = Dmatrix.y, params = prms, showsd = FALSE,
                       early_stopping_round = NULL,
                       nfold = 59,
                       nrounds = 100,
                       prediction = TRUE)


# Minimum test error
min(xgb.cv.model$evaluation_log$test_error_mean)

# The iteration where the minimun test error was obtained
best_iteration <- which.min(xgb.cv.model$evaluation_log$test_error_mean) # 100

# Train the final xgboost model using optimal number of rounds
set.seed(73)
final.xgb.model <- xgboost(params = prms,
                           data = Dmatrix.y,
                           nrounds = best_iteration)

# ----------------------------------------- #
###             Confusion matrix          ###
# ----------------------------------------- #
# (for first 100 rounds)

# Extract probability predictions from model results
xgb_pred <- xgb.cv.model$pred

# Binary classification based on cutoff value of 0.5

pred_xgb_labels <- ifelse(xgb_pred >= 0.5, 1, 0)

# Create confusion matrix table
base::table(matrix.y$condition, pred_xgb_labels)

# Manually check error rate: 1- (diagonal / total)
1 - (41/59)


# ----------------------------------------- #
###           Variable importance         ###
# ----------------------------------------- #
library(Ckmeans.1d.dp)

# Remove label
matrix.y <- subset(matrix.y, select = -condition)

# Extract variable importance
importance <- xgb.importance(names(matrix.y), model = final.xgb.model)
importance

# Convert importance data to a data frame and select top 10 most important features
importance_df <- data.frame(importance)
top_features <- head(importance_df[order(importance_df$Gain, decreasing = TRUE), ], 10)

# [Figure 7]
# Plot top 10 most important features in descending order
ggplot(top_features, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "darkgray", width = 0.5) +
  theme_classic() +
  labs(x = "Taxa", y = "Importance") +
  coord_flip() +  # Flip the plot to have horizontal bars
  theme(axis.text.y = element_text(size = 10)) +  # Adjust font size of y-axis labels if needed
  background_grid(major = "xy", minor = "none") +
  theme(
    panel.border = element_rect(colour = "black", fill = NA)
  )


# ----------------------------------------- #
###             Multi-tree plot           ###
# ----------------------------------------- #

# [Figure 6]
xgb.plot.multi.trees(model = final.xgb.model,
                     feature_names = names(matrix.y),
                     features_keep = 3,
                     render = TRUE)



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                           Random Forest                                     #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
library(randomForest)
library(caret)
library(tidyr)

# Define the cross-validation method
train_control <- trainControl(method = "cv", number = 59)  

# Add labels to CLR-matrix
matrix.y$condition <- matrix.raw$condition


# Specify Random Forest model
set.seed(73)
rf.model <- train(condition ~ ., data = matrix.y, method = "rf",
                  trControl = train_control, importance = TRUE, 
                  tuneGrid = data.frame(mtry = c(2,4,6,8,10,12)))

# Print model results
print(rf.model)
best.rf.model <- rf.model$finalModel
print(best.rf.model)


# ----------------------------------------- #
###             OOB error Line plot       ###
# ----------------------------------------- #

# Create long data frame before plotting
error.plot <- data.frame(best.rf.model$err.rate) 
error.plot$trees <- 1:500
colnames(error.plot) <- c("Overall", "Control", "Parkinson", "trees")
df.long <- gather(error.plot, key = "Variable", value = "Value", -trees)

# [Figure 8]
ggplot(df.long, aes(x = trees, y = Value, color = Variable)) +
  geom_line() +
  labs(x = "Number of trees", y = "Error rate") +
  theme_classic() + 
  theme(legend.position = c(0.9, 0.87)) +
  scale_color_brewer(palette = "Dark2") +
  labs(color = "OOB error") +
  geom_segment(x = 420, y = 0.53, xend = 600, yend = 0.53) +
  geom_segment(x = 420, y = 0.53, xend = 420, yend = 0.8) +
  theme(panel.border = element_rect(colour = "black", fill = NA)) +
  theme(axis.title.y = element_text(vjust = 3),
        axis.title.x = element_text(vjust = -1))


# ----------------------------------------- #
###         Variable importance plot      ###
# ----------------------------------------- #

# Create data frame of variable importance before plotting
var_importance <- data.frame(importance(best.rf.model, type = 1))
var_importance_df <- data.frame(Variables = rownames(var_importance),
                                Importance = var_importance$MeanDecreaseAccuracy)

# [Figure 9]
ggplot(var_importance_df[c(24,26,82,17,23,33,95,58,70,9),], 
       aes(y = reorder(Variables, Importance), x = Importance)) +
  geom_bar(stat = "identity", fill = "darkgrey", width = 0.5) +
  labs(title = "",
       y = "Taxa",
       x = "Mean Decrease in Accuracy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1)) +
  theme_classic() +
  background_grid(major = "xy", minor = "none") +
  theme(panel.border = element_rect(colour = "black", fill = NA)) +
  theme(axis.title.y = element_text(vjust = 4),
        axis.title.x = element_text(vjust = -1))


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                           Ternary Plot                                      #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# [Figure 1]
tern <- matrix(data = c(0.1, 0.6, 0.3), nrow = 1, ncol = 3)
colnames(tern) <- c("X", "Y", "Z")

ternaryplot(
  tern,
  prop_size = 1,
  col = "black",
  main = ""
)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#                                                                             #
#                            Violin plots                                     #
#                                                                             #
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# Add labels to CLR-matrix
matrix.y$condition <- matrix.raw$condition

# [Figure 10]
violin.plot <- function(data, y) {
  ggplot(data = matrix.y, aes(x = condition, {{y}}, fill = condition)) +
    geom_violin(show.legend = FALSE, adjust = 0.75, alpha = 0.75, trim = FALSE) +
    stat_summary(fun = median, show.legend = FALSE, geom = "crossbar", width = 1, size = 0.3) +
    stat_summary(fun = mean, show.legend = FALSE, geom = "point", col = "black", size = 2) +
    # geom_dotplot(binaxis = "y", binwidth = 0.5, stackdir = "center", show.legend = FALSE) +
    scale_x_discrete(labels = c("Control", "Parkinson's")) +
    labs(x = "Condition", y = "CLR value") +
    theme_classic() +
    scale_fill_brewer(palette = "Greys") +
    theme(
      axis.text.x = element_text(size = 7.5),
      axis.title.x = element_text(size = 10),
      axis.title.y = element_text(size = 10),
      plot.margin = unit(c(1,1,1,1), "lines"),
      panel.border = element_rect(colour = "black", fill = NA)
  )
}

plot1 <- violin.plot(matrix.y, Prevotella.sp..CAG.520) + 
            ggtitle("Prevotella sp. CAG 520") +
            theme(plot.title = element_text(face = "bold.italic", hjust = 0.5, size = 10)) +
            background_grid(major = "xy", minor = "none") +
            scale_y_continuous(limits = c(-10, 15))

plot2 <- violin.plot(matrix.y, Prevotella.copri) + 
            ggtitle("Prevotella copri") +
            theme(plot.title = element_text(face = "bold.italic", hjust = 0.5, size = 10)) +
            background_grid(major = "xy", minor = "none") +
            scale_y_continuous(limits = c(-10, 15))

plot3 <- violin.plot(matrix.y, Akkermansia.muciniphila) + 
            ggtitle("Akkermansia muciniphila") +
            theme(plot.title = element_text(face = "bold.italic", hjust = 0.5, size = 10)) +
            background_grid(major = "xy", minor = "none")

plot4 <- violin.plot(matrix.y, Butyricimonas.virosa) + 
            ggtitle("Butyricimonas virosa") +
            theme(plot.title = element_text(face = "bold.italic", hjust = 0.5, size = 10)) +
            background_grid(major = "xy", minor = "none") +
            scale_y_continuous(limits = c(-10, 15))

# All plots on same grid
plot_grid(plot1, plot2, plot3, plot4, ncol = 2, nrow = 2)

