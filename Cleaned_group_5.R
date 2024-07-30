#####################
### GROUP PROJECT ###
#####################

#####################
# LOAD DATA
#####################
library(readr)
library(base)
library(tidyverse)
library(dplyr)
library(pROC)
library(MASS)
library(caret)
library(randomForest)
library(gbm)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(glmnet)
library(ISLR2)
library(MLmetrics)
library(MLeval)
library(dplyr)
library(ggplot2)




df_all <- read_csv("telco_data.csv")
attach(df_all)
df_all %>% glimpse()




#####################################
####### Cleaning and Feature Building
#####################################

#Check if target variable has no missing values
summary(is.na(df_all$Churn))


#Convert chr types to factors.
df_all <- df_all %>% 
  mutate(Churn = factor(Churn, levels=c("Yes","No")),
         gender = as.factor(gender),
         SeniorCitizen = as.factor(SeniorCitizen),
         Partner = as.factor(Partner),
         Dependents = as.factor(Dependents),
         PhoneService = as.factor(PhoneService),
         MultipleLines = as.factor(MultipleLines),
         InternetService = as.factor(InternetService),
         OnlineSecurity = as.factor(OnlineSecurity),
         OnlineBackup = as.factor(OnlineBackup),
         DeviceProtection = as.factor(DeviceProtection),
         TechSupport = as.factor(TechSupport),
         StreamingTV = as.factor(StreamingTV),
         StreamingMovies = as.factor(StreamingMovies),
         Contract = as.factor(Contract),
         PaperlessBilling = as.factor(PaperlessBilling),
         PaymentMethod = as.factor(PaymentMethod),
         #Fix missing values in column
         TotalCharges = ifelse(is.na(TotalCharges), MonthlyCharges, TotalCharges),
         #New Columns
         add_ons = as.factor(case_when(
           `InternetService` == "No" ~ "No internet service",
           `OnlineSecurity` == "Yes" | `OnlineBackup` == "Yes" | `DeviceProtection` == "Yes" | 
             `TechSupport` == "Yes" | `StreamingTV` == "Yes" | `StreamingMovies` == "Yes" ~ "Yes",
           .default = "No")),
         Ratio = MonthlyCharges/TotalCharges)


#####################
# EDA
#####################
tel_tib<-tibble(df_all)
#Univariate Analysis

tel_tib %>%
  count(Churn) %>%
  mutate(proportion = n / sum(n))


tel_tib %>%
  count(MultipleLines) %>%
  mutate(proportion = n / sum(n))

tel_tib %>%
  count(InternetService) %>%
  mutate(proportion = n / sum(n))

tel_tib %>%
  count(OnlineSecurity) %>%
  mutate(proportion = n / sum(n))

tel_tib %>%
  count(OnlineBackup) %>%
  mutate(proportion = n / sum(n))

tel_tib %>%
  count(DeviceProtection) %>%
  mutate(proportion = n / sum(n))

tel_tib %>%
  count(TechSupport) %>%
  mutate(proportion = n / sum(n))

tel_tib %>%
  count(StreamingMovies) %>%
  mutate(proportion = n / sum(n))

tel_tib %>%
  count(StreamingTV) %>%
  mutate(proportion = n / sum(n))


#Bivariate Analysis

tel_tib %>%
  count( Churn, MultipleLines)%>%
  group_by(MultipleLines) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()%>%
  dplyr::select(-n) %>% # Remove the count column 
  spread(key = Churn, value = proportion)


tel_tib %>%
  count( Churn, OnlineSecurity)%>%
  group_by(OnlineSecurity) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()%>%
  dplyr::select(-n) %>% # Remove the count column 
  spread(key = Churn, value = proportion)

tel_tib %>%
  count( Churn, OnlineBackup)%>%
  group_by(OnlineBackup) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()%>%
  dplyr::select(-n) %>% # Remove the count column 
  spread(key = Churn, value = proportion)

tel_tib %>%
  count( Churn, DeviceProtection)%>%
  group_by(DeviceProtection) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()%>%
  dplyr::select(-n) %>% # Remove the count column 
  spread(key = Churn, value = proportion)

tel_tib %>%
  count( Churn, TechSupport)%>%
  group_by(TechSupport) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()%>%
  dplyr::select(-n) %>% # Remove the count column 
  spread(key = Churn, value = proportion)

tel_tib %>%
  count( Churn, StreamingTV)%>%
  group_by(StreamingTV) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()%>%
  dplyr::select(-n) %>% # Remove the count column 
  spread(key = Churn, value = proportion)

tel_tib %>%
  count( Churn, StreamingMovies)%>%
  group_by(StreamingMovies) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()%>%
  dplyr::select(-n) %>% # Remove the count column 
  spread(key = Churn, value = proportion)


tel_tib %>% dplyr::select(Churn, StreamingTV) %>% plot()
tel_tib %>% dplyr::select(Churn, StreamingMovies) %>% plot()
tel_tib %>% dplyr::select(Churn, Contract) %>% plot()
tel_tib %>% dplyr::select(Churn, PaperlessBilling) %>% plot()
tel_tib %>% dplyr::select(Churn, PaymentMethod) %>% plot()
tel_tib %>% dplyr::select(Churn, MonthlyCharges) %>% plot()
tel_tib %>% dplyr::select(Churn, TotalCharges) %>% plot()
tel_tib %>% dplyr::select(Churn, Ratio) %>% plot()

tel_tib %>%
  count(Churn, InternetService) %>%
  group_by(InternetService) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup() %>%
  dplyr::select(-n) %>% # Remove the 'n' column
  spread(key = Churn, value = proportion)



#############################
####### MODEL BUILDING
#############################

#Remove the CustomerID column
telco <- df_all[,-c(1)]


set.seed(18)

# Hold out 20% of the data as a final validation set
train_ix = createDataPartition(telco$Churn,
                               p = 0.8)

telco_train = telco[train_ix$Resample1,]
telco_test  = telco[-train_ix$Resample1,]

table(telco$Churn[train_ix$Resample1]) %>% 
  prop.table
table(telco$Churn[-train_ix$Resample1]) %>% 
  prop.table

###########################################################################
# Setup cross-validation
###########################################################################


# Number of folds
kcv = 10

cv_folds = createFolds(telco_train$Churn,
                       k = kcv)

my_summary = function(data, lev = NULL, model = NULL) {
  default = defaultSummary(data, lev, model)
  twoclass = twoClassSummary(data, lev, model)
  # Converting to TPR and FPR instead of sens/spec
  twoclass[3] = 1-twoclass[3]
  names(twoclass) = c("AUC_ROC", "TPR", "FPR")
  logloss = mnLogLoss(data, lev, model)
  c(default,twoclass, logloss)
}

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  classProbs = TRUE,
  savePredictions = TRUE,
  summaryFunction = my_summary,
  selectionFunction="oneSE")


###########################################################################
# Logistic
###########################################################################

###########################################################################
# CARET Version # for comparing engineered features across models
###########################################################################
colnames(telco_train)
dim(telco_train)
telco_train

# colnames(telco_train[,-c(22)])
# columns were sequentially dropped to do an a la carte of engineered features and see their impact on logistic fit
glm_model <-train(Churn~., data =  telco_train,
                  method = 'glm',
                  trControl = fit_control,
                  family = 'binomial'
)


print(glm_model)
confusionMatrix(glm_model)

# Based on multiple versions of logistic regression on combination of features, it is evident that the dataset with all features 
# plus the two engg features has superior performance. We use GLMNET implementation of logistic regression because it provides flexibility of tuning regularization paramter

###########################################################################
# GLM Version - Logistic CV fit
###########################################################################


# Convert `Churn` to binary response
y <- as.numeric(telco_train$Churn == "Yes")

# Prepare predictors
X <- model.matrix(Churn ~ ., data = telco_train)[, -1]

#glmnet does not support logloss, using deviance instead. Prefer using that since random forest and gbm will be trained on logLoss
# for a two class classification deviance = -2 sigma(yi log(pi)+ (1-yi)log(1-pi))
# Uses Lasso Regularization
cvfit_all_deviance <- cv.glmnet(X, y, family = "binomial", type.measure = "deviance")
plot(cvfit_all_deviance)
cvfit_all_deviance$lambda.1se
print(cvfit_all_deviance)
saveRDS(cvfit_all_deviance, file = "logistic_glmnet_deviance_opt_24_Jul.rds")

# cvfit_all_deviance <- readRDS("logistic_glmnet_deviance_opt_24_Jul.rds")
# print(cvfit_all_deviance)
optimal_lambda <- cvfit_all_deviance$lambda.1se
optimal_lambda

plot(cvfit_all_deviance)

###########################################################################
# Single tree
###########################################################################

rpart_grid = data.frame(cp = c(0, exp(seq(log(0.00001), log(0.03), length.out=500))))
single_tree_fit <- train( Churn ~ ., data = telco_train, 
                          method = "rpart", 
                          tuneGrid = rpart_grid,
                          trControl = fit_control)


# Access the underlying rpart model
rpart_model <- single_tree_fit$finalModel

# Create the plot
plotcp(rpart_model)

printcp(single_tree_fit)

# Extract the final fit
single_tree_fit$finalModel
rpart.plot(single_tree_fit$finalModel)
single_tree_fit$bestTune
ggplot(single_tree_fit)
plotcp(single_tree_fit)



saveRDS(single_tree_fit, file = "single_tree_26_Jul.rds")
# single_tree_fit <- readRDS("single_tree_26_Jul.rds")
# single_tree_fit
# colnames(single_tree_fit$results)



# Extract results
results <- single_tree_fit$results

min_logLoss <- min(results$logLoss)
one_se_logLoss <- min_logLoss + sd(results$logLoss)  # one standard error rule
results$cp
# Create a plot with ggplot2
ggplot(data = results, aes(x = cp, y = logLoss)) +
  geom_line() +
  # geom_vline(xintercept = results$cp[which.min(abs(results$logLoss - one_se_logLoss))],
  #            linetype = "dashed", color = "red") +
  labs(title = "logLoss vs Complexity Parameter (cp)",
       subtitle = "One-SE Rule",
       x = "Complexity Parameter (cp)",
       y = "logLoss") +
  theme_minimal()


## Using rpart for visualization of how error changes as size of tree changes

bigtree = rpart(Churn ~ ., data = telco_train,
                control = rpart.control(cp=0.0007370075, minsplit=5))

confusionMatrix(bigtree)

printcp(bigtree)
plotcp(bigtree)
bigtree$parms
round(prop.table(confusionMatrix(single_tree_fit$pred$pred,single_tree_fit$pred$obs)$table) * 100, 1)


###########################################################################
# Random Forest
###########################################################################
colnames(telco_train)
glimpse(telco_train)

rf_grid = data.frame(mtry = c(2,5,13,20))
rf_fit <- train( Churn ~ ., data = telco_train, 
                 method = "rf", 
                 trControl = fit_control,
                 tuneGrid = rf_grid,
                 metric = "logLoss",  
                 ntree = 1000)


# Getting a plot of CV error estimates
ggplot(rf_fit)

# Adding +/- one se
rf_fit$results
best = rf_fit$results[which.min(rf_fit$results$logLoss),]
onesd = best$RMSE + best$logLossSD/sqrt(kcv)

ggplot(rf_fit) + 
  geom_segment(aes(x=mtry, 
                   xend=mtry, 
                   y=logLoss-logLossSD/sqrt(kcv), 
                   yend=logLoss+logLossSD/sqrt(kcv)), 
               data=rf_fit$results) + 
  geom_hline(yintercept = onesd, linetype='dotted')

## add a plot for logLoss against num Trees

print(rf_fit)
confusionMatrix(rf_fit)
saveRDS(rf_fit, file = "random_forest_24_Jul.rds")
# rf_fit <- readRDS("random_forest_24_Jul.rds")
print(rf_fit)

###########################################################################
# Boosting
###########################################################################

gbm_grid <-  expand.grid(interaction.depth = 15, 
                         n.trees = 1000, 
                         shrinkage =  0.3,
                         n.minobsinnode = 10)


gbmfit <- train(Churn ~ ., data = telco_train, 
                method = "gbm", 
                trControl = fit_control,
                tuneGrid = gbm_grid,
                metric = "logLoss",         
                verbose = FALSE)

print(gbmfit$bestTune)

print(gbmfit$finalModel)

plot(gbmfit)
saveRDS(gbmfit, file = "gbm_26_Jul.rds")

write.csv(gbmfit$results, file = "gbmfit_results_selected_model.csv", row.names = FALSE)


# Extracting performance summaries
# Confusion matrix as proportions, not counts, since 
# the test dataset varies across folds
# These are CV estimates of error rates/accuracy using a *default* cutoff
# to classify cases

confusionMatrix(gbmfit)

thresholder(gbmfit, 
            threshold = 0.5, 
            final = TRUE,
            statistics = c("Sensitivity",
                           "Specificity"))

gbmfit_res = thresholder(gbmfit, 
                         threshold = seq(0, 1, by = 0.01), 
                         final = TRUE)


# How do metrics vary with the threshold?
pldf = gbmfit_res %>%
  mutate(TPR=Sensitivity, FPR = 1-Specificity, FNR = 1-Sensitivity) %>%
  dplyr::select(-c(n.trees, interaction.depth, shrinkage, n.minobsinnode)) %>%
  pivot_longer(-prob_threshold) 

ggplot(aes(x=prob_threshold, y=value, color=name), 
       data=pldf %>% filter(name %in% c("TPR", "FPR"))) + 
  geom_line() 

ggplot(aes(x=prob_threshold, y=value, color=name), 
       data=pldf %>% filter(name %in% c("FNR", "FPR"))) + 
  geom_line() 


thres = 0.1
tp = gbmfit_res %>% 
  dplyr::filter(prob_threshold==thres) %>% 
  dplyr::select(prob_threshold, Sensitivity, Specificity) %>%
  mutate(TPR=Sensitivity, FPR = 1-Specificity)

ggplot(aes(x=prob_threshold, y=value, color=name), 
       data=pldf %>% filter(name %in% c("TPR", "FPR"))) + 
  geom_line() + 
  geom_vline(xintercept=thres, lty=2) + 
  geom_point(aes(x=prob_threshold, y=TPR, color=NULL), data=tp) + 
  geom_point(aes(x=prob_threshold, y=FPR, color=NULL), data=tp) 

###########################################################################
# Performance Metrics
###########################################################################

###########################################################################
# Comparison on CV Folds for model selection
###########################################################################

# Log
log_res = thresholder(glm_model, 
                         threshold = seq(0, 1, by = 0.01), 
                         final = TRUE)
# Tree
single_tree_res = thresholder(single_tree_fit, 
                      threshold = seq(0, 1, by = 0.01), 
                      final = TRUE)


# RF

rf_fit_res = thresholder(rf_fit, 
                              threshold = seq(0, 1, by = 0.01), 
                              final = TRUE)


# GBM

gbmfit_res = thresholder(gbmfit, 
                         threshold = seq(0, 1, by = 0.01), 
                         final = TRUE)


# ROC curve

# Assuming gbmfit_res and rf_fit_res are your data frames with Specificity and Sensitivity columns
gbmfit_res$model <- 'GBM'
rf_fit_res$model <- 'RF'
log_res$model <- 'Logistic'
single_tree_res$model <- 'Single Tree'

combined_res <- rbind(gbmfit_res, rf_fit_res,log_res,single_tree_res)

# Initialize the ggplot object with the first dataset
ggplot(data = gbmfit_res, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_line(color = "blue") +  # Add line for gbmfit_res with a specific color
  geom_line(data = rf_fit_res, aes(x = 1 - Specificity, y = Sensitivity), color = "red") +
  geom_line(data = log_res, aes(x = 1 - Specificity, y = Sensitivity), color = "green") +
  geom_line(data = single_tree_res, aes(x = 1 - Specificity, y = Sensitivity), color = "brown") +
  ylab("TPR (Sensitivity)") +
  xlab("FPR (1-Specificity)") +
  geom_abline(intercept = 0, slope = 1, linetype = 'dotted') +
  theme_bw() +
  labs(title = "ROC Curves", color = "Model")  # Optional: Add a title and legend
  




###########################################################################
# Comparison on Test data for performance evaluation
###########################################################################


get_metrics = function(threshold, test_probs, true_class, 
                       pos_label, neg_label) {
  # Get class predictions
  pc = factor(ifelse(test_probs[pos_label]>threshold, pos_label, neg_label), levels=c(pos_label, neg_label))
  test_set = data.frame(obs = true_class, pred = pc, test_probs)
  my_summary(test_set, lev=c(pos_label, neg_label))
}

# Convert `Churn` to binary response
y_test <- as.numeric(telco_test$Churn == "Yes")

# Prepare predictors
X_test <- model.matrix(Churn ~ ., data = telco_test)[, -1]

# Assuming `X_test` are the predictor variables for the test set
logistic_pred_prob <- predict(cvfit_all_deviance, newx = X_test, s = "lambda.1se", type = "response")
str(logistic_pred_prob)




logistic_pred_prob <- as.numeric(logistic_pred_prob)
logistic_predictions_df <- data.frame(
  Yes = logistic_pred_prob,           
  No = 1 - logistic_pred_prob         
)

rf_pred_cv_fit <- predict(rf_fit, newdata = telco_test, type = "prob")[,1]

rf_predictions_df <- data.frame(
  Yes = 1 - rf_pred_cv_fit,           
  No = rf_pred_cv_fit         
)


single_tree_pred_prob <- predict(single_tree_fit, newdata = telco_test, type = "prob")
glimpse(single_tree_pred_prob)


gbm_pred_prob <- predict(gbmfit, newdata = telco_test, type = "prob")
glimpse(gbm_pred_prob)

# Print the first few predictions
head(gbm_pred_class)
head(gbm_pred_prob)



# Print the first few predictions
head(single_tree_pred_class)
head(single_tree_pred_prob)


thr_seq = seq(0, 1, length.out=500)


metrics = lapply(thr_seq, function(x) get_metrics(x, logistic_predictions_df, telco_test$Churn, "Yes", "No"))
logistic_metrics_df = data.frame(do.call(rbind, metrics))

metrics = lapply(thr_seq, function(x) get_metrics(x, rf_predictions_df, telco_test$Churn, "Yes", "No"))
rf_metrics_df = data.frame(do.call(rbind, metrics))

metrics = lapply(thr_seq, function(x) get_metrics(x, single_tree_pred_prob, telco_test$Churn, "Yes", "No"))
single_tree_metrics_df = data.frame(do.call(rbind, metrics))

metrics = lapply(thr_seq, function(x) get_metrics(x, gbm_pred_prob, telco_test$Churn, "Yes", "No"))
gbm_metrics_df = data.frame(do.call(rbind, metrics))


dim(rf_metrics_df)
write.csv(rf_metrics_df, file = "rf_metrics_df_holdout.csv", row.names = FALSE)
dim(logistic_metrics_df)

# ROC curve

ggplot(aes(x=FPR, y=TPR), data=rf_metrics_df) + 
  geom_line(color = "red") +  # Add line for gbmfit_res with a specific color
  geom_line(aes(x=FPR, y=TPR), data=logistic_metrics_df, color = "green") +
  geom_line(aes(x=FPR, y=TPR), data=single_tree_metrics_df, color = "brown") +
  geom_line(aes(x=FPR, y=TPR), data=gbm_metrics_df, color = "blue") +
  ylab("TPR (Sensitivity)") + 
  xlab("FPR (1-Specificity)") + 
  geom_abline(intercept=0, slope=1, linetype='dotted') +
  # annotate("text", x=0.65, y=0.25, 
  #          label=paste("AUC RF:",round(rf_metrics_df$AUC_ROC[1], 2))) +
  # annotate("text", x=0.65, y=0.20,
  #          label=paste("AUC Logistic:",round(logistic_metrics_df$AUC_ROC[1], 2))) +
  # annotate("text", x=0.65, y=0.15,
  #          label=paste("AUC Single Tree:",round(single_tree_metrics_df$AUC_ROC[1], 2))) +
  # annotate("text", x=0.65, y=0.10,
  #          label=paste("AUC GBM:",round(gbm_metrics_df$AUC_ROC[1], 2))) +
  theme_bw()
  
glimpse(single_tree_predictions_df)



###########################################################################
# Threshold Identification of the selected model using Youden's J
###########################################################################


rffit_res = thresholder(rf_fit, 
                         threshold = seq(0, 1, by = 0.01), 
                         final = TRUE)


optim_J = rffit_res[which.max(rffit_res$J),]

ggplot(aes(x=prob_threshold, y=J), 
       data=rffit_res) + 
  geom_line() + 
  geom_vline(aes(xintercept=optim_J$prob_threshold), lty=2)

optim_J$prob_threshold

ggplot(aes(x=1-Specificity, y=Sensitivity), data=rffit_res) + 
  geom_line() + 
  ylab("TPR (Sensitivity)") + 
  xlab("FPR (1-Specificity)") + 
  geom_abline(intercept=0, slope=1, linetype='dotted') +
  geom_segment(aes(x=1-Specificity, xend=1-Specificity, y=1-Specificity, yend=Sensitivity), color='darkred', data=optim_J) + 
  theme_bw()

###########################################################################
# Confusion Matrix. -Threshold selection using business intuition, math walkthrough in Excel file available here - 
# https://utexas-my.sharepoint.com/:x:/g/personal/as235548_my_utexas_edu/EeFg0I03sR5Kirl5q6EfykABhGrcoa-ElGpM9389_b6TMQ?e=D6eHsR
###########################################################################
# Threshold obtained from analysis in Excel
rf_class_pred <- ifelse(1 - rf_pred_cv_fit > 0.20, "Yes", "No")

confusion_matrix <- confusionMatrix(factor(rf_class_pred, levels = c("Yes", "No")),
                                    telco_test$Churn,
                                    positive = "Yes")
confusion_matrix


precision <-
  confusion_matrix$table[1,1]/(confusion_matrix$table[1,1]+confusion_matrix$table[2,1])

recall <- confusion_matrix$table[1,1]/
  (confusion_matrix$table[1,1]+confusion_matrix$table[1,2])
print(recall)
print(precision)



###########################################################################
# Variable Importance
###########################################################################

# From caret, for methods that support it
imp = varImp(rf_fit, scale=TRUE)

# Recreating the randomForest importance plot by hand
plot_df = data.frame(variable=rownames(imp$importance),
                     rel_importance = imp$importance$Overall)
ggplot(aes(x=reorder(variable, rel_importance), 
           y=rel_importance), data=plot_df) + 
  geom_point() + 
  ylab("Relative importance (RF)") + 
  xlab("Variable") + 
  coord_flip()

# Same as from the randomForest package directly!
varImp(rf_fit$finalModel, scale=FALSE)
varImpPlot(rf_fit$finalModel)



# Lift

rf_pred_cv_fit2 <- 1 - rf_pred_cv_fit

rf_oos_lift = caret::lift(telco_test$Churn~rf_pred_cv_fit2)

ggplot(rf_oos_lift) + 
  geom_abline(slope=1, linetype='dotted') +
  xlim(c(0, 100)) + 
  theme_bw()

prop.table(telco$Churn)


# Calibration
opt_mtry = rf_fit$bestTune$mtry

best_preds = 
  rf_fit$pred %>% filter(mtry==opt_mtry)


rf_cal = caret::calibration(telco_test$Churn~rf_pred_cv_fit2, 
                             data=best_preds, cuts=10)

ggplot(rf_cal) + theme_bw()
