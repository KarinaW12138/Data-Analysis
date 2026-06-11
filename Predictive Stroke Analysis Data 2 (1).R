
# 1. Load the data-handling packages
library(ggplot2)
library(dplyr)
library(randomForest)
library(pROC)
library(PRROC)
library(caret)

# 2. Read the local file directly from your folder
clean_data <- read.csv("~/Downloads/stroke.csv", stringsAsFactors = FALSE)

# 3. Clean the column names to make sure they are fully readable
clean_data <- clean_data %>% rename_with(tolower)

# 4. Check the size of the dataset to verify it loaded completely
dim(clean_data)

# 1. Convert BMI from text to numbers (this turns any "N/A" text into real missing values)
clean_data$bmi <- as.numeric(clean_data$bmi)

# 2. Calculate the median BMI of the dataset, skipping over the missing spots
median_bmi <- median(clean_data$bmi, na.rm = TRUE)

# 3. Fill in those missing BMI spots with our calculated median value
clean_data$bmi[is.na(clean_data$bmi)] <- median_bmi

# 4. Turn the categorical health and lifestyle columns into factor levels
clean_data <- clean_data %>%
  mutate(
    stroke            = as.factor(stroke),            # 0 = No Stroke, 1 = Stroke
    hypertension      = as.factor(hypertension),      # 0 = No, 1 = Yes
    heart_disease     = as.factor(heart_disease),     # 0 = No, 1 = Yes
    gender            = as.factor(gender),            
    ever_married      = as.factor(ever_married),      
    work_type         = as.factor(work_type),         
    residence_type    = as.factor(residence_type),    
    smoking_status    = as.factor(smoking_status)     
  )

# 5. Quick check to make sure the target variable factor looks right
table(clean_data$stroke)

# 1. Build the bar chart for class imbalance
plot1 <- ggplot(clean_data, aes(x = stroke, fill = stroke)) +
  geom_bar(color = "black", width = 0.5) +
  # Use professional contrasting colors (Light Gray vs Muted Red)
  scale_fill_manual(values = c("0" = "#D3D3D3", "1" = "#C0392B"), 
                    labels = c("0" = "No Stroke", "1" = "Stroke")) +
  coord_cartesian(ylim = c(0, 5400)) +
  labs(
    title = "Target Variable Distribution: Class Imbalance",
    x = "Patient Stroke Outcome",
    y = "Number of Patient Records"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    panel.grid.major.x = element_blank(), # Remove vertical grid lines
    legend.position = "none" # Hide extra legend since the x-axis says it all
  ) +
  # Put the exact raw counts directly on top of the bars
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5, fontface = "bold")

# 2. Show the plot in RStudio
print(plot1)

# 3. Save it as a high-res image for PowerPoint
ggsave("chart1_class_imbalance.png", plot = plot1, width = 6, height = 4.5, dpi = 300)

# 1. Build a boxplot comparing age distributions
plot2 <- ggplot(clean_data, aes(x = stroke, y = age, fill = stroke)) +
  geom_boxplot(color = "black", alpha = 0.8, width = 0.5) +
  scale_fill_manual(values = c("0" = "#D3D3D3", "1" = "#C0392B")) +
  scale_x_discrete(labels = c("0" = "No Stroke", "1" = "Stroke")) +
  labs(
    title = "Age Distribution by Stroke Outcome",
    x = "Patient Group",
    y = "Age (Years)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    panel.grid.major.x = element_blank(),
    legend.position = "none"
  )

# 2. Show the plot
print(plot2)

# 3. Save it as a high-res image
ggsave("chart2_age_boxplot.png", plot = plot2, width = 6, height = 4.5, dpi = 300)

# 1. Build a 100% stacked bar chart for hypertension relative proportions
plot3 <- ggplot(clean_data, aes(x = hypertension, fill = stroke)) +
  geom_bar(position = "fill", color = "black", width = 0.5) +
  scale_fill_manual(values = c("0" = "#D3D3D3", "1" = "#C0392B"), 
                    labels = c("No Stroke", "Stroke")) +
  scale_x_discrete(labels = c("0" = "Normal Blood Pressure", "1" = "Hypertension")) +
  scale_y_continuous(labels = scales::percent) + # Converts decimal scale to clean percentages
  labs(
    title = "Stroke Proportions by Hypertension Status",
    x = "Pre-existing Medical Condition",
    y = "Percentage of Patient Group",
    fill = "Outcome"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    panel.grid.major.x = element_blank(),
    legend.position = "right" # Keep the legend here to define the gray vs red sections
  )

# 2. Show the plot
print(plot3)

# 3. Save it as a high-res image
ggsave("chart3_hypertension_percentages.png", plot = plot3, width = 6.5, height = 4.5, dpi = 300)
# logistic regression and random forest section

# load the modeling packages
library(randomForest)
library(pROC)

# make sure the stroke variable is in the right order
clean_data$stroke <- factor(clean_data$stroke, levels = c("0", "1"))

# set seed so we get the same random split each time
set.seed(123)

# stratified train test split because stroke is really unbalanced
stroke0 <- clean_data[clean_data$stroke == "0", ]
stroke1 <- clean_data[clean_data$stroke == "1", ]

train0 <- sample(1:nrow(stroke0), 0.70 * nrow(stroke0))
train1 <- sample(1:nrow(stroke1), 0.70 * nrow(stroke1))

train_data <- rbind(stroke0[train0, ], stroke1[train1, ])
test_data  <- rbind(stroke0[-train0, ], stroke1[-train1, ])

# check that both train and test still have stroke cases
table(train_data$stroke)
table(test_data$stroke)

# build the logistic regression model
log_model <- glm(
  stroke ~ age + hypertension + heart_disease + avg_glucose_level + bmi +
    gender + ever_married + work_type + residence_type + smoking_status,
  data = train_data,
  family = binomial
)

# look at the logistic regression results
summary(log_model)

# predicted probabilities from logistic regression
log_prob <- predict(log_model, newdata = test_data, type = "response")

# convert probabilities into 0 or 1 predictions
# since the stroke class is small, 0.50 may be too strict, but we start with it as the baseline
log_pred <- ifelse(log_prob > 0.50, "1", "0")
log_pred <- factor(log_pred, levels = c("0", "1"))

# confusion matrix for logistic regression
table(log_pred, test_data$stroke)

# accuracy for logistic regression
log_acc <- mean(log_pred == test_data$stroke)
log_acc

# auc for logistic regression
log_roc <- roc(test_data$stroke, log_prob)
log_auc <- auc(log_roc)
log_auc

# plot roc curve for logistic regression
plot(log_roc, main = "ROC Curve for Logistic Regression")

# build the random forest model
set.seed(123)

rf_model <- randomForest(
  stroke ~ age + hypertension + heart_disease + avg_glucose_level + bmi +
    gender + ever_married + work_type + residence_type + smoking_status,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# look at the random forest model
rf_model

# variable importance from random forest
importance(rf_model)
varImpPlot(rf_model)

# predicted probabilities from random forest
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]

# convert probabilities into 0 or 1 predictions
rf_pred <- ifelse(rf_prob > 0.50, "1", "0")
rf_pred <- factor(rf_pred, levels = c("0", "1"))

# confusion matrix for random forest
table(rf_pred, test_data$stroke)

# accuracy for random forest
rf_acc <- mean(rf_pred == test_data$stroke)
rf_acc

# auc for random forest
rf_roc <- roc(test_data$stroke, rf_prob)
rf_auc <- auc(rf_roc)
rf_auc

# plot roc curve for random forest
plot(rf_roc, main = "ROC Curve for Random Forest")

# compare the two models
model_results <- data.frame(
  model = c("logistic regression", "random forest"),
  accuracy = c(log_acc, rf_acc),
  auc = c(as.numeric(log_auc), as.numeric(rf_auc))
)

model_results

# make a data frame with the predicted probabilities
predicted_probs <- data.frame(
  actual_stroke = test_data$stroke,
  logistic_probability = log_prob,
  random_forest_probability = rf_prob
)

# look at the first few predicted probabilities
head(predicted_probs)

# sort by highest predicted stroke risk from logistic regression
predicted_probs_log_sorted <- predicted_probs[order(-predicted_probs$logistic_probability), ]
head(predicted_probs_log_sorted, 10)

# sort by highest predicted stroke risk from random forest
predicted_probs_rf_sorted <- predicted_probs[order(-predicted_probs$random_forest_probability), ]
head(predicted_probs_rf_sorted, 10)

# optional lower cutoff because only about 5 percent of the dataset had strokes
# this helps us see more possible high risk patients instead of only using 0.50
log_pred_10 <- ifelse(log_prob > 0.10, "1", "0")
log_pred_10 <- as.factor(log_pred_10)

rf_pred_10 <- ifelse(rf_prob > 0.10, "1", "0")
rf_pred_10 <- as.factor(rf_pred_10)

# confusion matrix using the lower cutoff
table(log_pred_10, test_data$stroke)
table(rf_pred_10, test_data$stroke)
# the logistic regression is our baseline model because stroke is a yes or no outcome
# the predicted probabilities tell us each patients estimated chance of having a stroke
# random forest is added because it can catch more complicated patterns between the predictors
# since the data is unbalanced, accuracy alone is not enough
# auc is better here because it shows how well the model separates stroke and no stroke cases
# the 0.50 cutoff is strict, so the 0.10 cutoff is also checked because stroke cases are rare


# Model Performance Evaluation
actual <- as.numeric(as.character(test_data$stroke)) 

metrics <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(log_acc, rf_acc),
  AUC_ROC = c(as.numeric(log_auc), as.numeric(rf_auc)),
  PR_AUC = c(
    pr.curve(scores.class0 = log_prob[actual == 1], 
             scores.class1 = log_prob[actual == 0], curve = FALSE)$auc.integral,
    pr.curve(scores.class0 = rf_prob[actual == 1], 
             scores.class1 = rf_prob[actual == 0], curve = FALSE)$auc.integral
  )
)

calc_metrics <- function(pred, actual) {
  pred_factor <- factor(pred, levels = c("0", "1"))
  actual_factor <- factor(actual, levels = c("0", "1"))
  
  cm <- confusionMatrix(pred_factor, actual_factor, positive = "1", mode = "prec_recall")
  
  c(Precision = cm$byClass["Precision"],
    Recall    = cm$byClass["Recall"],
    F1        = cm$byClass["F1"])
}

metrics$Precision_0.5 <- c(
  calc_metrics(log_pred, actual)["Precision"],
  calc_metrics(rf_pred, actual)["Precision"]
)
metrics$Recall_0.5 <- c(
  calc_metrics(log_pred, actual)["Recall"],
  calc_metrics(rf_pred, actual)["Recall"]
)
metrics$F1_0.5 <- c(
  calc_metrics(log_pred, actual)["F1"],
  calc_metrics(rf_pred, actual)["F1"]
)

print("--- Model Comparison Table ---")
print(metrics)

pr_log <- pr.curve(scores.class0 = log_prob[actual == 1], 
                   scores.class1 = log_prob[actual == 0], curve = TRUE)
pr_rf  <- pr.curve(scores.class0 = rf_prob[actual == 1], 
                   scores.class1 = rf_prob[actual == 0], curve = TRUE)

png("~/Downloads/PR_Curves.png", width = 8, height = 6, units = "in", res = 300)
plot(pr_log, main = "Precision-Recall Curves", col = "blue", lwd = 2)
plot(pr_rf, add = TRUE, col = "red", lwd = 2)
legend("bottomleft", legend = c("Logistic", "Random Forest"), col = c("blue", "red"), lwd = 2)
dev.off()

png("~/Downloads/RF_Variable_Importance.png", width = 8, height = 6, units = "in", res = 300)
varImpPlot(rf_model, main = "Random Forest Variable Importance")
dev.off()
