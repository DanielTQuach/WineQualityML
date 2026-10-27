# Wine Quality Machine Learning
for SJSU Machine Learning Graduate Course
NOTE: The course assignment requirement was to find highest possible accuracy. The dataset is highly skewed, so realistically instead of accuracy, an F1 Score would have been better. 10/27/26 no tables or charts yet, not commited yet

# 1. Introduction and Purpose
The goal of this repository is to document the goal of attempting to achieve the highest possible accuracy model training on the provided wine_quality.csv dataset.

## 1.1 Dataset
The wine_quality.csv dataset consists of 5,000 wines and 12 columns with no missing values and no duplicate rows. Eleven of those columns contain numeric chemical measurements which are acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates, and alcohol. The target is quality, labeled in five classes.

The dataset was also analyzed and based on class representation and it was found that 83% of the dataset consisted of medium quality wines while only a small minority consisting of 1% of wines were high in quality. The overall count of quality is as follows: 163 low, 2,151 medium-low quality, 2,016 medium, 620 medium-high, and 50 high quality.

# 2. Methodology
Before tuning a model, several baseline models were used to determine the ideal model or models to tune further. While finding the baseline models, a train test split of 80/20 was used. In addition, each model used the default scikit-learn defaults.

The first model used was a baseline null classifier model used to determine if any machine learning model was necessary. The result of this initial baseline model with test accuracy of 0.430 and with a balanced accuracy of 0.200.

- The second model used was a random forest model using scikit-learn defaults which resulted in a 0.450 test accuracy.

- The third model used was SVM using scikit-learn defaults which resulted in an accuracy of 0.454 and a balanced accuracy of 0.216.

- The final model used was XGBoost using scikit-learn defaults resulting in a baseline accuracy of 0.412 and a balanced accuracy of 0.199.

All models are in relative proximity to the null classifier model, so they will all be tuned.

## 2.1 Preprocessing Methods
Before tuning these models the data from the dataset will be preprocessed. For the SVM model the wine quality labels will be transformed using one-hot-encoding to be represented using binaries instead of the 5 text representations of quality. This preprocessing method was applied to only SVM as tree-based models do not need one-hot encoding.

## 2.2 Training Methods and Overfitting
All models were trained using 5-fold cross validation.

### 2.2.1 Random Forests Model
For the random forest model, the hyperparameters tuned consisted of the class_weight, min_samples_leaf, max_depth, max_features, and n_estimators. The parameters which were independent alone were trained with and evaluated alone while those parameters which interacted together were combined and evaluated jointly. During the training process, keeping the diversity of the forest and reducing the complexity of trees was the strategy used to combat overfitting.

### 2.2.2 XGBoost 
For the XGBoost model, the hyperparameters tuned consisted of the learning rate, number of estimators, max depth, and the minimum child weight. The parameters which were independent alone were trained with and evaluated alone while those parameters which interacted together were combined and evaluated jointly. In order to reduce overfitting, this model also opted for using 5-fold cross validation. Moreover, overfitting was reduced by making each tree simpler and each boosting step smaller.

### 2.2.3 SVM
For the SVM model, the hyperparameters tuned consisted of the kernel, c, gamma, and class_weight. The parameters which were independent alone were trained with and evaluated alone while those parameters which interacted together were combined and evaluated jointly. In order to reduce overfitting, this model also opted for using 5-fold cross validation. Moreover, the model complexity was reduced if overfitting occurred by adjusting the c and gamma hyperparameters. 

# 3. Experiments and Results
Refer to document report for figures.
- Random Forest - max_depth=20, max_sample_leaf=2, class_weight=balanced_subsample, num_trees=100 resulting in accuracy of 0.439
- XGBoost - max_dept=4, min_child_weight=10, learning_rate=0.05, and n_estimators=100 resulting in accuracy of 4.34
- SVM - C=0.01, no gamma, resulting in accuracy of 0.430

# 5. Conclusion
From these experiments, the random forest was the model with highest accuracy which also surpassed the nullclassifier model, with a 5-fold training accuracy of 0.439 at 100 trees, leaf size 2, depth 20, 2 features per split, and balanced subsample weights.
