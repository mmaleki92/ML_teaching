import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import shap
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load a dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# # Correlation Analysis
# correlation_matrix = X.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=False)
# plt.title("Correlation Analysis")
# plt.show()

# # Feature Importance in Random Forest
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# feature_importances = pd.DataFrame(rf.feature_importances_,
#                                    index=X_train.columns,
#                                    columns=['importance']).sort_values('importance', ascending=False)

# # Principal Component Analysis (PCA)
# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(X)
# plt.figure()
# plt.scatter(principal_components[:, 0], principal_components[:, 1])
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title("PCA Analysis")
# plt.show()

# # Permutation Importance
# result = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42)
# perm_sorted_idx = result.importances_mean.argsort()
# plt.boxplot(result.importances[perm_sorted_idx].T, vert=False, labels=X_val.columns[perm_sorted_idx])
# plt.title("Permutation Importance")
# plt.show()

# # SHAP Values
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X_train)
# shap.summary_plot(shap_values[1], X_train, plot_type="bar")

# # Recursive Feature Elimination
# model = LogisticRegression(max_iter=10000)
# rfe = RFE(model, 5)  # selecting top 5 features
# fit = rfe.fit(X_train, y_train)
# print("Num Features: %s" % (fit.n_features_))
# print("Selected Features: %s" % (fit.support_))
# print("Feature Ranking: %s" % (fit.ranking_))
