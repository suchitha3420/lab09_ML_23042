import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, classification_report
import lime.lime_tabular
import os
import sys

# Load dataset
df = pd.read_csv("DCT_withoutduplicate 6 1 1 1.csv")  # Update your filename here
X = df.drop('LABEL', axis=1)
y = df['LABEL']
feature_names = X.columns.tolist()
class_names = sorted([str(c) for c in y.unique()])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Label balance
print("\n🔍 Label Distribution in y_test:")
print(y_test.value_counts())

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=7, n_jobs=-1, random_state=42)),
    ('hgb', HistGradientBoostingClassifier(max_iter=50, early_stopping=True, random_state=42)),
    ('svm', LinearSVC(dual=False, max_iter=10000, random_state=42))
]

# Meta model
meta_model = LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs', random_state=42)

# Function to create stacking classifier
def create_stacking_classifier(base_models, meta_model, X_train, y_train, X_test, y_test):
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3,
        n_jobs=-1,
        stack_method='auto'
    )

    print("\n🚀 Training stacking classifier...")
    start_time = time.time()
    stack_model.fit(X_train, y_train)
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")

    y_pred = stack_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return stack_model, y_pred, accuracy

# Function to create pipeline
def create_pipeline(base_models, meta_model):
    return Pipeline([
        ('selector', SelectKBest(k=100)),
        ('scaler', StandardScaler()),
        ('stacking', StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3,
            n_jobs=-1
        ))
    ])

# LIME explanation
def explain_with_lime(pipeline, X_train, X_test, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=False
    )

    # Predict function wrapper to include feature names
    def predict_fn(input_array):
        input_df = pd.DataFrame(input_array, columns=feature_names)
        return pipeline.predict_proba(input_df)

    sample = X_test.iloc[0]
    exp = explainer.explain_instance(
        data_row=sample.values,
        predict_fn=predict_fn,
        num_features=20
    )
    return explainer, exp

# A1: Stacking Classifier
print("\n==============================")
print("🔷 A1: Stacking Classifier")
print("==============================")
stack_model, y_pred, accuracy = create_stacking_classifier(base_models, meta_model, X_train, y_train, X_test, y_test)
print(f"\n🎯 Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\n🔢 Predicted Label Distribution:")
print(pd.Series(y_pred).value_counts())

# A2: Pipeline
print("\n==============================")
print("🔷 A2: Pipeline with Feature Selection")
print("==============================")
pipeline = create_pipeline(base_models, meta_model)
print("🚀 Training pipeline...")
start_time = time.time()
pipeline.fit(X_train, y_train)
print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")

y_pred_pipe = pipeline.predict(X_test)
accuracy_pipe = accuracy_score(y_test, y_pred_pipe)
print(f"\n🎯 Accuracy: {accuracy_pipe:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_pipe, zero_division=0))

# A3: LIME Explanation
print("\n==============================")
print("🔷 A3: LIME Explanation")
print("==============================")
if len(X_test) > 0:
    try:
        explainer, explanation = explain_with_lime(pipeline, X_train, X_test, feature_names, class_names)
        print(f"\n🧠 Explanation for First Test Sample:")
        print(f"Predicted Label: {pipeline.predict(X_test.iloc[0:1])[0]}")
        print(f"Actual Label:    {y_test.iloc[0]}")
        
        if "IPython" in sys.modules:
            explanation.show_in_notebook()
        else:
            explanation.save_to_file('lime_explanation.html')
            print("📁 LIME explanation saved as 'lime_explanation.html'")
    except Exception as e:
        print(f"❌ Error generating LIME explanation: {e}")
else:
    print("⚠️ No test samples available for LIME explanation.")
