"""
Hyperparameter Tuning Module for Student Performance Classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import joblib
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self):
        self.tuned_models = {}
        self.tuning_results = {}
        self.best_models = {}

    def get_param_grids(self):
        """Define parameter grids for each model"""
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            },

            'Decision Tree': {
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },

            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },

            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },

            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },

            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },

            'Naive Bayes': {
                'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
            }
        }

        return param_grids

    def get_models(self):
        """Get base models for tuning"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'Naive Bayes': GaussianNB()
        }

        return models

    def grid_search_tuning(self, X_train, y_train, cv_folds=3, scoring='f1_weighted'):
        """Perform GridSearchCV for all models"""
        models = self.get_models()
        param_grids = self.get_param_grids()

        tuned_models = {}

        for model_name, model in models.items():
            logger.info(f"Starting GridSearchCV for {model_name}...")

            try:
                # Create GridSearchCV
                grid_search = GridSearchCV(
                    model, 
                    param_grids[model_name],
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1
                )

                # Fit grid search
                grid_search.fit(X_train, y_train)

                # Store results
                tuned_models[model_name] = grid_search.best_estimator_
                self.tuning_results[f"{model_name}_grid"] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }

                logger.info(f"{model_name} GridSearchCV completed. "
                           f"Best score: {grid_search.best_score_:.4f}")
                logger.info(f"Best parameters: {grid_search.best_params_}")

            except Exception as e:
                logger.error(f"Error in GridSearchCV for {model_name}: {e}")
                continue

        self.best_models.update(tuned_models)
        return tuned_models

    def randomized_search_tuning(self, X_train, y_train, cv_folds=3, 
                               scoring='f1_weighted', n_iter=50):
        """Perform RandomizedSearchCV for all models"""
        models = self.get_models()
        param_grids = self.get_param_grids()

        tuned_models = {}

        for model_name, model in models.items():
            logger.info(f"Starting RandomizedSearchCV for {model_name}...")

            try:
                # Create RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    model,
                    param_grids[model_name],
                    cv=cv_folds,
                    scoring=scoring,
                    n_iter=n_iter,
                    random_state=42,
                    n_jobs=-1,
                    verbose=1
                )

                # Fit random search
                random_search.fit(X_train, y_train)

                # Store results
                tuned_models[model_name] = random_search.best_estimator_
                self.tuning_results[f"{model_name}_random"] = {
                    'best_params': random_search.best_params_,
                    'best_score': random_search.best_score_,
                    'cv_results': random_search.cv_results_
                }

                logger.info(f"{model_name} RandomizedSearchCV completed. "
                           f"Best score: {random_search.best_score_:.4f}")
                logger.info(f"Best parameters: {random_search.best_params_}")

            except Exception as e:
                logger.error(f"Error in RandomizedSearchCV for {model_name}: {e}")
                continue

        return tuned_models

    def compare_tuning_methods(self):
        """Compare GridSearchCV vs RandomizedSearchCV results"""
        comparison_data = []

        model_names = set()
        for key in self.tuning_results.keys():
            model_name = key.replace('_grid', '').replace('_random', '')
            model_names.add(model_name)

        for model_name in model_names:
            grid_key = f"{model_name}_grid"
            random_key = f"{model_name}_random"

            row = {'Model': model_name}

            if grid_key in self.tuning_results:
                row['GridSearch_Score'] = self.tuning_results[grid_key]['best_score']
                row['GridSearch_Params'] = str(self.tuning_results[grid_key]['best_params'])
            else:
                row['GridSearch_Score'] = None
                row['GridSearch_Params'] = None

            if random_key in self.tuning_results:
                row['RandomSearch_Score'] = self.tuning_results[random_key]['best_score']
                row['RandomSearch_Params'] = str(self.tuning_results[random_key]['best_params'])
            else:
                row['RandomSearch_Score'] = None
                row['RandomSearch_Params'] = None

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df

    def get_tuning_summary(self):
        """Get summary of tuning results"""
        summary_data = []

        for result_key, results in self.tuning_results.items():
            summary_data.append({
                'Model_Method': result_key,
                'Best_Score': results['best_score'],
                'Best_Params': str(results['best_params'])
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Best_Score', ascending=False)

        return summary_df

    def save_best_tuned_model(self, output_path):
        """Save the best tuned model"""
        if not self.tuning_results:
            logger.warning("No tuning results found")
            return False

        # Find best model across all tuning methods
        best_score = 0
        best_model_key = None

        for key, results in self.tuning_results.items():
            if results['best_score'] > best_score:
                best_score = results['best_score']
                best_model_key = key

        if best_model_key is None:
            logger.warning("No best model found")
            return False

        # Get the model name
        model_name = best_model_key.replace('_grid', '').replace('_random', '')

        # Save the model (assuming it's in best_models)
        if model_name in self.best_models:
            try:
                joblib.dump(self.best_models[model_name], output_path)
                logger.info(f"Best tuned model ({best_model_key}) saved to {output_path}")
                logger.info(f"Best score: {best_score:.4f}")
                return True
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                return False
        else:
            logger.warning(f"Model {model_name} not found in best_models")
            return False

def main():
    """Test the hyperparameter tuning pipeline"""
    from preprocess import DataPreprocessor

    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_data('../data/student-por.csv')

    # Initialize tuner
    tuner = HyperparameterTuner()

    # Perform RandomizedSearchCV (faster for testing)
    print("Starting RandomizedSearchCV...")
    tuned_models_random = tuner.randomized_search_tuning(
        data['X_train'], data['y_train'], n_iter=20
    )

    # Get tuning summary
    summary = tuner.get_tuning_summary()
    print("\nHyperparameter Tuning Summary:")
    print(summary)

    # Save best tuned model
    tuner.save_best_tuned_model('../outputs/best_tuned_model.pkl')

if __name__ == "__main__":
    main()
