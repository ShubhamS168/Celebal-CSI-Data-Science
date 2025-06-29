"""
Model Training Pipeline for Student Performance Classification
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0

    def initialize_models(self):
        """Initialize all machine learning models"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'Naive Bayes': GaussianNB()
        }
        logger.info(f"Initialized {len(self.models)} models")

    def train_model(self, model, X_train, y_train, model_name):
        """Train a single model"""
        try:
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            logger.info(f"{model_name} trained successfully")
            return model
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return None

    def evaluate_model(self, model, X_train, y_train, model_name, cv_folds=5):
        """Evaluate model using cross-validation"""
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv_folds, scoring='accuracy')

            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()

            logger.info(f"{model_name} - CV Accuracy: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")

            return {
                'cv_scores': cv_scores,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score
            }
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return None

    def train_all_models(self, X_train, y_train):
        """Train and evaluate all models"""
        self.initialize_models()
        trained_models = {}

        for model_name, model in self.models.items():
            # Train model
            trained_model = self.train_model(model, X_train, y_train, model_name)

            if trained_model is not None:
                # Evaluate model
                evaluation_results = self.evaluate_model(
                    trained_model, X_train, y_train, model_name
                )

                if evaluation_results is not None:
                    trained_models[model_name] = trained_model
                    self.results[model_name] = evaluation_results

                    # Track best model based on CV score
                    if evaluation_results['mean_cv_score'] > self.best_score:
                        self.best_score = evaluation_results['mean_cv_score']
                        self.best_model = trained_model
                        self.best_model_name = model_name

        logger.info(f"Training completed. Best model: {self.best_model_name} "
                   f"with CV score: {self.best_score:.4f}")

        return trained_models

    def save_best_model(self, output_path):
        """Save the best performing model"""
        if self.best_model is not None:
            try:
                joblib.dump(self.best_model, output_path)
                logger.info(f"Best model ({self.best_model_name}) saved to {output_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                return False
        else:
            logger.warning("No best model found to save")
            return False

    def get_results_summary(self):
        """Get summary of all model results"""
        if not self.results:
            return None

        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name,
                'CV_Mean_Accuracy': results['mean_cv_score'],
                'CV_Std_Accuracy': results['std_cv_score'],
                'CV_Min_Accuracy': results['cv_scores'].min(),
                'CV_Max_Accuracy': results['cv_scores'].max()
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('CV_Mean_Accuracy', ascending=False)

        return summary_df

def main():
    """Test the training pipeline"""
    from preprocess import DataPreprocessor

    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_data('../data/student-por.csv')

    # Initialize trainer
    trainer = ModelTrainer()

    # Train all models
    trained_models = trainer.train_all_models(data['X_train'], data['y_train'])

    # Get results summary
    results_summary = trainer.get_results_summary()
    print("\nModel Performance Summary:")
    print(results_summary)

    # Save best model
    trainer.save_best_model('../outputs/best_model.pkl')

    print(f"\nBest model: {trainer.best_model_name}")
    print(f"Best CV score: {trainer.best_score:.4f}")

if __name__ == "__main__":
    main()
