"""
Model Evaluation Module for Student Performance Classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}

    def evaluate_single_model(self, model, X_test, y_test, model_name, 
                            target_encoder=None, average='weighted'):
        """Evaluate a single model on test data"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average=average, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

            # Store results
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred
            }

            self.evaluation_results[model_name] = results

            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                       f"F1-Score: {f1:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return None

    def evaluate_all_models(self, models, X_test, y_test, target_encoder=None):
        """Evaluate all models"""
        for model_name, model in models.items():
            self.evaluate_single_model(model, X_test, y_test, model_name, target_encoder)

    def create_confusion_matrix(self, y_true, y_pred, model_name, 
                              target_encoder=None, save_path=None):
        """Create and display confusion matrix"""
        plt.figure(figsize=(8, 6))

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Get class labels
        if target_encoder is not None:
            labels = target_encoder.classes_
        else:
            labels = np.unique(np.concatenate([y_true, y_pred]))

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/confusion_matrix_{model_name.replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')

        plt.show()

        return cm

    def plot_all_confusion_matrices(self, y_test, target_encoder=None, save_path=None):
        """Plot confusion matrices for all evaluated models"""
        n_models = len(self.evaluation_results)
        if n_models == 0:
            logger.warning("No evaluation results found")
            return

        # Calculate grid dimensions
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        # Get class labels
        if target_encoder is not None:
            labels = target_encoder.classes_
        else:
            labels = np.unique(y_test)

        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            row = idx // cols
            col = idx % cols

            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]

            # Create confusion matrix
            cm = confusion_matrix(y_test, results['predictions'])

            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f'{model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/all_confusion_matrices.png", 
                       dpi=300, bbox_inches='tight')

        plt.show()

    def create_comparison_table(self):
        """Create a comparison table of all model results"""
        if not self.evaluation_results:
            logger.warning("No evaluation results found")
            return None

        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'], 
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

        return comparison_df

    def plot_performance_comparison(self, save_path=None):
        """Plot performance comparison across models"""
        comparison_df = self.create_comparison_table()
        if comparison_df is None:
            return

        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # Create bar plot
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=colors[idx], alpha=0.7)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/performance_comparison.png", 
                       dpi=300, bbox_inches='tight')

        plt.show()

    def get_best_model(self, metric='f1_score'):
        """Get the best performing model based on specified metric"""
        if not self.evaluation_results:
            return None, None

        best_model_name = max(self.evaluation_results.keys(),
                            key=lambda x: self.evaluation_results[x][metric])
        best_score = self.evaluation_results[best_model_name][metric]

        return best_model_name, best_score

    def print_detailed_report(self, y_test, target_encoder=None):
        """Print detailed classification report for each model"""
        if target_encoder is not None:
            target_names = target_encoder.classes_
        else:
            target_names = None

        for model_name, results in self.evaluation_results.items():
            print(f"\n{'='*50}")
            print(f"Detailed Classification Report - {model_name}")
            print(f"{'='*50}")

            report = classification_report(y_test, results['predictions'],
                                         target_names=target_names)
            print(report)

def main():
    """Test the evaluation pipeline"""
    from preprocess import DataPreprocessor
    from train import ModelTrainer

    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_data('../data/student-por.csv')

    # Train models
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models(data['X_train'], data['y_train'])

    # Evaluate models
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(trained_models, data['X_test'], data['y_test'], 
                                 data['target_encoder'])

    # Create comparison table
    comparison_table = evaluator.create_comparison_table()
    print("\nModel Performance Comparison:")
    print(comparison_table)

    # Get best model
    best_model, best_score = evaluator.get_best_model()
    print(f"\nBest model: {best_model} with F1-Score: {best_score:.4f}")

    # Plot comparisons
    evaluator.plot_performance_comparison()
    evaluator.plot_all_confusion_matrices(data['y_test'], data['target_encoder'])

if __name__ == "__main__":
    main()
