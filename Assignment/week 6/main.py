"""
Main Entry Point for Student Performance Classification Project
"""

import os
import sys
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import DataPreprocessor
from train import ModelTrainer
from evaluate import ModelEvaluator
from tune import HyperparameterTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self, data_path='data/student-por.csv', output_dir='outputs'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.tuner = HyperparameterTuner()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def run_preprocessing(self):
        """Step 1: Data Preprocessing"""
        logger.info("Starting data preprocessing...")
        self.data = self.preprocessor.preprocess_data(self.data_path)
        logger.info("Data preprocessing completed successfully")
        return self.data

    def run_training(self):
        """Step 2: Model Training"""
        logger.info("Starting model training...")
        self.trained_models = self.trainer.train_all_models(
            self.data['X_train'], self.data['y_train']
        )

        # Save training results
        results_summary = self.trainer.get_results_summary()
        if results_summary is not None:
            results_summary.to_csv(f"{self.output_dir}/training_results.csv", index=False)
            logger.info(f"Training results saved to {self.output_dir}/training_results.csv")

        # Save best model
        self.trainer.save_best_model(f"{self.output_dir}/best_model.pkl")

        logger.info("Model training completed successfully")
        return self.trained_models

    def run_evaluation(self):
        """Step 3: Model Evaluation"""
        logger.info("Starting model evaluation...")

        # Evaluate all models
        self.evaluator.evaluate_all_models(
            self.trained_models, 
            self.data['X_test'], 
            self.data['y_test'],
            self.data['target_encoder']
        )

        # Create comparison table
        comparison_table = self.evaluator.create_comparison_table()
        if comparison_table is not None:
            comparison_table.to_csv(f"{self.output_dir}/evaluation_results.csv", index=False)
            logger.info(f"Evaluation results saved to {self.output_dir}/evaluation_results.csv")

            # Print results
            print("\n" + "="*60)
            print("MODEL PERFORMANCE COMPARISON")
            print("="*60)
            print(comparison_table.to_string(index=False))

        # Get best model
        best_model, best_score = self.evaluator.get_best_model()
        if best_model:
            logger.info(f"Best model: {best_model} with F1-Score: {best_score:.4f}")

        # Generate plots
        logger.info("Generating performance plots...")
        self.evaluator.plot_performance_comparison(save_path=self.output_dir)
        self.evaluator.plot_all_confusion_matrices(
            self.data['y_test'], 
            self.data['target_encoder'],
            save_path=self.output_dir
        )

        # Print detailed reports
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*60)
        self.evaluator.print_detailed_report(
            self.data['y_test'], 
            self.data['target_encoder']
        )

        logger.info("Model evaluation completed successfully")

    def run_hyperparameter_tuning(self, method='random', n_iter=50):
        """Step 4: Hyperparameter Tuning (Optional)"""
        logger.info(f"Starting hyperparameter tuning using {method} search...")

        if method == 'grid':
            tuned_models = self.tuner.grid_search_tuning(
                self.data['X_train'], self.data['y_train']
            )
        elif method == 'random':
            tuned_models = self.tuner.randomized_search_tuning(
                self.data['X_train'], self.data['y_train'], n_iter=n_iter
            )
        else:
            logger.error(f"Unknown tuning method: {method}")
            return

        # Save tuning results
        tuning_summary = self.tuner.get_tuning_summary()
        if tuning_summary is not None:
            tuning_summary.to_csv(f"{self.output_dir}/tuning_results.csv", index=False)
            logger.info(f"Tuning results saved to {self.output_dir}/tuning_results.csv")

            print("\n" + "="*60)
            print("HYPERPARAMETER TUNING RESULTS")
            print("="*60)
            print(tuning_summary.to_string(index=False))

        # Save best tuned model
        self.tuner.save_best_tuned_model(f"{self.output_dir}/best_tuned_model.pkl")

        logger.info("Hyperparameter tuning completed successfully")
        return tuned_models

    def run_complete_pipeline(self, include_tuning=False, tuning_method='random'):
        """Run the complete ML pipeline"""
        start_time = datetime.now()
        logger.info("Starting complete ML pipeline...")

        try:
            # Step 1: Preprocessing
            self.run_preprocessing()

            # Step 2: Training
            self.run_training()

            # Step 3: Evaluation
            self.run_evaluation()

            # Step 4: Hyperparameter Tuning (Optional)
            if include_tuning:
                self.run_hyperparameter_tuning(method=tuning_method)

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info(f"Complete pipeline finished successfully in {duration}")

            print("\n" + "="*60)
            print("PIPELINE SUMMARY")
            print("="*60)
            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")
            print(f"Total duration: {duration}")
            print(f"Output directory: {self.output_dir}")
            print(f"Number of models trained: {len(self.trained_models)}")

            if hasattr(self, 'trainer') and self.trainer.best_model:
                print(f"Best model: {self.trainer.best_model_name}")
                print(f"Best CV score: {self.trainer.best_score:.4f}")

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise

def main():
    """Main function to run the ML pipeline"""
    print("="*70)
    print("STUDENT PERFORMANCE CLASSIFICATION - ML MODEL COMPARISON")
    print("="*70)

    # Initialize pipeline
    pipeline = MLPipeline()

    # Run complete pipeline
    pipeline.run_complete_pipeline(
        include_tuning=True,  # Set to False to skip hyperparameter tuning
        tuning_method='random'  # Use 'grid' for GridSearchCV
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Check the outputs/ directory for:")
    print("- training_results.csv: Cross-validation scores")
    print("- evaluation_results.csv: Test set performance")
    print("- tuning_results.csv: Hyperparameter tuning results") 
    print("- best_model.pkl: Best performing model")
    print("- best_tuned_model.pkl: Best hyperparameter-tuned model")
    print("- Various plots and visualizations")

if __name__ == "__main__":
    main()
