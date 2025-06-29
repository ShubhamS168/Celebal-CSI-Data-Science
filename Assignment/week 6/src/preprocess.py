"""
Data Preprocessing Module for Student Performance Classification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self, filepath):
        """Load the dataset from CSV file"""
        try:
            df = pd.read_csv(filepath, sep=';')
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_target_variable(self, df, target_col='G3'):
        """Convert numerical grades to categorical performance levels"""
        # Create grade categories based on G3 scores
        # 0-9: Poor, 10-13: Average, 14-16: Good, 17-20: Excellent
        def categorize_grade(grade):
            if grade <= 9:
                return 'Poor'
            elif grade <= 13:
                return 'Average'
            elif grade <= 16:
                return 'Good'
            else:
                return 'Excellent'

        df['Performance'] = df[target_col].apply(categorize_grade)
        logger.info(f"Target variable created. Distribution:")
        logger.info(f"{df['Performance'].value_counts()}")
        return df

    def encode_categorical_features(self, df):
        """Encode categorical features using LabelEncoder"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Performance']

        df_encoded = df.copy()

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_encoded[col] = self.label_encoders[col].fit_transform(df[col])

        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df_encoded

    def prepare_features(self, df):
        """Prepare feature matrix and target vector"""
        # Remove correlated grades (G1, G2) and target from features
        feature_cols = [col for col in df.columns 
                       if col not in ['G1', 'G2', 'G3', 'Performance']]

        X = df[feature_cols].copy()
        y = df['Performance'].copy()

        # Encode target variable
        if 'Performance' not in self.label_encoders:
            self.label_encoders['Performance'] = LabelEncoder()
        y_encoded = self.label_encoders['Performance'].fit_transform(y)

        self.feature_columns = feature_cols
        logger.info(f"Features prepared. Shape: {X.shape}")
        logger.info(f"Feature columns: {feature_cols}")

        return X, y_encoded, y

    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Features scaled successfully")
        return X_train_scaled, X_test_scaled

    def preprocess_data(self, filepath, test_size=0.2, random_state=42):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(filepath)

        # Create target variable
        df = self.create_target_variable(df)

        # Encode categorical features
        df_encoded = self.encode_categorical_features(df)

        # Prepare features and target
        X, y_encoded, y_original = self.prepare_features(df_encoded)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        logger.info(f"Data preprocessing completed successfully")
        logger.info(f"Train set shape: {X_train_scaled.shape}")
        logger.info(f"Test set shape: {X_test_scaled.shape}")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_columns,
            'target_encoder': self.label_encoders['Performance']
        }

def main():
    """Test the preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_data('../data/student-por.csv')

    print("Preprocessing completed successfully!")
    print(f"Training set shape: {data['X_train'].shape}")
    print(f"Test set shape: {data['X_test'].shape}")
    print(f"Number of features: {len(data['feature_names'])}")

if __name__ == "__main__":
    main()
