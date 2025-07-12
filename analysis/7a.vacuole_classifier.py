import pandas as pd
import numpy as np
import dill
from abc import ABC, abstractmethod

class CellClassifier(ABC):
    """Base class for cell classifiers."""
    @abstractmethod
    def classify_cells(self, metadata, features):
        """Classify cells based on feature data."""
        print("No classification method defined! Returning original cell data...")
    
    def save(self, filename):
        """Save the classifier to a file."""
        with open(filename, "wb") as f:
            dill.dump(self, f)
    
    @staticmethod
    def load(filename):
        """Load a classifier from a file."""
        with open(filename, "rb") as f:
            return dill.load(f)

class infectedClassifier(CellClassifier):
    """Simple classifier for infected presence using existing has_vacuole annotation."""
    
    def __init__(self):
        # Set the classes that will be available after classification
        self.classes = ["infected", "naive"]
    
    def classify_cells(self, metadata_df, features_df):
        """
        Classify cells based on has_vacuole column in metadata
        """
        # Check if has_vacuole column exists
        if 'has_vacuole' not in metadata_df.columns:
            raise ValueError("'has_vacuole' column not found in metadata_df")
        
        # Create a copy to avoid modifying the original
        classified_metadata = metadata_df.copy()
        classified_features = features_df.copy()
        
        # Simple binary classification based on has_vacuole
        classified_metadata["class"] = metadata_df["has_vacuole"].map({
            True: "infected",
            False: "naive"
        })
        
        # Set confidence to 1.0 since this is direct annotation
        classified_metadata["confidence"] = 1.0
        
        # Handle any NaN values in has_vacuole (if they exist)
        nan_mask = metadata_df["has_vacuole"].isna()
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} cells with NaN has_vacuole values, classifying as 'naive'")
            classified_metadata.loc[nan_mask, "class"] = "naive"
            classified_metadata.loc[nan_mask, "confidence"] = 0.5
        
        return classified_metadata, classified_features

if __name__ == "__main__":
    # Create the classifier
    classifier = infectedClassifier()
    
    # Save it to a dill file
    output_path = "config/vacuole_classifier.dill"
    classifier.save(output_path)
    
    print(f"infectedClassifier saved to {output_path}")
    print(f"Available classes: {classifier.classes}")
    
    # Test loading it back
    loaded_classifier = infectedClassifier.load(output_path)
    print(f"Successfully loaded classifier with classes: {loaded_classifier.classes}")