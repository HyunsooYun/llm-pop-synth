import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Union

class TextEncoder:
    """Transforms tabular data into textual format."""
    
    def __init__(self, feature_order=None):
        """
        TextEncoder initialization
        
        Parameters:
            feature_order: List of feature order or a function that returns feature order
        """
        self.feature_order = feature_order
        
        # Forward mappings (Enum -> Text)
        self.gender_map = {1: 'Male', 2: 'Female'}
        self.binary_map = {1: 'Yes', 2: 'No'}
        self.homeincome_map = {
            1: '< 1M KRW',
            2: '1M-3M KRW',
            3: '3M-5M KRW',
            4: '5M-10M KRW',
            5: '> 10M KRW'
        }
        self.hometype_map = {
            1: 'Apartment',
            2: 'Villa',
            3: 'Multi-family',
            4: 'Single-family',
            5: 'Studio-type residence',
            6: 'Other'
        }
        self.workdays_map = {
            1: '5 days',
            2: '6 days',
            3: '1~4 days',
            4: 'Inoccupation/non-regular'
        }
        self.worktype_map = {
            1: 'Student',
            2: 'Inoccupation/Housewife',
            3: 'Experts',
            4: 'Service',
            5: 'Sales',
            6: 'Manager/Office',
            7: 'Agriculture/fisher',
            8: 'Simple labor',
            9: 'Other'
        }
        self.student_map = {
            1: 'Elementary/Middle/High School',
            2: 'Preschool',
            3: 'University',
            4: 'Not student'
        }

        self.features = [
            'Age', 'Gender', 'Homeincome', 'Hometype', 
            'CarOwn', 'Driver', 'Workdays', 'Worktype',
            'Student', 'NumHH', 'KidinHH', 'ComMode', 'ComTime'
        ]
        
        if self.feature_order is None:
            self.feature_order = self.features
    
    def encode_feature(self, feature: str, value) -> str:
        """Convert each feature to text"""
        if feature == 'Age':
            return f"Age group is {value}"
        elif feature == 'Gender':
            return f"Gender is {self.gender_map[value]}"
        elif feature == 'Homeincome':
            return f"Household monthly income level is {self.homeincome_map[value]}"
        elif feature == 'Hometype':
            return f"Home type is {self.hometype_map[value]}"
        elif feature == 'CarOwn':
            return f"Car ownership of household is {self.binary_map[value]}"
        elif feature == 'Driver':
            return f"Driver license is {self.binary_map[value]}"
        elif feature == 'Workdays':
            return f"Work days is {self.workdays_map[value]}"
        elif feature == 'Worktype':
            return f"Work type is {self.worktype_map[value]}"
        elif feature == 'Student':
            return f"Education status is {self.student_map[value]}"
        elif feature == 'NumHH':
            return f"Number of household members is {value}"
        elif feature == 'KidinHH':
            return f"Kid in household is {self.binary_map[value]}"
        elif feature == 'ComMode':
            return f"Major travel mode is {value}"
        elif feature == 'ComTime':
            return f"Major departure time is {value}"
    
    def encode_row(self, row: pd.Series, seed: int = None) -> str:
        """
        Converts a row into a text sentence.
        If self.feature_order is callable, it calls it each time to use a random linear extension.
        """
        if callable(self.feature_order):
            order = self.feature_order()
        else:
            order = self.feature_order
                
        text_parts = [self.encode_feature(feat, row[feat]) for feat in order]
        return ", ".join(text_parts) + "."
        
    def encode_dataset(self, df: pd.DataFrame, random_seed: int = None) -> List[str]:
        """Converts the entire dataset into a list of text sentences."""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        encoded_texts = []
        for idx, row in df.iterrows():
            encoded_texts.append(self.encode_row(row))
            
        return encoded_texts