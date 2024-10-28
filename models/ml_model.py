from abc import ABC, abstractmethod
import pandas as pd

class ml_model(ABC):

    @abstractmethod
    def classify_datapoint(self, datapoint : pd.Series) -> str :
        pass
    
    @abstractmethod
    def classify_dataset(self, df : pd.core.frame.DataFrame) -> pd.core.frame.DataFrame :
        pass