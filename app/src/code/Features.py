from abc import ABC
from abc import abstractmethod

class Features(ABC):

    @abstractmethod
    def calculate_features(self):
        pass