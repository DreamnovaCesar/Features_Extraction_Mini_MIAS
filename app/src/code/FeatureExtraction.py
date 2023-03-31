from abc import ABC
from abc import abstractmethod

class FeatureExtraction(ABC):

    # * Initializing (Constructor)
    def __init__(
        self, 
        label
    ) -> None:

        self._label = label;

    @abstractmethod
    def extractor(self):
        pass