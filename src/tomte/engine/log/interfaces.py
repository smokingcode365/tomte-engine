from abc import ABC, abstractmethod

class LogProvider(ABC):
    @abstractmethod
    def info(self, message: str): pass

    @abstractmethod
    def warn(self, message: str): pass

    @abstractmethod
    def error(self, message: str, exc: Exception = None): pass

    @abstractmethod
    def debug(self, message: str): pass