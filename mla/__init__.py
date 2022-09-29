from mla.dataset import Dataset
from mla.model import Model_RF

class Dataset:
    def __init__(self, train_dir='Data/train.csv', test_dir='Data/test.csv'):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)


       


