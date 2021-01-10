import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
#from dataset import Dataset

class DataGenerator(Sequence):
    def __init__(self, dataset, subset, batch_size):
        self.dataset = dataset
        self.subset =subset
        self.datasize = len(self.dataset.labels_dict[subset])

        self.batch_size = batch_size
        self.batches_per_epoch = int(np.ceil(self.datasize/ float(batch_size)))

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        
        end = min(start + self.batch_size, self.datasize)
        images, labels = self.dataset.load_batch(self.subset, start, end)
        norm_images = (images - self.dataset.mean) / self.dataset.stddev

        return norm_images, to_categorical(labels, 4)
