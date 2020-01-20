from .dataset import CellImageDataset, CellImageDatasetwithTargets

dataset_dict = {'default': CellImageDataset, 'labeled': CellImageDatasetwithTargets}
