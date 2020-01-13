from dataset.dataset import CellImageDataset

def test():
    dataset = CellImageDataset(datadir='data', metafile='splits/train_NIH3T3.csv', mode='train')
    print(len(dataset))
    sample = dataset[0]
    for key in sample:
        print(sample[key])
    print(sample['image'].shape)

    dataset = CellImageDataset(datadir='data', metafile='splits/val_NIH3T3.csv', mode='val')
    print(len(dataset))
    sample = dataset[0]
    for key in sample:
        print(sample[key])
    print(sample['image'].shape)

if __name__ == "__main__":
    test()
