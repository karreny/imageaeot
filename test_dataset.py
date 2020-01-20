from dataset import dataset_dict

def test():
    dataset = dataset_dict['labeled'](datadir='data', metafile='splits/train_total_labeled.csv', mode='train')
    print(len(dataset))
    sample = dataset[0]
    for key in sample:
        print(sample[key])
    print(sample['image'].shape)

    dataset = dataset_dict['labeled'](datadir='data', metafile='splits/val_total_labeled.csv', mode='val')
    print(len(dataset))
    sample = dataset[0]
    for key in sample:
        print(sample[key])
    print(sample['image'].shape)

if __name__ == "__main__":
    test()
