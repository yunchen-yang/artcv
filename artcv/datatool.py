from torch.utils.data import Dataset
import sys
from artcv.utils import *
import math
from glob import glob


class ImgDataset(Dataset):
    def __init__(self, x, y, path, attr2indexing, length_list, task,
                 ext='png', dimension=256, transform='val', grey_scale=False):
        super().__init__()
        self.x = x
        self.y = y
        self.path = path
        self.attr2indexing =attr2indexing
        self.length_list = length_list
        self.task = task
        self.ext = ext
        self.dimension = dimension
        self.transform = transform
        self.grey_scale = grey_scale

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img, ys = imgreader(self.x[index], self.ext, self.path, self.y[index],
                            self.attr2indexing, self.length_list,
                            self.dimension, self.task, self.transform, self.grey_scale)
        y0, y1, y2, y3, y4 = ys

        return img, y0, y1, y2, y3, y4


class ImgTestset(Dataset):
    def __init__(self, x, dimension=256, transform='val', grey_scale=False):
        super().__init__()
        self.x = x

        self.dimension = dimension
        self.transform = transform
        self.grey_scale = grey_scale

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = imgreader_test(self.x[index], self.dimension, self.transform, self.grey_scale)

        return img


class TrainValSet:
    def __init__(self, ext='png', path=None, indices=None, dimension=256, data_info_path=None, labels_info_path=None,
                 task=('ml', 'ml', 'mc', 'ml', 'ml'), train_transform='train', train_val_split=0.7, seed=0,
                 test_path=None, test_csv_path=None):
        super().__init__()
        self.ext = ext
        self.dimension = dimension
        self.indices = indices
        self.seed = seed
        self.train_transform = train_transform

        if path is None:
            self.path = f'{sys.path[0]}/train'
        else:
            self.path = path

        self.test_path = test_path

        if data_info_path is None:
            self.data_info_path = f'{sys.path[0]}/train.csv'
        else:
            self.data_info_path = data_info_path

        if labels_info_path is None:
            self.labels_info_path = f'{sys.path[0]}/labels.csv'
        else:
            self.labels_info_path = labels_info_path

        self.labels_info = pd.read_csv(self.labels_info_path)
        self.data_info = pd.read_csv(self.data_info_path)

        self.labels_indexing_df, self.attr2indexing, self.indexing2attr = label_indexer_coarse(self.labels_info)
        self.length_list = counting_elements(self.labels_indexing_df)
        self.X_all, self.Y_all = image_list_scan(self.data_info, indices=self.indices)
        self.task = task

        if self.test_path is not None:
            self.test_csv_path = test_csv_path
            if self.test_csv_path is not None:
                self.test_csv = pd.read_csv(self.test_csv_path)
                self.X_test = [f'{self.test_path}/{_filename}.{self.ext}' for _filename in list(self.test_csv['id'])]
            else:
                self.X_test = glob(f'{self.test_path}/*.{self.ext}', recursive=True)
            self.test = ImgTestset(self.X_test, dimension=256, transform='val', grey_scale=False)

        self.train_val_split = train_val_split

        if bool(self.train_val_split):
            self.all = ImgDataset(self.X_all, self.Y_all, self.path, self.attr2indexing, self.length_list,
                                  task=self.task, ext=self.ext, dimension=256, transform='val', grey_scale=False)
            assert(0 < self.train_val_split < 1)
            num_train = math.ceil(len(self.X_all) * self.train_val_split)
            np.random.seed(seed=self.seed)
            indices_array = np.random.permutation(len(self.X_all))
            self.X_train = [self.X_all[i] for i in indices_array[:num_train]]
            self.Y_train = [self.Y_all[i] for i in indices_array[:num_train]]
            self.train = ImgDataset(self.X_train, self.Y_train, self.path, self.attr2indexing, self.length_list,
                                    task=self.task, ext=self.ext, dimension=256,
                                    transform=self.train_transform, grey_scale=False)
            self.X_val = [self.X_all[i] for i in indices_array[num_train:]]
            self.Y_val = [self.Y_all[i] for i in indices_array[num_train:]]
            self.val = ImgDataset(self.X_val, self.Y_val, self.path, self.attr2indexing, self.length_list,
                                  task=self.task, ext=self.ext, dimension=256, transform='val', grey_scale=False)
        else:
            if self.test_path is not None:
                self.all = ImgDataset(self.X_all, self.Y_all, self.path, self.attr2indexing, self.length_list,
                                      task=self.task, ext=self.ext, dimension=256,
                                      transform=self.train_transform, grey_scale=False)
            else:
                self.all = ImgDataset(self.X_all, self.Y_all, self.path, self.attr2indexing, self.length_list,
                                      task=self.task, ext=self.ext, dimension=256, transform='val', grey_scale=False)



