from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, RandomResizedCrop, Normalize, ToTensor
import pandas as pd
import torch


def label_indexer_fine(labels_dataframe):
    split_labels_dict = dict(attribute_id=[], attr_tier1=[], attr_tier2=[], attr_tier3=[])
    for i in range(labels_dataframe.shape[0]):
        tem = [item.strip() for split_list in [item_.split(';')
                                               for item_ in labels_dataframe['attribute_name'][i].split('::')]
               for item in split_list]
        split_labels_dict['attribute_id'].append(labels_dataframe['attribute_id'][i])
        split_labels_dict['attr_tier1'].append(tem[0])
        split_labels_dict['attr_tier2'].append(tem[1])
        try:
            split_labels_dict['attr_tier3'].append(tem[2])
        except:
            split_labels_dict['attr_tier3'].append('None')
    split_labels = pd.DataFrame(split_labels_dict,
                                columns=['attribute_id', 'attr_tier1', 'attr_tier2', 'attr_tier3'])

    tier1 = dict()
    tier2 = dict()
    counting_dict = dict()
    attr2indexing = dict()
    indexing2attr = dict()
    label_indexing_list = []
    for i_1, item1 in enumerate(sorted(list(set(list(split_labels['attr_tier1']))))):
        tier1[item1] = i_1
        tier2[item1] = dict()
        list_tem = sorted(list(set(list(split_labels['attr_tier2'][split_labels['attr_tier1'] == item1]))))
        for i_2, item2 in enumerate(list_tem):
            tier2[item1][item2] = i_2 + 1
        counting_dict[item1] = np.ones(len(list_tem), dtype='int')
    for idx in range(split_labels.shape[0]):
        tier1_idx = tier1[split_labels['attr_tier1'][idx]]
        tier2_idx = tier2[split_labels['attr_tier1'][idx]][split_labels['attr_tier2'][idx]]
        tier3_idx = counting_dict[split_labels['attr_tier1'][idx]][tier2_idx - 1]
        counting_dict[split_labels['attr_tier1'][idx]][tier2_idx - 1] += 1
        label_indexing_list.append([tier1_idx, tier2_idx, tier3_idx])
        attr2indexing[split_labels['attribute_id'][idx]] = [tier1_idx, tier2_idx, tier3_idx]
        indexing2attr[str([tier1_idx, tier2_idx, tier3_idx])] = split_labels['attribute_id'][idx]
    labels_indexing_df = split_labels.copy()
    labels_indexing_df['indexing'] = label_indexing_list
    return labels_indexing_df, attr2indexing, indexing2attr


def label_indexer_coarse(labels_dataframe):
    split_labels_dict = dict(attribute_id=[], attr_tier1=[], attr_tier2=[])
    for i in range(labels_dataframe.shape[0]):
        tem = [item.strip() for item in labels_dataframe['attribute_name'][i].split('::')]
        split_labels_dict['attribute_id'].append(labels_dataframe['attribute_id'][i])
        split_labels_dict['attr_tier1'].append(tem[0])
        split_labels_dict['attr_tier2'].append(tem[1])

    split_labels = pd.DataFrame(split_labels_dict,
                                columns=['attribute_id', 'attr_tier1', 'attr_tier2'])

    tier1 = dict()
    tier2 = dict()
    attr2indexing = dict()
    indexing2attr = dict()
    label_indexing_list = []
    for i_1, item1 in enumerate(sorted(list(set(list(split_labels['attr_tier1']))))):
        assert len(list(set(list(split_labels['attr_tier2'][split_labels['attr_tier1'] == item1])))) \
               == len(list(split_labels['attr_tier2'][split_labels['attr_tier1'] == item1]))
        tier1[item1] = i_1
        tier2[item1] = dict()
        list_tem = list(split_labels['attr_tier2'][split_labels['attr_tier1'] == item1])
        for i_2, item2 in enumerate(list_tem):
            tier2[item1][item2] = i_2
    for idx in range(split_labels.shape[0]):
        tier1_idx = tier1[split_labels['attr_tier1'][idx]]
        tier2_idx = tier2[split_labels['attr_tier1'][idx]][split_labels['attr_tier2'][idx]]
        label_indexing_list.append([tier1_idx, tier2_idx])
        attr2indexing[split_labels['attribute_id'][idx]] = [tier1_idx, tier2_idx]
        indexing2attr[str([tier1_idx, tier2_idx])] = split_labels['attribute_id'][idx]
    labels_indexing_df = split_labels.copy()
    labels_indexing_df['indexing'] = label_indexing_list
    return labels_indexing_df, attr2indexing, indexing2attr


def imgreader(img_id, ext, path, attr_ids, attr2indexing, length_list, dimension=256,
              task=('ml', 'ml', 'mc', 'ml', 'ml'), transform='val', grey_scale=False):
    file_path = f'{path}/{img_id}.{ext}'
    with open(file_path, 'rb') as f:
        img_ = Image.open(f)
        if grey_scale:
            img = img_.convert('L')
        else:
            img = img_.convert('RGB')

    transformer = {
        'train': Compose([RandomResizedCrop(size=(dimension, dimension)),
                          ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                         ),
        'val': Compose([Resize(size=(dimension, dimension)),
                        ToTensor(),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    }
    x = transformer[transform](img)
    y_list = [attr2indexing[int(attr_id)] for attr_id in attr_ids.split()]
    y_dict = labels_list2array(y_list, length_list, task)
    return x, tuple(y_dict.values())


def counting_elements(labels_indexing_df):
    return [len(labels_indexing_df['indexing'][labels_indexing_df['attr_tier1']==catagory])
            for catagory in sorted(list(set(list(labels_indexing_df['attr_tier1']))))]


def labels_list2array(y_list, length_list, task):
    y_dict = dict()
    for i in range(len(task)):
        if task[i] != 'mc':
            y_dict[i] = torch.FloatTensor(np.zeros(length_list[i]))
        else:
            y_dict[i] = torch.LongTensor([0])

    for idx_list in y_list:
        if task[idx_list[0]] != 'mc':
            y_dict[idx_list[0]][idx_list[1]] = 1
        else:
            y_dict[idx_list[0]][0] = idx_list[1]

    return y_dict


def image_list_scan(data_info, indices):
    if indices is None:
        return list(data_info['id']), list(data_info['attribute_ids'])
    else:
        return list(data_info['id'][indices]), list(data_info['attribute_ids'][indices])



