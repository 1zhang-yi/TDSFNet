import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import torch.utils.data as data_utils

tags = pd.DataFrame([
        {'names': 'Diagnosis', 'abbrevs': 'DIAG', 'colnames': 'diagnosis', 'seven_pt': 0},
        {'names': 'Pigment Network', 'abbrevs': 'PN', 'colnames': 'pigment_network', 'seven_pt': 1},
        {'names': 'Blue Whitish Veil', 'abbrevs': 'BWV', 'colnames': 'blue_whitish_veil', 'seven_pt': 1},
        {'names': 'Vascular Structures', 'abbrevs': 'VS', 'colnames': 'vascular_structures', 'seven_pt': 1},
        {'names': 'Pigmentation', 'abbrevs': 'PIG', 'colnames': 'pigmentation', 'seven_pt': 1},
        {'names': 'Streaks', 'abbrevs': 'STR', 'colnames': 'streaks', 'seven_pt': 1},
        {'names': 'Dots and Globules', 'abbrevs': 'DaG', 'colnames': 'dots_and_globules', 'seven_pt': 1},
        {'names': 'Regression Structures', 'abbrevs': 'RS', 'colnames': 'regression_structures', 'seven_pt': 1},
    ])

diagnosis = pd.DataFrame([
        {'nums': 0, 'names': 'basal cell carcinoma', 'abbrevs': 'BCC', 'info': 'Common non-melanoma cancer'},
        {'nums': 1,
         'names': ['nevus', 'blue nevus', 'clark nevus', 'combined nevus', 'congenital nevus', 'dermal nevus',
                   'recurrent nevus', 'reed or spitz nevus'], 'abbrevs': 'NEV'},
        {'nums': 2,
         'names': ['melanoma', 'melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                   'melanoma (0.76 to 1.5 mm)',
                   'melanoma (more than 1.5 mm)', 'melanoma metastasis'], 'abbrevs': 'MEL'},
        {'nums': 3, 'names': ['DF/LT/MLS/MISC', 'dermatofibroma', 'lentigo', 'melanosis',
                              'miscellaneous', 'vascular lesion'], 'abbrevs': 'MISC'},
        {'nums': 4, 'names': 'seborrheic keratosis', 'abbrevs': 'SK'},
    ])

pigment_network = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'typical', 'abbrevs': 'TYP', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'atypical', 'abbrevs': 'ATP', 'scores': 2, 'info': ''},
    ])

blue_whitish_veil = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'present', 'abbrevs': 'PRS', 'scores': 2, 'info': ''},
])

streaks = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'regular', 'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'irregular', 'abbrevs': 'IR', 'scores': 1, 'info': ''},
    ])

dots_and_globules = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': 'regular', 'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': 'irregular', 'abbrevs': 'IR', 'scores': 1, 'info': ''},
])

vascular_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['regular', 'arborizing', 'comma', 'hairpin', 'within regression', 'wreath'],
         'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': ['dotted/irregular', 'dotted', 'linear irregular'], 'abbrevs': 'IR', 'scores': 2,
         'info': ''},
])

pigmentation = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['regular', 'diffuse regular', 'localized regular'], 'abbrevs': 'REG', 'scores': 0,
         'info': ''},
        {'nums': 2, 'names': ['irregular', 'diffuse irregular', 'localized irregular'], 'abbrevs': 'IR', 'scores': 1,
         'info': ''},
])

regression_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['present', 'blue areas', 'white areas', 'combinations'], 'abbrevs': 'PRS', 'scores': 1,
         'info': ''},
])

def get_label(label_dict, data):
    for i, name in enumerate(label_dict.names.values):
        if type(name) is str:
            if data == name:
                DIAG_label = i
        else:
            if data in name:
                DIAG_label = i

    return DIAG_label

train_tf = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=40),
    transforms.ToTensor()
])

test_tf = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

class Dataset(data_utils.Dataset):
    def __init__(self, meta_df, index, img_path, type='train', train_transform=train_tf, test_transform=test_tf):
        self.meta_df = meta_df
        self.index = index
        if type == 'train':
            self.transform = train_transform
        else:
            self.transform = test_transform
        self.img_path = img_path
        self.data = self.get_data()

    def get_data(self):
        data_list = []
        df = self.meta_df.iloc[self.index]
        for id in self.index:
            row = df[df.case_num == id + 1]

            DIAG = row['diagnosis'].values[0]

            PN = row['pigment_network'].values[0]
            STR = row['streaks'].values[0]
            PIG = row['pigmentation'].values[0]
            RS = row['regression_structures'].values[0]
            DaG = row['dots_and_globules'].values[0]
            BWV = row['blue_whitish_veil'].values[0]
            VS = row['vascular_structures'].values[0]

            DIAG_label = get_label(diagnosis, DIAG)
            PN_label = get_label(pigment_network, PN)
            BWV_label = get_label(blue_whitish_veil, BWV)
            VS_label = get_label(vascular_structures, VS)
            PIG_label = get_label(pigmentation, PIG)
            STR_label = get_label(streaks, STR)
            DaG_label = get_label(dots_and_globules, DaG)
            RS_label = get_label(regression_structures, RS)

            clinic_path = self.img_path + '/' + row['clinic'].values[0]
            derm_path = self.img_path + '/' + row['derm'].values[0]


            data_list.append((id, DIAG_label, PN_label, BWV_label, VS_label,
                               PIG_label, STR_label, DaG_label, RS_label,
                               clinic_path, derm_path
                               ))

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'case id': self.data[idx][0],

                'DIAG_label': self.data[idx][1],

                'PN_label': self.data[idx][2],
                'BWV_label': self.data[idx][3],
                'VS_label': self.data[idx][4],
                'PIG_label': self.data[idx][5],
                'STR_label': self.data[idx][6],
                'DaG_label': self.data[idx][7],
                'RS_label': self.data[idx][8],

                'clinic': self.transform(Image.open(self.data[idx][9]).convert('RGB')),
                'derm': self.transform(Image.open(self.data[idx][10]).convert('RGB'))

                # 'clinic': self.data[idx][9],
                # 'derm': self.data[idx][10]
                }












