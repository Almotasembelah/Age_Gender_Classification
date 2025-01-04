from torch.utils.data import Dataset
from PIL import Image

class FairFaceDataset(Dataset):
    '''
    This class is used to create a dataset for the FairFace dataset. 
    It can be used to create a dataset for multiple tasks or a single task.
    '''
    def __init__(self, df, transform=None, root=root, gender=True, age=True, race=True):
        self.df = df
        self.transform = transform
        self.root = root
        
        self.gender = gender
        self.age = age
        self.race = race

        self.classes = []
        if self.gender:
            self.gender_labels = {class_name: torch.tensor(i)  for i, class_name in enumerate(df['gender'].value_counts().index.values)}
            self.gender_classes = list(self.gender_labels.keys())
            self.classes.append(self.gender_classes)
            
        if self.age:
            self.age_labels = {class_name: torch.tensor(i)  for i, class_name in enumerate(df['age'].value_counts().index.values)}
            self.age_classes = list(self.age_labels.keys())
            self.classes.append(self.age_classes)
            
        if self.race:
            self.race_labels = {class_name: torch.tensor(i)  for i, class_name in enumerate(df['race'].value_counts().index.values)}
            self.race_classes = list(self.race_labels.keys())
            self.classes.append(self.race_classes)

        if len(self.classes) == 1:
            self.classes = self.classes[0]
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx].squeeze()
        img = Image.open(self.root + item['file']) if root else Image.open(item['file'])/255.
        if self.transform:
            img = self.transform(img)

        labels = []
        if self.gender:
            gen_label = self.gender_labels[item['gender']]
            labels.append(gen_label)
        if self.age:
            age_label = self.age_labels[item['age']]
            labels.append(age_label)
        if self.race:
            race_label = self.race_labels[item['race']]
            labels.append(race_label)

        if len(labels) > 1:
            return img, labels
        elif len(labels) == 1:
            return img, *labels
        else:
            raise 


class FairFaceDatasetV2(Dataset):
    '''
    Age is not a classification task, but a regression task.
    '''
    def __init__(self, df, transform=None, root=root, gender=True, age=True, race=True):
        self.df = df
        self.transform = transform
        self.root = root
        
        # make the return values flexible to choose
        self.gender = gender
        self.age = age
        self.race = race

        self.classes = []
        if self.gender:
            self.gender_labels = {class_name: torch.tensor(i)  for i, class_name in enumerate(df['gender'].value_counts().index.values)}
            self.gender_classes = list(self.gender_labels.keys())
            self.classes.append(self.gender_classes)
            
        if self.age:
            self.age_labels = {class_name: class_name  for i, class_name in enumerate(df['age'].value_counts().index.values)}
            self.age_classes = list(self.age_labels.keys())
            self.classes.append(self.age_classes)
            
        if self.race:
            self.race_labels = {class_name: torch.tensor(i)  for i, class_name in enumerate(df['race'].value_counts().index.values)}
            self.race_classes = list(self.race_labels.keys())
            self.classes.append(self.race_classes)

        if len(self.classes) == 1:
            self.classes = self.classes[0]
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx].squeeze()
        img = Image.open(self.root + item['file']) if root else Image.open(item['file'])/255.
        if self.transform:
            img = self.transform(img)

        labels = []
        if self.gender:
            gen_label = self.gender_labels[item['gender']]
            labels.append(gen_label)
        if self.age:
            age_label = item['age']

            if age_label == 'more than 70':
                age_label = torch.tensor(75).float()
            else:
                ages = age_label.split('-')
                age_label = (int(ages[0])+int(ages[1]))//2
            labels.append(torch.tensor(age_label/80).float())
        if self.race:
            race_label = self.race_labels[item['race']]
            labels.append(race_label)

        if len(labels) > 1:
            return img, labels
        elif len(labels) == 1:
            return img, *labels
        else:
            raise 


