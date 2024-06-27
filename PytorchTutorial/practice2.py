import os
import pandas as pd 
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transfor = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx): #idx에 해당하는 샘플을 데이터셋에서 불러오고 반환
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) #인덱스를 기반으로 디스크에서 이미지의 위치 식별
        image = read_image(img_path)    #이미지를 텐서로 변환 
        label = self.img_labels.ilo[idx, 1] #csv 데이터로부터 해당하는 정답(label)을 가져옴
        if self.transform:
            image = self.tansform(image)
        if self.target_transform:
            label = self.target_transform(lable)
        sample = {"image": image, "label": label}   #텐서 이미지와 라벨을 python dict 형으로 반환
        return sample
