from torch.utils.data import Dataset, DataLoader
import os 


class BrainTumourDataset(Dataset): 

    def __init__(self, data_dir: str, transform = None): 
        self.data_dict = self.transform_data_to_dict(data_dir)
        self.transform = transform 
        self.labels =  ['glioma', 'meningioma', 'notumor', 'pituitary']
        

    def __getitem__(self, idx):
        datapoint = self.data_dict[idx]
        path = datapoint['path']
        label = self.labels.index(datapoint['label']) 

        img = Image.open(path).convert('RGB')
        if self.transform : img = self.transform(img)
        return img, label
            
        
    def __len__(self): 
        return len(self.data_dict)
    

    def transform_data_to_dict(self, folder_path):
        image_dict = {} 
        index = 0

        for root, _, files in os.walk(folder_path):
            label = os.path.basename(root)
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, file)
                    image_dict[index] = {'path': file_path, 'label': label}
                    index += 1
        return image_dict