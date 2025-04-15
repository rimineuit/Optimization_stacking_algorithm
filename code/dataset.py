import torch 
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, Subset
class DERMNET_DATASET(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        self.labels_dict = {
            'Light Diseases and Disorders of Pigmentation': 0, 
            'Lupus and other Connective Tissue diseases': 1,
            'Acne and Rosacea Photos': 2,
            'Systemic Disease': 3,
            'Poison Ivy Photos and other Contact Dermatitis': 4,
            'Vascular Tumors': 5,
            'Urticaria Hives': 6,
            'Atopic Dermatitis Photos': 7,
            'Bullous Disease Photos': 8,
            'Hair Loss Photos Alopecia and other Hair Diseases': 9,
            'Tinea Ringworm Candidiasis and other Fungal Infections': 10,
            'Psoriasis pictures Lichen Planus and related diseases': 11,
            'Melanoma Skin Cancer Nevi and Moles': 12,
            'Nail Fungus and other Nail Disease': 13,
            'Scabies Lyme Disease and other Infestations and Bites': 14,
            'Eczema Photos': 15,
            'Exanthems and Drug Eruptions': 16,
            'Herpes HPV and other STDs Photos': 17,
            'Seborrheic Keratoses and other Benign Tumors': 18,
            'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 19,
            'Vasculitis Photos': 20,
            'Cellulitis Impetigo and other Bacterial Infections': 21,
            'Warts Molluscum and other Viral Infections': 22
        }

        dataset_type = 'train' if self.train else 'test'
        dataset_dir = os.path.join(root_dir, dataset_type)

        # Tạo một dictionary để lưu danh sách ảnh cho từng nhãn
        label_to_images = {label: [] for label in range(len(self.labels_dict))}

        # Duyệt qua các thư mục con và lưu đường dẫn ảnh theo nhãn tương ứng
        for class_name, class_idx in self.labels_dict.items():
            class_folder = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_folder):
                image_paths = sorted([os.path.join(class_folder, img) 
                                      for img in os.listdir(class_folder) 
                                      if img.endswith(('.png', '.jpg', '.jpeg'))])
                label_to_images[class_idx].extend(image_paths)

        # Tạo danh sách ảnh và nhãn xen kẽ
        self.image_paths = []
        self.labels = []
        for label, images in label_to_images.items():
            for img in images:
                self.image_paths.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)  # Tổng số lượng ảnh

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Mở ảnh và áp dụng transform nếu có
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)

        return image, label

    def show_image(self, idx):
        image, label = self.__getitem__(idx)
        keys = [k for k, v in self.labels_dict.items() if v == label]
        
        # Hiển thị ảnh
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Label: {keys[0]}")
        plt.show()




class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = Subset(dataset, indices)
        self.indices = indices
        self.labels_dict = dataset.labels_dict  # Giữ nguyên labels_dict từ dataset gốc
        self.dataset_cls = dataset  # Duy trì tham chiếu tới dataset gốc

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_class_counts(self):
        """Tính toán số lượng ảnh trong mỗi lớp."""
        class_counts = {class_name: 0 for class_name in self.labels_dict.keys()}
        for idx in self.indices:
            label = self.dataset_cls.labels[idx]
            class_name = [k for k, v in self.labels_dict.items() if v == label][0]
            class_counts[class_name] += 1
        return class_counts

    def show_image(self, idx):
        """Hiển thị ảnh."""
        actual_idx = self.indices[idx]
        self.dataset_cls.show_image(actual_idx)