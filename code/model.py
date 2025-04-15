from torchvision.models import convnext_tiny, densenet121, mobilenet_v2, resnet50
import timm
import torch.nn as nn

class ConvNeXtTinyModel(nn.Module):
    def __init__(self, num_classes=23, pretrained=True, freeze_backbone=False):
        super(ConvNeXtTinyModel, self).__init__()
        
        # Load pre-trained ConvNeXt Tiny model
        self.model = convnext_tiny(pretrained=pretrained)
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
        
        # Replace the classifier layer (fully connected layer) to match the number of classes
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, frozen=False):
        super(DenseNet121, self).__init__()
        self.model = densenet121(pretrained=pretrained)
        
        if frozen:
            for param in self.model.features.parameters():  # Freeze chỉ các lớp feature extractor
                param.requires_grad = False
        
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        # Truyền input qua mô hình DenseNet121
        return self.model(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, frozen=False):
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=True)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=10, frozen=False):
        super(EfficientNetV2, self).__init__()
        self.model = timm.create_model('efficientnetv2_rw_t', pretrained=True)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, frozen=False):
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=True)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class ViTClassifier128(nn.Module):
    def __init__(self, num_classes=23, pretrained=True, img_size=128, freeze_backbone=False):
        super(ViTClassifier128, self).__init__()
        
        # Tải mô hình ViT với kích thước ảnh 128x128
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            img_size=img_size
        )
        
        # Freeze backbone nếu cần
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Lấy số lượng đầu ra từ lớp head
        in_features = self.vit.head.in_features
        
        # Thay thế lớp head bằng lớp phân loại tùy chỉnh
        self.vit.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)