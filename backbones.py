import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, mobilenet_v2
import timm  # For EfficientNet and other models
from timm import create_model  # For ViT and other models from the "timm" library


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained ResNet-50
        resnet = resnet50(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = nn.Sequential(
            *(list(resnet.children())[:-2])  # Removes FC and avg pooling
        )
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (512 for ResNet-50)
        self.num_features = resnet.fc.in_features

        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through ResNet backbone
        x = self.feature_extractor(x)
        
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained MobileNetV2
        mobilenet = mobilenet_v2(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = mobilenet.features
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (1280 for MobileNetV2)
        self.num_features = mobilenet.classifier[1].in_features

        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through MobileNetV2 backbone
        x = self.feature_extractor(x)
        
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]

class EfficientNetB0(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Use the timm library to load EfficientNetB0
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = self.model.classifier.in_features
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_optimizer_list(self):
        return [{'params': self.model.parameters(), 'lr': 1e-4}]

class ViT(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained Vision Transformer
        vit = create_model('vit_base_patch16_224', pretrained=pretrained)
        # Remove the fully connected layer and retain only the transformer backbone
        self.feature_extractor = vit.forward_features
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (768 for ViT base model)
        self.num_features = vit.head.in_features

        # Store the device and move the model to the correct device
        self.device = device
        self.to(device)  # Make sure the model is on the correct device

    def forward(self, x):
        # Ensure the input is on the correct device (same as the model)
        x = x.to(self.device)  # Ensure the input is on the same device as the model

        # Pass through ViT backbone
        x = self.feature_extractor(x)
        
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]


# Function to initialize the backbone
def get_backbone(name, pretrained=True, device="cuda"):
    if name == "resnet50":
        return ResNet50(pretrained, device)
    elif name == "mobilenetv2":
        return MobileNetV2(pretrained, device)
    elif name == "efficientnetb0":
        return EfficientNetB0(pretrained, device)
    elif name == "vit":
        return ViT(pretrained, device)
    else:
        raise ValueError(f"Backbone {name} is not supported.")






if __name__ == "__main__":
    # Set device (change to "cuda" if you want to use GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a dummy input tensor (e.g., 8 images of size 224x224)
    dummy_input = torch.randn(8, 3, 224, 224).to(device)  # Ensure the input is on the correct device


    # Initialize the backbone (e.g., ResNet-50)
    backbone = get_backbone("resnet50", pretrained=True, device=device)
    print(f"Using backbone: {backbone.__class__.__name__}")

    # Perform a forward pass with dummy input
    output = backbone(dummy_input)
    print("Output shape:", output.shape)

    # Check with other backbones as well
    # For example, using MobileNetV2:
    backbone = get_backbone("mobilenetv2", pretrained=True, device=device)
    output = backbone(dummy_input)
    print(f"Output shape with {backbone.__class__.__name__}:", output.shape)
    
    # You can similarly test with EfficientNetB0 and ViT
    backbone = get_backbone("efficientnetb0", pretrained=True, device=device)
    output = backbone(dummy_input)
    print(f"Output shape with {backbone.__class__.__name__}:", output.shape)

    #VIT NO ME ESTÁ FUNCIONANDO POR UN ERROR EN DE CUDA.TENSOR VS TORCH.TENSOR
    #backbone = get_backbone("vit", pretrained=True, device=device)
    #output = backbone(dummy_input)
    #print(f"Output shape with {backbone.__class__.__name__}:", output.shape)