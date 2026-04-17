import torch
from torchvision import transforms
from model.mobilefacenet import MobileFaceNet


def load_model(model_path="model/best_mobilefacenet.pth"):
    device = torch.device("cpu")

    model = MobileFaceNet(embedding_size=512)

    checkpoint = torch.load(model_path, map_location=device)

    # FIX: Handle both raw state_dict and full checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])