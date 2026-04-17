import torch
import random
import torchvision.transforms as transforms

# ==================== CUSTOM OCCLUSION ====================

class RandomFaceOcclusion:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        _, h, w = img.shape

        choice = random.choice(['top', 'middle', 'bottom', 'left', 'right'])

        if choice == 'top':
            region = img[:, :h//3, :]
        elif choice == 'middle':
            region = img[:, h//3:2*h//3, :]
        elif choice == 'bottom':
            region = img[:, 2*h//3:, :]
        elif choice == 'left':
            region = img[:, :, :w//3]
        else:  # right
            region = img[:, :, 2*w//3:]

        # 🔥 More realistic occlusion (not pure black)
        region[:] = torch.randn_like(region) * 0.1

        return img