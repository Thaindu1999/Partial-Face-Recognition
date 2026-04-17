import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== INVERTED RESIDUAL BLOCK ====================
class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand=2):
        super(InvertedResidual, self).__init__()

        hidden = in_c * expand
        self.use_res = (stride == 1 and in_c == out_c)

        self.block = nn.Sequential(
            # Expansion
            nn.Conv2d(in_c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.PReLU(hidden),

            # Depthwise
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.PReLU(hidden),

            # Projection
            nn.Conv2d(hidden, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        if self.use_res:
            return x + self.block(x)
        else:
            return self.block(x)


# ==================== MOBILEFACENET ====================
class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512, dropout=0.4):
        super(MobileFaceNet, self).__init__()

        # ------------------ Feature Extractor ------------------
        self.features = nn.Sequential(
            # Initial Conv
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),

            # Depthwise Conv
            nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),

            # Bottleneck Blocks
            InvertedResidual(64, 64, stride=2, expand=2),
            InvertedResidual(64, 64, stride=1, expand=2),
            InvertedResidual(64, 64, stride=1, expand=2),
            InvertedResidual(64, 64, stride=1, expand=2),

            InvertedResidual(64, 128, stride=2, expand=2),
            InvertedResidual(128, 128, stride=1, expand=2),
            InvertedResidual(128, 128, stride=1, expand=2),
            InvertedResidual(128, 128, stride=1, expand=2),

            InvertedResidual(128, 128, stride=2, expand=2),
            InvertedResidual(128, 128, stride=1, expand=2),
        )

        # ------------------ Final Convolution ------------------
        self.conv_sep = nn.Sequential(
            nn.Conv2d(128, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512),

            nn.Conv2d(512, 512, 7, groups=512, bias=False),
            nn.BatchNorm2d(512),
        )

        # ------------------ Embedding Head ------------------
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

        self._init_weights()

    # ------------------ Weight Initialization ------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ------------------ Forward Pass ------------------
    def forward(self, x):
        x = self.features(x)
        x = self.conv_sep(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        # Normalize embedding (VERY IMPORTANT for cosine similarity)
        x = F.normalize(x, p=2, dim=1)

        return x