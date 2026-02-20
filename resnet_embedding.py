import img_split
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# works best with apple silicon gpu
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load pretrained ResNet50 + correct preprocess
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

model = models.resnet50(weights=weights)

# Remove final classifier so output is a 2048-d feature vector
model.fc = torch.nn.Identity()

model.to(device)
model.eval()

def build_xy(name_list):
    X_list, y_list = [], []

    for name in name_list:
        # REAL image
        img_real = Image.open(f"data/sample_real/{name}").convert("RGB")
        x_real = preprocess(img_real).unsqueeze(0).to(device)  # (1, 3, H, W)

        # AI image 
        img_ai = Image.open(f"data/sample_ai/{name}").convert("RGB")
        x_ai = preprocess(img_ai).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_real = model(x_real)  # (1, 2048)
            feat_ai = model(x_ai)      # (1, 2048)

            # optional normalization (often helpful)
            feat_real = feat_real / feat_real.norm(dim=-1, keepdim=True)
            feat_ai = feat_ai / feat_ai.norm(dim=-1, keepdim=True)

        X_list.append(feat_real.squeeze(0).cpu().numpy())
        y_list.append(0)

        X_list.append(feat_ai.squeeze(0).cpu().numpy())
        y_list.append(1)

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


train_names, test_names = img_split.img_splt("data/sample_real", train_size=49, seed=123)
X_train, y_train = build_xy(train_names)
X_test, y_test = build_xy(test_names)

#print("train:", X_train.shape, y_train.shape, "counts:", (y_train==0).sum(), (y_train==1).sum())
#print("test :", X_test.shape, y_test.shape, "counts:", (y_test==0).sum(), (y_test==1).sum())


