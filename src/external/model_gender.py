# From https://github.com/ndb796/Face-Gender-Classification-PyTorch
# Training in https://colab.research.google.com/drive/12sE3HtO6coTOSlQFjhL-G-YNsOnRMXkM
import copy

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class ModelGen:
    def __init__(self, device='cpu'):
        self.model = None
        self.device = device
        self.transforms = None
        # Init variables
        self.__build_model__()

    @torch.no_grad()
    def __build_model__(self):
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # binary classification (num_of_class == 2)
        model = model.to(self.device).eval()

        model.load_state_dict(torch.load('src/external/model_gender_ResNet18-UTKFace_15.pth'))  # Better than rescaling to 224 (original)
        # model.load_state_dict(torch.load('src/external/model_gender_ResNet18-asian_16-128_11.pth'))  # Better than rescaling to 224 (original)
        # model.load_state_dict(torch.load('src/external/model_gender_ResNet18-asian_16-128_20.pth'))  # Better than rescaling to 224 (original)
        # model.load_state_dict(torch.load('src/external/model_gender_ResNet18-asian_16-128.pth'))  # Better than rescaling to 224 (original)
        # model.load_state_dict(torch.load('src/external/model_gender_ResNet18_16-128.pth'))  # Better than rescaling to 224 (original)
        # model.load_state_dict(torch.load('src/external/model_gender_ResNet18_64.pth'))  # Better than rescaling to 224 (original)
        # model.load_state_dict(torch.load('src/external/model_gender_ResNet18_32.pth'))  # Better than rescaling to 224 (original)
        self.model = model

    def __call__(self, input_tensor):
        # Predict gender
        emb = self.model(input_tensor)
        prob = F.softmax(emb).detach().cpu().numpy()
        return emb, prob


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device object
    gender_classifier = ModelGen(device=device)

    transforms_val = transforms.Compose([
        # transforms.Resize((128, 128)),
        transforms.Resize((64, 64)),
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_male = Image.open('src/external/test_face_male.png', 'r')
    img_male = img_male.convert('RGB')
    # img_male = np.array(img_male)
    # img_male = img_male[:, :, ::-1]
    # img_male = Image.fromarray(img_male)
    img_input_male = transforms_val(img_male)
    img_female = Image.open('src/external/test_face_female.png', 'r')
    img_female = img_female.convert('RGB')
    # img_female = np.array(img_female)
    # img_female = img_female[:, :, ::-1]
    # img_female = Image.fromarray(img_female)
    img_input_female = transforms_val(img_female)
    img_female2 = Image.open('src/external/test_face_female2.png', 'r')
    img_female2 = img_female2.convert('RGB')
    img_input_female2 = transforms_val(img_female2)
    img_input = torch.stack((img_input_male, img_input_female, img_input_female2)).to(device)

    embs, probs = gender_classifier(img_input)

    print(probs)


if __name__ == '__main__':
    main()


