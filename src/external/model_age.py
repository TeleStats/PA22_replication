# Adapted from https://github.com/yu4u/age-estimation-pytorch
import numpy as np
import pretrainedmodels as pm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelAge:
    def __init__(self, device='cpu'):
        self.model = None
        self.device = device
        # Init variables
        self.__build_model__()

    def __build_model__(self):
        model = pm.se_resnext50_32x4d(pretrained=None)
        dim_feats = model.last_linear.in_features
        model.last_linear = nn.Linear(dim_feats, 101)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        print('Loading Age model')
        checkpoint = torch.load('src/external/model_age_gender.pth')
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model

    def __call__(self, faces):
        # predict age
        inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(self.device)
        outputs = F.softmax(self.model(inputs), dim=-1).cpu().numpy()
        ages = np.arange(0, 101)
        predicted_ages = (outputs * ages).sum(axis=-1)
        print('h')


def main():
    model = ModelAge()
    img = cv2.imread('src/external/test_face.png')
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aa = model(input_img)


if __name__ == '__main__':
    import cv2
    main()
