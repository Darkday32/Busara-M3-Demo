import os
import torch
import torch.nn.functional as F
import numpy as np
import skimage.io
import skimage.transform

from .base_expert import BaseExpert

import torchxrayvision as xrv

class ExpertTB(BaseExpert):
    def __init__(self, model_path="/data/checkpoints/tbtrained8b/bestmetricmodel.pth"):
        super().__init__()
        self.expert_name = "tb"
        self.labels = ["Normal", "TB"]
        self.model_path = model_path

        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.op_threshs = None
        self.model.classifier = torch.nn.Linear(1024, 2)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        else:
            print(f"Warning: Model path does not exist: {self.model_path}")

        self.model.eval()

    def get_expert_name(self) -> str:
        return self.expert_name

    def get_prediction(self, image_path: str) -> dict:
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)

        if len(img.shape) > 2:
            img = img[:, :, 0]
        elif len(img.shape) < 2:
            raise ValueError("Image dimension lower than 2")

        img = img[None, :, :]

        transform = xrv.datasets.XRayCenterCrop()
        img = transform(img)
        img = torch.from_numpy(img).unsqueeze(0).float()

        with torch.no_grad():
            outputs = self.model(img)
            probs = F.softmax(outputs, dim=1)

        prediction = {label: probs[0, i].item() for i, label in enumerate(self.labels)}
        return prediction

    def run(self, image_path: str) -> dict:
        return self.get_prediction(image_path)

    def mentioned_by(self, text: str) -> bool:
        return self.expert_name in text.lower()
