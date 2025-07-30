import os
import torch
import torch.nn.functional as F
import numpy as np
import skimage.io
import skimage.transform

from .base_expert import BaseExpert

import torchxrayvision as xrv

class ExpertTB(BaseExpert):
    def __init__(self, model_path="/data/checkpoints/tbtrained8b/best_metric_model.pth"):
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

    def run(self, image_url=None, input=None, output_dir=None, img_file=None, slice_index=None, prompt=None):
        """Main run method that processes the image and returns results"""
        # Use img_file if provided, otherwise use image_url
        image_path = img_file if img_file else image_url
        
        if not image_path or not os.path.exists(image_path):
            return "Error: Image file not found for TB expert", None, ""
        
        try:
            prediction = self.get_prediction(image_path)
            tb_prob = prediction.get('TB', 0)
            
            if tb_prob > 0.5:
                result_text = f"TB Detection Results:\n- TB: {tb_prob:.2f}\n- Normal: {prediction.get('Normal', 0):.2f}"
            else:
                result_text = f"TB Detection Results:\n- TB: {tb_prob:.2f}\n- Normal: {prediction.get('Normal', 0):.2f}\nNo significant TB findings detected."
            
            return result_text, None, ""
            
        except Exception as e:
            return f"Error processing TB detection: {str(e)}", None, ""

    def mentioned_by(self, text: str) -> bool:
        """Check if this expert should be mentioned based on the text"""
        tb_keywords = ["tb", "tuberculosis", "lung conditions", "pulmonary"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in tb_keywords)
