import os

import numpy as np
import torch
import torchxrayvision as xrv

from .base_expert import BaseExpert
from .utils import get_bbox_from_mask, get_organ_segmentation, post_process_text
from .expert_tb import ExpertTB


class TorchXRayVisionExpert(BaseExpert):
    def __init__(self, device=0, tb_expert=None):
        super().__init__()
        self.device = f"cuda:{device}"
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.to(self.device)
        self.model.eval()
        self.pathologies = self.model.pathologies
        self.tb_expert = tb_expert

    def __call__(self, image_path, seg_image, question):
        img = xrv.utils.load_image(image_path)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img)

        output = {}
        for i, pathology in enumerate(self.pathologies):
            output[pathology] = pred[0, i].item()

        question = question.lower()
        if "lung" in question:
            lung_mask = get_organ_segmentation(seg_image, "lung")
            if lung_mask is not None:
                lung_bbox = get_bbox_from_mask(lung_mask)
                output["lung_bbox"] = lung_bbox.tolist()

        findings = []
        for pathology, prob in output.items():
            if prob > 0.5:
                findings.append(pathology)

        tb_findings = self.tb_expert(image_path)
        findings.extend(tb_findings)
        findings = list(set(findings))

        return post_process_text(", ".join(findings))
