import os
import numpy as np
import torch
import torchxrayvision as xrv
from experts.base_expert import BaseExpert
from experts.utils import get_bbox_from_mask, get_organ_segmentation, post_process_text
from experts.expert_tb import ExpertTB

class TorchXRayVisionExpert(BaseExpert):
    def __init__(self, device=0, tb_expert=None):
        super().__init__()
        self.device = f"cuda:{device}"
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.to(self.device)
        self.model.eval()
        self.pathologies = self.model.pathologies
        self.tb_expert = tb_expert

    def mentioned_by(self, text: str) -> bool:
        """Check if this expert is mentioned in the text"""
        cxr_keywords = ["cxr", "chest x-ray", "chest xray", "x-ray", "xray", "radiograph"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in cxr_keywords)

    def run(self, image_url=None, input=None, output_dir=None, img_file=None, slice_index=None, prompt=None):
        """Main run method that processes the image and returns results"""
        # Use img_file if provided, otherwise use image_url
        image_path = img_file if img_file else image_url
        
        if not image_path or not os.path.exists(image_path):
            return "Error: Image file not found", None, ""
        
        try:
            # Load and preprocess image
            img = xrv.utils.load_image(image_path)
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                pred = self.model(img)
            
            # Process predictions
            output = {}
            for i, pathology in enumerate(self.pathologies):
                output[pathology] = pred[0, i].item()
            
            # Find significant findings (probability > 0.5)
            findings = []
            for pathology, prob in output.items():
                if prob > 0.5:
                    findings.append(f"{pathology} ({prob:.2f})")
            
            # Get TB predictions if TB expert is available
            if self.tb_expert:
                try:
                    tb_result_text, _, _ = self.tb_expert.run(image_url=image_path)
                    # Parse the TB result to extract probability
                    if "TB:" in tb_result_text:
                        # Extract TB probability from the result text
                        import re
                        tb_match = re.search(r'TB: ([0-9.]+)', tb_result_text)
                        if tb_match:
                            tb_prob = float(tb_match.group(1))
                            if tb_prob > 0.5:
                                findings.append(f"TB ({tb_prob:.2f})")
                except Exception as e:
                    print(f"TB expert error: {e}")
            
            # Format output
            if findings:
                result_text = f"CXR Analysis Results:\n" + "\n".join([f"- {finding}" for finding in findings])
            else:
                result_text = "CXR Analysis: No significant pathological findings detected."
            
            return result_text, None, ""
            
        except Exception as e:
            return f"Error processing chest X-ray: {str(e)}", None, ""

    def __call__(self, image_path, seg_image=None, question=""):
        """Legacy call method for backward compatibility"""
        try:
            img = xrv.utils.load_image(image_path)
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(img)
            
            output = {}
            for i, pathology in enumerate(self.pathologies):
                output[pathology] = pred[0, i].item()
            
            question = question.lower()
            if "lung" in question and seg_image is not None:
                lung_mask = get_organ_segmentation(seg_image, "lung")
                if lung_mask is not None:
                    lung_bbox = get_bbox_from_mask(lung_mask)
                    output["lung_bbox"] = lung_bbox.tolist()
            
            findings = []
            for pathology, prob in output.items():
                if prob > 0.5:
                    findings.append(pathology)
            
            # Get TB findings if available
            if self.tb_expert:
                try:
                    tb_result_text, _, _ = self.tb_expert.run(image_url=image_path)
                    # Parse the TB result to extract probability
                    if "TB:" in tb_result_text:
                        # Extract TB probability from the result text
                        import re
                        tb_match = re.search(r'TB: ([0-9.]+)', tb_result_text)
                        if tb_match:
                            tb_prob = float(tb_match.group(1))
                            if tb_prob > 0.5:
                                findings.append("TB")
                except Exception as e:
                    print(f"TB expert error: {e}")
            
            findings = list(set(findings))
            return post_process_text(", ".join(findings))
            
        except Exception as e:
            return f"Error: {str(e)}"
