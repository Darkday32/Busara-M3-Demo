import torch
import torchxrayvision as xrv
import os

def debug_model_loading():
    # Try to load the model as it's done in expert_tb.py
    print("Loading model with torchxrayvision...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    print(f"Original model classifier: {model.classifier}")
    print(f"Original classifier shape: {model.classifier.weight.shape}")
    
    # Modify the classifier as done in ExpertTB
    model.classifier = torch.nn.Linear(1024, 2)
    print(f"Modified model classifier: {model.classifier}")
    print(f"Modified classifier shape: {model.classifier.weight.shape}")
    
    # Try to load the state dict
    model_path = "m3/demo/data/checkpoints/tbtrained8b/best_metric_model.pth"
    if os.path.exists(model_path):
        print(f"Model path exists: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            print(f"State dict keys: {list(state_dict.keys())[:5]}...")  # Show first 5 keys
            print(f"Classifier weight shape in state dict: {state_dict['classifier.weight'].shape}")
            print(f"Classifier bias shape in state dict: {state_dict['classifier.bias'].shape}")
            
            # Try to load the state dict
            model.load_state_dict(state_dict)
            print("Successfully loaded state dict!")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            
            # Try loading with strict=False to see what's mismatched
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Loaded with strict=False - some parameters may be missing or unexpected")
            except Exception as e2:
                print(f"Error even with strict=False: {e2}")
    else:
        print(f"Model path does not exist: {model_path}")

if __name__ == "__main__":
    debug_model_loading()
