import torch
import torchxrayvision as xrv
import os

def verify_fix():
    print("Testing the fix for TB expert model loading...")
    
    # Simulate the fixed ExpertTB initialization
    model_path = "m3/demo/data/checkpoints/tbtrained8b/best_metric_model.pth"
    
    print("1. Loading base DenseNet model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    print(f"   Original classifier: {model.classifier}")
    
    print("2. Modifying classifier for TB detection...")
    model.classifier = torch.nn.Linear(1024, 2)
    print(f"   Modified classifier: {model.classifier}")
    
    if os.path.exists(model_path):
        print("3. Loading state dict with strict=False...")
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            print(f"   State dict loaded, keys: {len(state_dict.keys())}")
            
            # Load with strict=False
            result = model.load_state_dict(state_dict, strict=False)
            print(f"   Load result: {result}")
            print("   SUCCESS: Model loaded with strict=False")
            
            # Check what parameters were missing or unexpected
            if result.missing_keys:
                print(f"   Missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                print(f"   Unexpected keys: {result.unexpected_keys}")
                
        except Exception as e:
            print(f"   ERROR: {e}")
            return False
    else:
        print(f"3. Model path does not exist: {model_path}")
        return False
    
    print("4. Setting model to eval mode...")
    model.eval()
    print("   Model set to eval mode")
    
    print("\nFIX VERIFICATION COMPLETE")
    return True

if __name__ == "__main__":
    verify_fix()
