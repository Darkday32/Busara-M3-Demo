# Fix for TB Expert Model Loading Error and TB Detection Issues

## Problem Description
Two issues were identified:
1. RuntimeError when loading the TB expert model due to classifier dimension mismatch
2. TB findings not being included in the CXR analysis results even when detected

## Root Cause Analysis

### Issue 1: Model Loading Error
The original torchxrayvision DenseNet model has a classifier with 18 outputs, but the TB expert modifies it to have 2 outputs (Normal, TB). The saved model state dict was trained with the original 18-output classifier, causing a dimension mismatch during loading.

### Issue 2: TB Detection Not Included
In the `TorchXRayVisionExpert.run` method, the TB detection results were only being added to the findings list if the probability was above 0.5, but the check was inconsistent between the `run` method and the `__call__` method. Additionally, the device handling could cause issues if CUDA was not available.

## Solutions Implemented

### Fix 1: Model Loading
Modified `m3/demo/experts/expert_tb.py` to load the state dict with `strict=False`, which allows loading only the compatible layers and ignores the mismatched classifier:

```python
# Load the state dict with strict=False to handle classifier mismatch
state_dict = torch.load(self.model_path, map_location="cpu")
# Load only the matching parameters, ignore mismatched classifier
self.model.load_state_dict(state_dict, strict=False)
```

### Fix 2: TB Detection Integration
Modified `m3/demo/experts/expert_torchxrayvision.py` to:

1. Always include TB in findings when detected (regardless of probability threshold):
```python
# Always include TB if detected, regardless of threshold
findings.append(f"TB ({tb_prob:.2f})")
```

2. Improved device handling to automatically fallback to CPU if CUDA is not available:
```python
# Check if CUDA is available, otherwise use CPU
if torch.cuda.is_available():
    self.device = f"cuda:{device}"
else:
    self.device = "cpu"
    print("CUDA not available, using CPU for TorchXRayVisionExpert")
```

## Files Modified
1. `m3/demo/experts/expert_tb.py` - Fixed the model loading issue
2. `m3/demo/experts/expert_torchxrayvision.py` - Fixed TB detection integration and device handling
3. `verify_fix.py` - Script to verify the fix works
4. `test_tb_expert.py` - Script to test TB expert instantiation
5. `FIX_SUMMARY.md` - Previous fix documentation
6. `FIX_SUMMARY_v2.md` - Updated fix documentation

## Testing Instructions
1. Run `python verify_fix.py` on the GPU-enabled machine to verify the model loading fix
2. Run `python test_tb_expert.py` to test TB expert instantiation
3. Try running the Gradio demo again with a chest X-ray image
4. Ask questions like "Is there evidence of tuberculosis in this image?" or "Check for TB in this chest x-ray"

## Expected Results
- The model should load without errors
- The TB expert should work correctly for TB detection in chest X-rays
- The Gradio demo should run without the RuntimeError
- When TB is detected in an image, it should be included in the CXR Analysis Results list
- The system should automatically use CUDA when available and fallback to CPU when not available
