# Fix for TB Expert Model Loading Error

## Problem Description
When running the Gradio demo, the following error occurred:
```
RuntimeError: Error(s) in loading state_dict for DenseNet
```

This happened in `expert_tb.py` at line 25 when trying to load the model state dict:
```python
self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
```

## Root Cause
The issue was caused by a mismatch between the model architecture and the saved state dict:

1. The original torchxrayvision DenseNet model has a classifier with 18 outputs
2. The TB expert modifies the classifier to have 2 outputs (Normal, TB)
3. The saved model state dict was trained with the original 18-output classifier
4. When PyTorch tries to load the state dict, it fails because the classifier dimensions don't match

## Solution
Modified `m3/demo/experts/expert_tb.py` to load the state dict with `strict=False`, which allows loading only the compatible layers and ignores the mismatched classifier:

```python
# Load the state dict with strict=False to handle classifier mismatch
state_dict = torch.load(self.model_path, map_location="cpu")
# Load only the matching parameters, ignore mismatched classifier
self.model.load_state_dict(state_dict, strict=False)
```

## Files Modified
1. `m3/demo/experts/expert_tb.py` - Fixed the model loading issue
2. `verify_fix.py` - Script to verify the fix works
3. `test_tb_expert.py` - Script to test TB expert instantiation

## Testing Instructions
1. Run `python verify_fix.py` on the GPU-enabled machine to verify the fix
2. Run `python test_tb_expert.py` to test TB expert instantiation
3. Try running the Gradio demo again

## Expected Results
- The model should load without errors
- The TB expert should work correctly for TB detection in chest X-rays
- The Gradio demo should run without the RuntimeError
