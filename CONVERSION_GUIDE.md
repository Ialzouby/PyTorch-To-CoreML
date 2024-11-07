# Wav2Lip iOS Integration - Changes Documented

## 📝 Overview
This document highlights the modifications made to integrate the Wav2Lip model into an iOS application using CoreML, with the conversion from PyTorch handled through ONNX.

## 🔄 Changes Made
### Model Conversion from PyTorch to CoreML
The following changes were made to convert the Wav2Lip PyTorch model into CoreML format:

```python
import torch
from models.wav2lip import Wav2Lip
import onnx
from onnx_coreml import convert

# Step 1: Export PyTorch Model to ONNX
model = Wav2Lip()
checkpoint = torch.load('/path/to/checkpoint.pth', map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
else:
    model.load_state_dict(checkpoint)
model.eval()

# Create dummy inputs
dummy_audio_input = torch.randn(1, 1, 80, 16)
dummy_face_input = torch.randn(1, 6, 96, 96)

# Export the model to ONNX
torch.onnx.export(model, (dummy_audio_input, dummy_face_input), 'wav2lip.onnx')

# Step 2: Convert ONNX Model to CoreML
onnx_model_path = '/path/to/wav2lip.onnx'
onnx_model = onnx.load(onnx_model_path)
coreml_model = convert(onnx_model)
coreml_model.save('wav2lip.mlmodel')
