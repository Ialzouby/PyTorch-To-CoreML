# Wav2Lip iOS Integration - Changes Documented

## 📝 Overview
This document highlights the modifications made to integrate the Wav2Lip model into an iOS application using CoreML, with the conversion from PyTorch handled through ONNX.

---

## 🔄 Changes Made

### 1. Model Conversion from PyTorch to CoreML
- **Added Scripts for Model Conversion**:
  - **`export.py`**: Script to export the Wav2Lip PyTorch model to ONNX format.
  - **`coremlconversion.py`**: Script to convert the ONNX model to CoreML format.

---

### 📂 `export.py`
```python
import torch
from models.wav2lip import Wav2Lip

# Initialize and load your model
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
