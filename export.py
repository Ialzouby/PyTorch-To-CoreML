
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
                    
