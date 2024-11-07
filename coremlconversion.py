
        import onnx
        from onnx_coreml import convert
        
        # Path to the ONNX model
        onnx_model_path = '/path/to/wav2lip.onnx'
        
        # Load the ONNX model
        onnx_model = onnx.load(onnx_model_path)
        
        # Convert ONNX to CoreML
        coreml_model = convert(onnx_model)
        
        # Save the CoreML model
        coreml_model.save('wav2lip.mlmodel')
                    
