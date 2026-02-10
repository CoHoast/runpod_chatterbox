import runpod
import torchaudio 
import os
import tempfile
import base64
from chatterbox.tts import ChatterboxTTS

model = None

def initialize_model():
    global model
    if model is not None:
        return model
    print("Initializing ChatterboxTTS model...")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    print("Model initialized!")
    return model

def audio_tensor_to_base64(audio_tensor, sample_rate):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
        with open(tmp_file.name, 'rb') as f:
            audio_data = f.read()
        os.unlink(tmp_file.name)
        return base64.b64encode(audio_data).decode('utf-8')

def handler(event):
    global model
    try:
        input_data = event.get('input', {})
        text = input_data.get('text')
        if not text:
            return {"error": "Missing 'text' parameter"}
        
        exaggeration = float(input_data.get('exaggeration', 0.5))
        cfg_weight = float(input_data.get('cfg_weight', 0.5))
        
        print(f"Generating: {text[:50]}...")
        audio_tensor = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight)
        audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
        
        return {"status": "success", "audio_base64": audio_base64, "sample_rate": model.sr}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    initialize_model()
    runpod.serverless.start({'handler': handler})
