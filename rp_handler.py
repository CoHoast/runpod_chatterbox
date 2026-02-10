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
        audio_prompt_base64 = input_data.get('audio_prompt_base64')
        
        # Handle voice cloning
        audio_prompt_path = None
        if audio_prompt_base64:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(base64.b64decode(audio_prompt_base64))
                audio_prompt_path = tmp.name
            print("Using voice clone from uploaded audio")
        
        print(f"Generating: {text[:50]}...")
        audio_tensor = model.generate(
            text, 
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration, 
            cfg_weight=cfg_weight
        )
        
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)
        
        audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
        return {"status": "success", "audio_base64": audio_base64, "sample_rate": model.sr}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    initialize_model()
    runpod.serverless.start({'handler': handler})
