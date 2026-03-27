import os
import tempfile
from flask import Flask, request, jsonify, send_file
from infer import InferenceEngine

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'saved_models/enhanced_attention_unet_20260326_200212_best.pth')

app = Flask(__name__)
engine = None

def init_engine():
    global engine
    print(f"[INFO] Loading model from {CHECKPOINT_PATH}")
    engine = InferenceEngine(CHECKPOINT_PATH)
    print("[INFO] Model loaded successfully")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': engine is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    threshold = request.form.get('threshold', 0.5, type=float)
    
    try:
        image_file = request.files['image']
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
            image_file.save(tmp_input.name)
            tmp_input_path = tmp_input.name
        
        mask, prob = engine.predict(tmp_input_path, threshold=threshold)
        
        os.unlink(tmp_input_path)
        
        mask_img = Image.fromarray(mask * 255)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_output:
            mask_img.save(tmp_output.name)
            tmp_output_path = tmp_output.name
        
        return send_file(tmp_output_path, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-overlay', methods=['POST'])
def predict_overlay():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    threshold = request.form.get('threshold', 0.5, type=float)
    
    try:
        image_file = request.files['image']
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
            image_file.save(tmp_input_path := tmp_input.name)
        
        mask, prob = engine.predict(tmp_input_path, threshold=threshold)
        
        os.unlink(tmp_input_path)
        
        original = Image.open(image_file).convert('RGB')
        original = original.resize(mask.shape[::-1])
        
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
        colored_mask[mask == 1] = [255, 0, 0, 128]
        
        overlay_image = Image.fromarray(colored_mask, 'RGBA')
        result = Image.alpha_composite(original.convert('RGBA'), overlay_image)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_output:
            result.save(tmp_output_path := tmp_output.name)
        
        return send_file(tmp_output_path, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_engine()
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
