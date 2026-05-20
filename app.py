"""
NeuroCanvas — Flask Web App
===========================
Run: python app.py
Open: http://localhost:5000
"""

import os
import uuid
import torch
from flask import Flask, request, jsonify, send_file, render_template
from torchvision import transforms
from PIL import Image
import io

from utils.models import VGGEncoder, Decoder
from utils.utils import adapative_instance_normalization

# ── Config ────────────────────────────────────────────────────────────────────
DECODER_PATH = 'experiments/experiment_gpu/decoder_11.pth'
VGG_PATH     = 'vgg_normalised.pth'
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app    = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load models once at startup ───────────────────────────────────────────────
print('Loading models...')
encoder = VGGEncoder(VGG_PATH).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
encoder.eval()
decoder.eval()
print('Models loaded!')

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_image(file_bytes, size=512):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor)


def run_style_transfer(content_bytes, style_bytes, alpha=1.0, size=512):
    content = load_image(content_bytes, size).to(device)
    style   = load_image(style_bytes,   size).to(device)

    with torch.no_grad():
        c_feats = encoder(content)
        s_feats = encoder(style)
        t = adapative_instance_normalization(c_feats[-1], s_feats[-1])
        t = alpha * t + (1.0 - alpha) * c_feats[-1]
        output = decoder(t)

    return tensor_to_pil(output)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stylize', methods=['POST'])
def stylize():
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({'error': 'Both content and style images required'}), 400

    content_file = request.files['content']
    style_file   = request.files['style']
    alpha        = float(request.form.get('alpha', 1.0))
    size         = int(request.form.get('size', 512))

    content_bytes = content_file.read()
    style_bytes   = style_file.read()

    try:
        result = run_style_transfer(content_bytes, style_bytes, alpha, size)
        result_id   = str(uuid.uuid4())[:8]
        result_path = os.path.join(RESULT_FOLDER, f'{result_id}.png')
        result.save(result_path)
        return jsonify({'result_url': f'/static/results/{result_id}.png'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)