import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
import gdown
import os
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 
    "torch==2.0.1+cpu", "torchvision==0.15.2+cpu",
    "--index-url", "https://download.pytorch.org/whl/cpu"])

if not os.path.exists('best_model.pth'):
    st.info('Downloading model weights...')
    gdown.download(
        'https://drive.google.com/file/d/1GDoVyuQYzUgG5CFzQ-VE0jSr6LXTv85b/view?usp=sharing',
        'best_model.pth', fuzzy=True
    )

# ===== CONFIG =====
IMG_SIZE      = 256
CROP_SIZE     = 384
SHAPE_CLASSES = ['Cross', 'L-Shape', 'Square']
NUM_CLASSES   = 3
CLASS_TO_IDX  = {c: i for i, c in enumerate(SHAPE_CLASSES)}
IDX_TO_CLASS  = {i: c for c, i in CLASS_TO_IDX.items()}

# ===== MODEL =====
class GCPNet(nn.Module):
    def __init__(self, backbone='efficientnet_b0', num_classes=3, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool='avg')
        fd = self.backbone.num_features
        self.dropout  = nn.Dropout(0.3)
        self.reg_head = nn.Sequential(nn.Linear(fd,256), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(256,2), nn.Sigmoid())
        self.cls_head = nn.Sequential(nn.Linear(fd,256), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(256,num_classes))

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.reg_head(f), self.cls_head(f)


@st.cache_resource
def load_model(weights_path):
    model = GCPNet()
    ckpt  = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def preprocess(crop):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img  = img.astype(np.float32) / 255.0
    img  = (img - mean) / std
    img  = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img


def predict(image_np, model):
    h, w = image_np.shape[:2]
    cs = CROP_SIZE
    stride = cs // 2

    crops = []; offsets = []
    ys = list(range(0, h - cs, stride)) + [h - cs]
    xs = list(range(0, w - cs, stride)) + [w - cs]

    for y1 in ys:
        for x1 in xs:
            y1c = max(0, min(y1, h - cs))
            x1c = max(0, min(x1, w - cs))
            crop = image_np[y1c:y1c+cs, x1c:x1c+cs]
            crops.append(preprocess(crop))
            offsets.append((x1c, y1c))

    best_conf = -1; best = None
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = torch.cat(crops[i:i+batch_size])
            kp_p, cls_p = model(batch)
            probs = torch.softmax(cls_p, dim=1)
            confs, preds = probs.max(dim=1)
            for j in range(len(batch)):
                conf = confs[j].item()
                if conf > best_conf:
                    best_conf = conf
                    ox, oy = offsets[i+j]
                    best = {
                        'x': float(kp_p[j,0].item() * cs + ox),
                        'y': float(kp_p[j,1].item() * cs + oy),
                        'shape': IDX_TO_CLASS[preds[j].item()],
                        'confidence': conf
                    }
    return best


# ===== UI =====
st.set_page_config(page_title='GCP Marker Detection', page_icon='🎯', layout='wide')

st.title('🎯 GCP Marker Detection')
st.markdown('**Aerial Ground Control Point Detection** — Keypoint Localization + Shape Classification')
st.markdown('---')

# Sidebar
st.sidebar.title('Settings')
weights_path = st.sidebar.text_input('Model weights path', value='best_model.pth')
st.sidebar.markdown('---')
st.sidebar.markdown('**Classes:**')
st.sidebar.markdown('- 🔵 Cross\n- 🟢 L-Shape\n- 🔴 Square')

# Load model
if Path(weights_path).exists():
    model = load_model(weights_path)
    st.sidebar.success('✅ Model loaded!')
else:
    st.sidebar.error('❌ Model weights not found!')
    st.sidebar.info('Place best_model.pth in the same folder as app.py')
    model = None

# Upload
st.subheader('Upload Aerial Image')
uploaded = st.file_uploader('Choose a JPG image', type=['jpg', 'jpeg', 'png'])

if uploaded and model:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Input Image')
        st.image(img_rgb, caption=f'Size: {img_rgb.shape[1]}×{img_rgb.shape[0]}', use_column_width=True)

    # Predict
    with st.spinner('Detecting GCP marker...'):
        result = predict(img_rgb, model)

    with col2:
        st.subheader('Detection Result')

        # Draw on image
        vis = img_rgb.copy()
        cx, cy = int(result['x']), int(result['y'])

        # Draw crosshair
        cv2.circle(vis, (cx, cy), 20, (255, 0, 0), 3)
        cv2.line(vis, (cx-30, cy), (cx+30, cy), (255, 0, 0), 2)
        cv2.line(vis, (cx, cy-30), (cx, cy+30), (255, 0, 0), 2)

        st.image(vis, caption='Detected GCP center (red crosshair)', use_column_width=True)

    # Results
    st.markdown('---')
    st.subheader('📊 Prediction Results')

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric('X Coordinate', f'{result["x"]:.1f} px')
    with col4:
        st.metric('Y Coordinate', f'{result["y"]:.1f} px')
    with col5:
        st.metric('Shape Class', result['shape'])

    st.metric('Confidence', f'{result["confidence"]*100:.1f}%')

    # JSON output
    st.subheader('📄 JSON Output')
    output = {
        uploaded.name: {
            'mark': {'x': result['x'], 'y': result['y']},
            'verified_shape': result['shape']
        }
    }
    st.json(output)

    # Download
    st.download_button(
        '⬇️ Download prediction JSON',
        data=json.dumps(output, indent=2),
        file_name='prediction.json',
        mime='application/json'
    )

elif uploaded and not model:
    st.error('Please provide valid model weights path in the sidebar!')

else:
    st.info('👆 Upload an aerial image to detect the GCP marker')
    st.markdown('''
    ### How it works:
    1. Upload a high-resolution aerial image (JPG)
    2. The model slides a window across the image
    3. Detects the GCP marker center coordinates (x, y)
    4. Classifies the marker shape (Cross / L-Shape / Square)
    5. Returns results in JSON format
    ''')
