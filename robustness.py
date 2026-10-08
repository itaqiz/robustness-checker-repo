import io
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import models, transforms

SEVERITIES = {
    'blur':      [1, 2, 3, 4, 6],
    'low_light': [0.7, 0.5, 0.35, 0.25, 0.15],
    'noise':     [8, 16, 28, 40, 55],
    'jpeg':      [60, 40, 25, 15, 10],
    'occlusion': [0.05, 0.10, 0.20, 0.30, 0.40],
}
LABELS = {'blur': 'Blur', 'low_light': 'Low light', 'noise': 'Noise',
          'jpeg': 'JPEG compression', 'occlusion': 'Blocked region'}
CLASSES = ['fresh', 'rotten']

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BASE_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
FINAL_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def corrupt(img, kind, level, seed=0):
    if level == 0:
        return img
    p = SEVERITIES[kind][level - 1]

    if kind == 'blur':
        return img.filter(ImageFilter.GaussianBlur(p))

    if kind == 'low_light':
        return ImageEnhance.Brightness(img).enhance(p)

    if kind == 'noise':
        rng = np.random.default_rng(seed)
        arr = np.asarray(img).astype(np.float32)
        arr = arr + rng.normal(0, p, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if kind == 'jpeg':
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=p)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    if kind == 'occlusion':
        w, h = img.size
        side = int((p * w * h) ** 0.5)
        rng = np.random.default_rng(seed)
        x0 = int(rng.integers(0, w - side + 1))
        y0 = int(rng.integers(0, h - side + 1))
        out = img.copy()
        out.paste((0, 0, 0), (x0, y0, x0 + side, y0 + side))
        return out

    raise ValueError(kind)


def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def load_config(path):
    with open(path) as f:
        return json.load(f)


def analyze(model, img, config, level=3, seed=0):
    threshold = config['threshold']
    probe = config['probes'][0]

    base = BASE_TF(img.convert('RGB'))
    views = [('Original', base)]
    for kind in SEVERITIES:
        views.append((f"{LABELS[kind]} (level {level})", corrupt(base, kind, level, seed)))
    views.append(('probe', corrupt(base, probe['kind'], probe['level'], seed)))

    batch = torch.stack([FINAL_TF(v) for _, v in views])
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)

    orig_pred = int(probs[0].argmax())
    rows = []
    for (name, im), p in zip(views[:-1], probs[:-1]):
        pred, conf = int(p.argmax()), float(p.max())
        notes = []
        if pred != orig_pred:
            notes.append('prediction flipped')
        if conf < threshold:
            notes.append('low confidence')
        rows.append({'name': name, 'image': im, 'prediction': CLASSES[pred],
                     'confidence': conf, 'note': ', '.join(notes) or 'ok'})

    reasons = []
    if rows[0]['confidence'] < threshold:
        reasons.append(f"confidence {rows[0]['confidence']:.2f} is below {threshold}")
    if int(probs[-1].argmax()) != orig_pred:
        reasons.append(f"prediction flips under a mild {LABELS[probe['kind']].lower()}")
    return rows, bool(reasons), reasons
