"""
pipeline/pipeline.py — Pipeline de détection/segmentation de toitures

Classes :
  YOLODetector       : YOLO détection (class/yolo_classification), retourne boxes+labels+scores
  YOLOSegDetector    : YOLO segmentation (seg/yolo), retourne boxes+labels+scores+masks
  MaskRCNNDetector   : Mask R-CNN (seg/mark_r_cnn)
  DeepLabDetector    : DeepLabV3+ (seg/DeepLabV3)
  SoftWANMSEnsemble  : Fusion Soft+WA-NMS sur YOLOSeg + MaskRCNN + DeepLab
  Pipeline           : Orchestrateur principal

Routing par préfixe de nom de fichier :
  Snapshot*   → YOLODetector uniquement
  Production* → YOLODetector d'abord, puis SoftWANMSEnsemble (triplet Soft+WA-NMS)

Soft+WA-NMS triplet :
  1. Pool global des 3 modèles (YOLO-seg, Mask R-CNN, DeepLab), scores pondérés.
  2. Par classe : NMS par IoU masque → box = moyenne pondérée, mask = union pondérée, score = max.

Sorties :
  <output_dir>/annotated/        images annotées (.jpg)
  <output_dir>/descriptions/     descriptions JSON par image
  <output_dir>/summary.json      rapport global
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Décorateur no_grad compatible avec ou sans torch
if TORCH_AVAILABLE:
    _no_grad = torch.no_grad()
else:
    def _no_grad(f): return f  # no-op quand torch absent

try:
    from ultralytics import YOLO as _UltralyticsYOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from scipy import ndimage as _scipy_ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# CONSTANTES
# =============================================================================

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

# Classes de segmentation (ensemble : YOLO-seg + Mask R-CNN + DeepLabV3+)
SEG_CLASSES     = ['__background__', 'toiture_tole_ondulee', 'toiture_tole_bac', 'toiture_dalle']
SEG_CLASS_NAMES = SEG_CLASSES[1:]
SEG_NUM_CLASSES = len(SEG_CLASSES)

# Couleurs par classe (R, G, B) — couvre oblique + nadir + segmentation
DET_COLORS = {
    # nadir
    'panneau_solaire':       (255, 215,   0),
    # oblique
    'batiment_peint':        (255, 127,  80),
    'batiment_non_enduit':   (147, 112, 219),
    'batiment_enduit':       ( 64, 224, 208),
    'menuiserie_metallique': (255, 165,   0),
}
SEG_COLORS = {
    'toiture_tole_ondulee': ( 70, 130, 180),
    'toiture_tole_bac':     ( 60, 179, 113),
    'toiture_dalle':        (220,  20,  60),
}


# =============================================================================
# UTILITAIRES INTERNES
# =============================================================================

def _empty_pred() -> dict:
    return {
        'boxes':  np.zeros((0, 4), dtype=np.float32),
        'labels': np.zeros(0, dtype=np.int32),
        'scores': np.zeros(0, dtype=np.float32),
        'masks':  [],
    }


def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    m1b = m1.astype(bool); m2b = m2.astype(bool)
    inter = (m1b & m2b).sum()
    union = (m1b | m2b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _box_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = a1 + a2 - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def _weighted_mask(masks: list, weights: list) -> Optional[np.ndarray]:
    if not masks:
        return None
    w = np.array(weights, dtype=float)
    w /= w.sum() + 1e-8
    acc = np.zeros_like(masks[0], dtype=float)
    for m, wi in zip(masks, w):
        acc += wi * m.astype(float)
    return acc >= 0.5


def _semantic_to_instances(semantic_mask: np.ndarray, min_area: int = 100) -> list:
    """Convertit un masque sémantique (H×W) en liste d'instances."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy requis : pip install scipy")
    instances = []
    for class_id in range(1, SEG_NUM_CLASSES):
        binary = (semantic_mask == class_id)
        if not binary.any():
            continue
        labeled, n = _scipy_ndimage.label(binary)
        total_area = float(binary.sum())
        for i in range(1, n + 1):
            m = (labeled == i)
            area = int(m.sum())
            if area < min_area:
                continue
            row_idx = np.where(np.any(m, axis=1))[0]
            col_idx = np.where(np.any(m, axis=0))[0]
            y1, y2 = float(row_idx[0]), float(row_idx[-1])
            x1, x2 = float(col_idx[0]), float(col_idx[-1])
            instances.append({
                'mask':  m,
                'label': class_id,
                'box':   [x1, y1, x2, y2],
                'score': area / (total_area + 1e-8),
            })
    return instances


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    for candidate in ['arial.ttf', 'Arial.ttf', 'DejaVuSans.ttf', 'LiberationSans-Regular.ttf']:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


# =============================================================================
# MODÈLE 1 — YOLODetector (détection, class/yolo_classification)
# =============================================================================

class YOLODetector:
    """YOLO détection (class/yolo_classification).

    Retourne des boîtes de détection sans masques.
    Classes : panneau_solaire, batiment_peint, batiment_non_enduit, batiment_enduit
    """

    def __init__(self, model_path: str, threshold: float = 0.25):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics requis : pip install ultralytics")
        self.model = _UltralyticsYOLO(model_path)
        self.threshold = threshold
        self.class_names = list(self.model.names.values())  # lu depuis le modèle
        print(f"   [OK] YOLODetector       : {model_path}")

    def predict(self, img_path: str) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks': []}."""
        result = self.model(img_path, verbose=False)[0]
        boxes, labels, scores = [], [], []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                s = float(box.conf)
                if s < self.threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(int(box.cls) + 1)   # 0-indexed → 1-indexed global
                scores.append(s)
        if not boxes:
            return _empty_pred()
        return {
            'boxes':  np.array(boxes,  dtype=np.float32),
            'labels': np.array(labels, dtype=np.int32),
            'scores': np.array(scores, dtype=np.float32),
            'masks':  [],
        }


# =============================================================================
# MODÈLE 2 — YOLOSegDetector (segmentation, seg/yolo)
# =============================================================================

class YOLOSegDetector:
    """YOLO segmentation (seg/yolo).

    Retourne des masques d'instance.
    Classes : toiture_tole_ondulee, toiture_tole_bac, toiture_dalle
    """

    def __init__(self, model_path: str, threshold: float = 0.25):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics requis : pip install ultralytics")
        self.model = _UltralyticsYOLO(model_path)
        self.threshold = threshold
        self.class_names = SEG_CLASS_NAMES
        print(f"   [OK] YOLOSegDetector    : {model_path}")

    def predict(self, img_path: str) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks'}."""
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        result = self.model.predict(img_path, conf=self.threshold, verbose=False)[0]
        if result.masks is None or len(result.boxes) == 0:
            return _empty_pred()
        boxes  = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int) + 1
        scores = result.boxes.conf.cpu().numpy()
        masks  = []
        for raw in result.masks.data.cpu().numpy():
            m = Image.fromarray(raw.astype(np.uint8))
            m = m.resize((orig_w, orig_h), Image.NEAREST)
            masks.append(np.array(m) > 0)
        return {
            'boxes':  boxes.astype(np.float32),
            'labels': labels.astype(np.int32),
            'scores': scores.astype(np.float32),
            'masks':  masks,
        }


# =============================================================================
# MODÈLE 3 — MaskRCNNDetector (seg/mark_r_cnn)
# =============================================================================

class MaskRCNNDetector:
    """Mask R-CNN (seg/mark_r_cnn).

    Classes : toiture_tole_ondulee, toiture_tole_bac, toiture_dalle
    """

    def __init__(self, model_path: str, num_classes: int = 4,
                 threshold: float = 0.25, device: str = None):
        if not TORCH_AVAILABLE:
            raise ImportError("torch requis : pip install torch torchvision")
        self.device    = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold = threshold
        self.class_names = SEG_CLASS_NAMES

        ckpt  = torch.load(model_path, map_location=self.device, weights_only=False)
        state = ckpt['model_state_dict']
        try:
            nc = state['roi_heads.box_predictor.cls_score.bias'].shape[0]
        except KeyError:
            nc = num_classes

        model = maskrcnn_resnet50_fpn_v2(weights=None)
        in_f  = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, nc)
        in_fm = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, nc)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"   [OK] MaskRCNNDetector   : {model_path}  (classes={nc}, device={self.device})")

    @_no_grad
    def predict(self, img_path: str) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks'}."""
        img    = Image.open(img_path).convert('RGB')
        tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
        out    = self.model(tensor)[0]
        keep   = out['scores'] >= self.threshold
        if not keep.any():
            return _empty_pred()
        boxes  = out['boxes'][keep].cpu().numpy()
        labels = out['labels'][keep].cpu().numpy()
        scores = out['scores'][keep].cpu().numpy()
        raw_m  = out['masks'][keep].cpu().numpy()
        masks  = [raw_m[i, 0] > 0.5 for i in range(len(boxes))]
        return {
            'boxes':  boxes.astype(np.float32),
            'labels': labels.astype(np.int32),
            'scores': scores.astype(np.float32),
            'masks':  masks,
        }


# =============================================================================
# MODÈLE 4 — DeepLabDetector (seg/DeepLabV3)
# =============================================================================

class DeepLabDetector:
    """DeepLabV3+ (seg/DeepLabV3) — segmentation sémantique → instances.

    Classes : toiture_tole_ondulee, toiture_tole_bac, toiture_dalle
    """

    def __init__(self, model_path: str, backbone: str = 'resnet50',
                 num_classes: int = 4, image_size: int = 512,
                 device: str = None):
        if not TORCH_AVAILABLE:
            raise ImportError("torch requis : pip install torch torchvision")
        self.device     = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.image_size = image_size
        self.class_names = SEG_CLASS_NAMES

        ckpt  = torch.load(model_path, map_location=self.device, weights_only=False)
        state = ckpt['model_state_dict']
        try:
            nc = state['classifier.4.weight'].shape[0]
        except KeyError:
            nc = num_classes

        model = (deeplabv3_resnet101(weights=None) if backbone == 'resnet101'
                 else deeplabv3_resnet50(weights=None))
        model.classifier = DeepLabHead(2048, nc)
        if any(k.startswith('aux_classifier') for k in state.keys()):
            model.aux_classifier = nn.Sequential(
                nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(256, nc, 1))
        model.load_state_dict(state)
        model.aux_classifier = None
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"   [OK] DeepLabDetector    : {model_path}  (backbone={backbone}, classes={nc}, device={self.device})")

    @_no_grad
    def predict(self, img_path: str) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks'}."""
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img_r = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.normalize(
            TF.to_tensor(img_r),
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        out = self.model(tensor.unsqueeze(0).to(self.device))
        pred_sm = torch.argmax(out['out'], dim=1).squeeze().cpu().numpy()
        pred_mask = np.array(
            Image.fromarray(pred_sm.astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST))
        instances = _semantic_to_instances(pred_mask, min_area=100)
        if not instances:
            return _empty_pred()
        return {
            'boxes':  np.array([i['box']   for i in instances], dtype=np.float32),
            'labels': np.array([i['label'] for i in instances], dtype=np.int32),
            'scores': np.array([i['score'] for i in instances], dtype=np.float32),
            'masks':  [i['mask'] for i in instances],
        }


# =============================================================================
# ENSEMBLE — SOFT + WA-NMS TRIPLET
# =============================================================================

class SoftWANMSEnsemble:
    """Ensemble Soft+WA-NMS sur YOLO-seg + Mask R-CNN + DeepLabV3+.

    Algorithme :
      1. Pool global des prédictions des 3 modèles (scores pondérés par modèle).
      2. Par classe, tri décroissant par score puis NMS sur IoU masque :
         - box   = moyenne pondérée par score du cluster
         - mask  = moyenne pondérée par score (seuil 0.5)
         - score = max du cluster
      3. Filtrage final par conf_thr.

    Poids par défaut : YOLO-seg=1.0, Mask R-CNN=1.0, DeepLab=0.8
    (DeepLab légèrement sous-pondéré car pas de vrai score de confiance).
    """

    def __init__(self,
                 yolo_seg:  YOLOSegDetector,
                 maskrcnn:  MaskRCNNDetector,
                 deeplab:   DeepLabDetector,
                 w_yolo:    float = 1.0,
                 w_maskrcnn: float = 1.0,
                 w_deeplab: float = 0.8,
                 nms_iou_thr: float = 0.5,
                 conf_thr:    float = 0.25):
        self.yolo_seg = yolo_seg
        self.maskrcnn = maskrcnn
        self.deeplab  = deeplab
        self.class_names = SEG_CLASS_NAMES
        self.nms_iou_thr = nms_iou_thr
        self.conf_thr    = conf_thr

        total = w_yolo + w_maskrcnn + w_deeplab + 1e-8
        self.w_yolo     = w_yolo    / total
        self.w_maskrcnn = w_maskrcnn / total
        self.w_deeplab  = w_deeplab / total

    def predict(self, img_path: str) -> dict:
        """Exécute les 3 modèles et retourne la prédiction fusionnée."""
        pa = self.yolo_seg.predict(img_path)
        pb = self.maskrcnn.predict(img_path)
        pc = self.deeplab.predict(img_path)
        return self._fuse(pa, pb, pc)

    def _fuse(self, pa: dict, pb: dict, pc: dict) -> dict:
        wa, wb, wc = self.w_yolo, self.w_maskrcnn, self.w_deeplab

        # Pool global
        all_boxes  = list(pa['boxes'])  + list(pb['boxes'])  + list(pc['boxes'])
        all_labels = list(pa['labels']) + list(pb['labels']) + list(pc['labels'])
        all_scores = (
            [float(s) * wa for s in pa['scores']] +
            [float(s) * wb for s in pb['scores']] +
            [float(s) * wc for s in pc['scores']]
        )
        all_masks = list(pa['masks']) + list(pb['masks']) + list(pc['masks'])

        if not all_boxes:
            return _empty_pred()

        all_boxes  = np.array(all_boxes,  dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32)
        all_scores = np.array(all_scores, dtype=np.float32)

        fused_boxes, fused_labels, fused_scores, fused_masks = [], [], [], []

        for cls_id in np.unique(all_labels):
            sel   = all_labels == cls_id
            c_b   = all_boxes[sel]
            c_s   = all_scores[sel]
            c_m   = [all_masks[i] for i, v in enumerate(sel) if v]

            order = np.argsort(-c_s)
            c_b = c_b[order]; c_s = c_s[order]
            c_m = [c_m[i] for i in order]
            suppressed = np.zeros(len(c_b), dtype=bool)

            for i in range(len(c_b)):
                if suppressed[i]:
                    continue
                cluster = [i]
                for j in range(i + 1, len(c_b)):
                    if suppressed[j]:
                        continue
                    iou_val = (
                        _mask_iou(c_m[i], c_m[j])
                        if (c_m[i] is not None and c_m[j] is not None)
                        else _box_iou(c_b[i], c_b[j])
                    )
                    if iou_val >= self.nms_iou_thr:
                        cluster.append(j)
                        suppressed[j] = True

                cb = c_b[cluster]; cs = c_s[cluster]
                tot_w = cs.sum() + 1e-8
                fused_boxes.append((cb * cs[:, None]).sum(0) / tot_w)
                fused_labels.append(int(cls_id))
                fused_scores.append(float(cs.max()))

                clust_masks = [c_m[k] for k in cluster if c_m[k] is not None]
                clust_ws    = [c_s[k] for k in range(len(cluster))
                               if c_m[cluster[k]] is not None]
                fused_masks.append(
                    _weighted_mask(clust_masks, clust_ws) if clust_masks else None)

        if not fused_boxes:
            return _empty_pred()

        fb = np.array(fused_boxes,  dtype=np.float32)
        fl = np.array(fused_labels, dtype=np.int32)
        fs = np.array(fused_scores, dtype=np.float32)
        keep = fs >= self.conf_thr
        return {
            'boxes':  fb[keep],
            'labels': fl[keep],
            'scores': fs[keep],
            'masks':  [fused_masks[i] for i, k in enumerate(keep) if k],
        }


# =============================================================================
# ANNOTATION
# =============================================================================

def annotate_image(img: Image.Image, pred: dict,
                   class_names: List[str], colors: dict,
                   mask_alpha: float = 0.40) -> Image.Image:
    """Dessine boîtes, masques et étiquettes sur l'image.

    Retourne une PIL Image annotée (RGB).
    """
    base  = img.convert('RGB')
    boxes  = pred.get('boxes',  [])
    labels = pred.get('labels', [])
    scores = pred.get('scores', [])
    masks  = pred.get('masks',  [])

    # ── Masques semi-transparents ────────────────────────────────────────────
    if len(boxes) > 0 and any(
        i < len(masks) and masks[i] is not None for i in range(len(boxes))
    ):
        overlay = base.copy()
        for i in range(len(boxes)):
            label_idx = int(labels[i])
            cls_name  = (class_names[label_idx - 1]
                         if 1 <= label_idx <= len(class_names) else f'cls{label_idx}')
            color = colors.get(cls_name, (128, 128, 128))
            if i < len(masks) and masks[i] is not None:
                m_arr     = masks[i].astype(bool)
                m_img     = Image.fromarray((m_arr * 255).astype(np.uint8), mode='L')
                col_layer = Image.new('RGB', base.size, color)
                overlay.paste(col_layer, mask=m_img)
        result = Image.blend(base, overlay, mask_alpha)
    else:
        result = base.copy()

    # ── Boîtes + étiquettes (net, au-dessus) ─────────────────────────────────
    draw = ImageDraw.Draw(result)
    font = _get_font(14)
    lw   = max(2, int(min(base.width, base.height) / 400))

    for i in range(len(boxes)):
        label_idx = int(labels[i])
        cls_name  = (class_names[label_idx - 1]
                     if 1 <= label_idx <= len(class_names) else f'cls{label_idx}')
        color = colors.get(cls_name, (128, 128, 128))
        score = float(scores[i])
        x1, y1, x2, y2 = (float(boxes[i][0]), float(boxes[i][1]),
                           float(boxes[i][2]), float(boxes[i][3]))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        txt = f"{cls_name} {score:.2f}"
        try:
            bb  = draw.textbbox((0, 0), txt, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            tw, th = len(txt) * 7, 14
        pad = 3
        tx1 = x1
        ty1 = max(0.0, y1 - th - 2 * pad)
        draw.rectangle([tx1, ty1, tx1 + tw + 2 * pad, ty1 + th + 2 * pad], fill=color)
        draw.text((tx1 + pad, ty1 + pad), txt, fill=(255, 255, 255), font=font)

    return result


# =============================================================================
# DESCRIPTION
# =============================================================================

def _detection_list(pred: dict, class_names: List[str]) -> list:
    dets = []
    boxes  = pred.get('boxes',  [])
    labels = pred.get('labels', [])
    scores = pred.get('scores', [])
    masks  = pred.get('masks',  [])
    for i in range(len(boxes)):
        idx      = int(labels[i])
        cls_name = (class_names[idx - 1] if 1 <= idx <= len(class_names)
                    else f'class_{idx}')
        box  = boxes[i]
        area = max(0.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))
        det  = {
            'id':         i + 1,
            'class':      cls_name,
            'confidence': round(float(scores[i]), 4),
            'bbox':       [round(float(v), 1) for v in box],
            'area_px':    round(area),
        }
        if i < len(masks) and masks[i] is not None:
            det['mask_area_px'] = int(masks[i].astype(bool).sum())
        dets.append(det)
    return dets


def _text_summary(mode: str, yolo_dets: list, ens_dets: list = None) -> str:
    """Génère une description textuelle en français."""
    lines = []
    if mode == 'snapshot':
        lines.append(f"Analyse Snapshot (YOLO uniquement).")
        if yolo_dets:
            from collections import Counter
            cnt = Counter(d['class'] for d in yolo_dets)
            parts = [f"{v} {k.replace('_', ' ')}" for k, v in cnt.items()]
            lines.append(f"YOLO a détecté : {', '.join(parts)}.")
        else:
            lines.append("YOLO n'a détecté aucun objet.")
    else:
        lines.append("Analyse Production (YOLO + Ensemble Soft+WA-NMS).")
        if yolo_dets:
            from collections import Counter
            cnt = Counter(d['class'] for d in yolo_dets)
            parts = [f"{v} {k.replace('_', ' ')}" for k, v in cnt.items()]
            lines.append(f"YOLO a détecté : {', '.join(parts)}.")
        else:
            lines.append("YOLO n'a détecté aucun objet.")
        if ens_dets:
            from collections import Counter
            cnt = Counter(d['class'] for d in ens_dets)
            parts = [f"{v} zone(s) de {k.replace('_', ' ')}" for k, v in cnt.items()]
            lines.append(
                f"L'ensemble (YOLO-seg + Mask R-CNN + DeepLabV3+) a identifié : {', '.join(parts)}.")
        elif ens_dets is not None:
            lines.append("L'ensemble n'a détecté aucune zone de toiture.")
    return " ".join(lines)


def make_description(image_name: str, mode: str,
                     yolo_pred: dict,
                     ensemble_pred: dict = None,
                     yolo_class_names: List[str] = None) -> dict:
    """Génère le dict de description d'une image."""
    _yolo_names = yolo_class_names or list(DET_COLORS.keys())
    yolo_dets = _detection_list(yolo_pred, _yolo_names)
    ens_dets  = (_detection_list(ensemble_pred, SEG_CLASS_NAMES)
                 if ensemble_pred is not None else None)

    desc = {
        'image':            image_name,
        'mode':             mode,
        'timestamp':        datetime.now().isoformat(),
        'text_description': _text_summary(mode, yolo_dets, ens_dets),
        'yolo': {
            'model':         'YOLODetector (class/yolo_classification)',
            'total':         len(yolo_dets),
            'classes_found': list({d['class'] for d in yolo_dets}),
            'detections':    yolo_dets,
        },
    }
    if ens_dets is not None:
        desc['ensemble'] = {
            'model':         'SoftWANMSEnsemble (YOLO-seg + Mask R-CNN + DeepLabV3+)',
            'total':         len(ens_dets),
            'classes_found': list({d['class'] for d in ens_dets}),
            'detections':    ens_dets,
        }
    return desc


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class Pipeline:
    """Pipeline de détection/segmentation de toitures.

    Routing par préfixe de nom de fichier :
      Snapshot*   → YOLODetector uniquement
                    → 1 image annotée  + 1 JSON
      Production* → YOLODetector, puis SoftWANMSEnsemble
                    → 2 images annotées (_yolo / _ensemble) + 1 JSON

    Args:
        yolo_detector : instance de YOLODetector (obligatoire)
        ensemble      : instance de SoftWANMSEnsemble (optionnel)
        output_dir    : dossier racine de sortie
    """

    def __init__(self,
                 yolo_oblique: YOLODetector,
                 yolo_nadir: YOLODetector,
                 ensemble: Optional[SoftWANMSEnsemble] = None,
                 output_dir: str = 'output'):
        self.yolo_oblique = yolo_oblique   # Snapshot (vue oblique)
        self.yolo_nadir   = yolo_nadir     # Production (vue nadir)
        self.ensemble = ensemble
        self.out_dir  = output_dir
        self._ann_dir  = os.path.join(output_dir, 'annotated')
        self._desc_dir = os.path.join(output_dir, 'descriptions')
        os.makedirs(self._ann_dir,  exist_ok=True)
        os.makedirs(self._desc_dir, exist_ok=True)

    # ── Points d'entrée publics ───────────────────────────────────────────────

    def process_directory(self, images_dir: str) -> List[dict]:
        """Traite toutes les images d'un dossier."""
        files = sorted([
            f for f in os.listdir(images_dir)
            if Path(f).suffix.lower() in IMG_EXTS
        ])
        paths = [os.path.join(images_dir, f) for f in files]
        return self.process_images(paths)

    def process_images(self, image_paths: List[str]) -> List[dict]:
        """Traite une liste de chemins d'images.

        Retourne la liste des descriptions (une par image traitée).
        """
        stats = {'snapshot': 0, 'production': 0, 'skip': 0, 'error': 0}
        results = []

        print(f"\n{'=' * 62}")
        print(f"  Pipeline Toitures - {len(image_paths)} image(s)")
        print(f"  Sortie : {self.out_dir}")
        if self.ensemble:
            print(f"  Ensemble : Soft+WA-NMS (YOLO-seg + Mask R-CNN + DeepLab)")
        else:
            print(f"  Ensemble : desactive (YOLO uniquement)")
        print(f"{'=' * 62}\n")

        for img_path in image_paths:
            img_name = Path(img_path).name
            stem     = Path(img_path).stem

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  [ERREUR]     {img_name}: {e}")
                stats['error'] += 1
                continue

            if stem.startswith(('Snapshot', 'snapshot')):
                desc = self._process_snapshot(img_path, img_name, img)
                stats['snapshot'] += 1
                results.append(desc)

            elif stem.startswith(('Production', 'production')):
                desc = self._process_production(img_path, img_name, img)
                stats['production'] += 1
                results.append(desc)

            else:
                print(f"  [SKIP]       {img_name}  (prefixe inconnu - attendu : Snapshot | Production)")
                stats['skip'] += 1

        # ── Rapport global ────────────────────────────────────────────────────
        summary = {
            'timestamp':     datetime.now().isoformat(),
            'output_dir':    self.out_dir,
            'total_input':   len(image_paths),
            'stats':         stats,
            'results':       results,
        }
        summary_path = os.path.join(self.out_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # ── Rapport final agrégé ──────────────────────────────────────────────
        rapport_path = os.path.join(self.out_dir, 'rapport_final.json')
        try:
            from pipeline.aggregator import aggregate
            _res = os.getenv('RESOLUTION_M_PX')
            aggregate(self._desc_dir, output_path=rapport_path,
                      resolution_m_px=float(_res) if _res else None)
        except Exception as _exc:
            print(f"  [!] Rapport final non genere : {_exc}")

        print(f"\n{'=' * 62}")
        print(f"  Snapshot        : {stats['snapshot']}")
        print(f"  Production      : {stats['production']}")
        print(f"  Ignorees        : {stats['skip']}")
        print(f"  Erreurs         : {stats['error']}")
        print(f"  Summary         : {summary_path}")
        print(f"  Rapport final   : {rapport_path}")
        print(f"{'=' * 62}\n")
        return results

    # ── Traitement Snapshot ───────────────────────────────────────────────────

    def _process_snapshot(self, img_path: str, img_name: str,
                          img: Image.Image) -> dict:
        print(f"  [Snapshot]   {img_name}")

        pred   = self.yolo_oblique.predict(img_path)
        n      = len(pred.get('boxes', []))
        print(f"               YOLO-oblique -> {n} detection(s)")

        cls_names = self.yolo_oblique.class_names
        ann    = annotate_image(img, pred, cls_names, DET_COLORS)
        ann_p  = os.path.join(self._ann_dir, img_name)
        ann.save(ann_p, quality=95)

        desc   = make_description(img_name, 'snapshot', pred,
                                  yolo_class_names=cls_names)
        desc_p = os.path.join(self._desc_dir, Path(img_name).stem + '.json')
        with open(desc_p, 'w', encoding='utf-8') as f:
            json.dump(desc, f, indent=2, ensure_ascii=False)

        print(f"               -> {ann_p}")
        print(f"               -> {desc_p}")
        return desc

    # ── Traitement Production ─────────────────────────────────────────────────

    def _process_production(self, img_path: str, img_name: str,
                             img: Image.Image) -> dict:
        print(f"  [Production] {img_name}")
        stem = Path(img_name).stem
        ext  = Path(img_name).suffix

        # Étape 1 : YOLO nadir
        yolo_pred  = self.yolo_nadir.predict(img_path)
        nadir_names = self.yolo_nadir.class_names
        n_yolo    = len(yolo_pred.get('boxes', []))
        print(f"               YOLO-nadir -> {n_yolo} detection(s)")

        yolo_ann  = annotate_image(img, yolo_pred, nadir_names, DET_COLORS)
        yolo_path = os.path.join(self._ann_dir, f"{stem}_yolo{ext}")
        yolo_ann.save(yolo_path, quality=95)
        print(f"               -> {yolo_path}")

        # Étape 2 : Ensemble Soft+WA-NMS
        ens_pred = None
        if self.ensemble is not None:
            ens_pred = self.ensemble.predict(img_path)
            n_ens    = len(ens_pred.get('boxes', []))
            print(f"               Ensemble (Soft+WA-NMS) -> {n_ens} instance(s)")

            ens_ann  = annotate_image(img, ens_pred, SEG_CLASS_NAMES, SEG_COLORS)
            ens_path = os.path.join(self._ann_dir, f"{stem}_ensemble{ext}")
            ens_ann.save(ens_path, quality=95)
            print(f"               -> {ens_path}")
        else:
            print(f"               Ensemble desactive - seul YOLO utilise")

        desc   = make_description(img_name, 'production', yolo_pred, ens_pred,
                                  yolo_class_names=nadir_names)
        desc_p = os.path.join(self._desc_dir, stem + '.json')
        with open(desc_p, 'w', encoding='utf-8') as f:
            json.dump(desc, f, indent=2, ensure_ascii=False)
        print(f"               -> {desc_p}")
        return desc
