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
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn
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

# Classes de détection — espace de labels global (1-indexé, commun YOLO + FRCNN)
DET_CLASS_NAMES = ['panneau_solaire', 'batiment_peint', 'batiment_non_enduit',
                   'batiment_enduit', 'menuiserie_metallique']


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


def iou_calc(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


def fuse_soft_wanms(pred_a: dict, pred_b: dict,
                    w_a: float = 0.55, w_b: float = 0.45,
                    nms_iou_thr: float = 0.5,
                    conf_thr: float = 0.25) -> dict:
    all_boxes  = list(pred_a['boxes'])  + list(pred_b['boxes'])
    all_labels = list(pred_a['labels']) + list(pred_b['labels'])
    all_scores = ([float(s) * w_a for s in pred_a['scores']] +
                  [float(s) * w_b for s in pred_b['scores']])

    if not all_boxes:
        return {'boxes': np.zeros((0, 4), dtype=np.float32),
                'labels': np.zeros(0, dtype=np.int32),
                'scores': np.zeros(0, dtype=np.float32)}

    all_boxes  = np.array(all_boxes,  dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)
    all_scores = np.array(all_scores, dtype=np.float32)

    fused_boxes, fused_labels, fused_scores = [], [], []

    for cls_id in np.unique(all_labels):
        mask = all_labels == cls_id
        c_b  = all_boxes[mask]
        c_s  = all_scores[mask]

        order = np.argsort(-c_s)
        c_b   = c_b[order]
        c_s   = c_s[order]

        suppressed = np.zeros(len(c_b), dtype=bool)
        for i in range(len(c_b)):
            if suppressed[i]:
                continue
            cluster = [i]
            for j in range(i + 1, len(c_b)):
                if not suppressed[j] and iou_calc(c_b[i], c_b[j]) >= nms_iou_thr:
                    cluster.append(j)
                    suppressed[j] = True
            cb    = c_b[cluster]
            cs    = c_s[cluster]
            tot_w = cs.sum() + 1e-8
            fused_boxes.append((cb * cs[:, None]).sum(0) / tot_w)
            fused_labels.append(int(cls_id))
            fused_scores.append(float(cs.max()))

    if not fused_boxes:
        return {'boxes': np.zeros((0, 4), dtype=np.float32),
                'labels': np.zeros(0, dtype=np.int32),
                'scores': np.zeros(0, dtype=np.float32)}

    fused_boxes  = np.array(fused_boxes,  dtype=np.float32)
    fused_labels = np.array(fused_labels, dtype=np.int32)
    fused_scores = np.array(fused_scores, dtype=np.float32)
    keep = fused_scores >= conf_thr
    return {'boxes':  fused_boxes[keep],
            'labels': fused_labels[keep],
            'scores': fused_scores[keep]}


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


def _letterbox(img: Image.Image, target_w: int, target_h: int):
    """Redimensionne sans déformation puis remplit de blanc (letterbox).

    Returns:
        padded  : PIL Image (target_w × target_h)
        pad_l   : pixels ajoutés à gauche
        pad_t   : pixels ajoutés en haut
        scale   : facteur appliqué à l'image originale
    """
    ow, oh = img.size
    scale  = min(target_w / ow, target_h / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    pad_l = (target_w - nw) // 2
    pad_t = (target_h - nh) // 2
    canvas = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    canvas.paste(resized, (pad_l, pad_t))
    return canvas, pad_l, pad_t, scale


def _unpad_pred(pred: dict, pad_l: int, pad_t: int, scale: float) -> dict:
    """Transforme les boîtes du repère letterboxé vers le repère original."""
    if len(pred['boxes']) == 0:
        return pred
    b = pred['boxes'].copy().astype(np.float32)
    b[:, 0] = (b[:, 0] - pad_l) / scale
    b[:, 1] = (b[:, 1] - pad_t) / scale
    b[:, 2] = (b[:, 2] - pad_l) / scale
    b[:, 3] = (b[:, 3] - pad_t) / scale
    b = np.clip(b, 0.0, None)
    return {**pred, 'boxes': b}


# =============================================================================
# MODÈLE 1 — YOLODetector (détection, class/yolo_classification)
# =============================================================================

class YOLODetector:
    """YOLO détection (class/yolo_classification).

    Le label global est déterminé par le nom de classe (model.names) indexé dans
    DET_CLASS_NAMES — indépendant de l'ordre d'entraînement.
    """

    def __init__(self, model_path: str, threshold: float = 0.25,
                 class_thresholds: dict = None, label_offset: int = 0):
        # label_offset conservé pour compatibilité d'appel mais non utilisé
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics requis : pip install ultralytics")
        self.model = _UltralyticsYOLO(model_path)
        self.threshold = threshold
        self.class_thresholds = class_thresholds or {}
        self.class_names = list(self.model.names.values())
        print(f"   [OK] YOLODetector       : {model_path}  classes={self.class_names}")

    def _thr(self, cls_name: str) -> float:
        return self.class_thresholds.get(cls_name, self.threshold)

    def predict(self, img_path: str, img: Image.Image = None) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks': []}.
        Le label global est l'index 1-based dans DET_CLASS_NAMES.
        """
        src = img if img is not None else img_path
        result = self.model(src, verbose=False)[0]
        boxes, labels, scores = [], [], []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                s = float(box.conf)
                cls_name = self.class_names[int(box.cls)]
                if cls_name not in DET_CLASS_NAMES:
                    continue
                if s < self._thr(cls_name):
                    continue
                global_label = DET_CLASS_NAMES.index(cls_name) + 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(global_label)
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

    def __init__(self, model_path: str, threshold: float = 0.25,
                 class_thresholds: dict = None):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics requis : pip install ultralytics")
        self.model = _UltralyticsYOLO(model_path)
        self.threshold = threshold
        self.class_thresholds = class_thresholds or {}
        self.class_names = SEG_CLASS_NAMES
        print(f"   [OK] YOLOSegDetector    : {model_path}")

    def _thr(self, cls_name: str) -> float:
        return self.class_thresholds.get(cls_name, self.threshold)

    def predict(self, img_path: str, img: Image.Image = None) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks'}.
        img : image pré-traitée (ex. letterboxée) fournie par le Pipeline.
        """
        src = img.convert('RGB') if img is not None else img_path
        src_pil = src if isinstance(src, Image.Image) else Image.open(src).convert('RGB')
        orig_w, orig_h = src_pil.size
        _min_conf = min(self.class_thresholds.values()) if self.class_thresholds else self.threshold
        result = self.model.predict(src, conf=_min_conf, verbose=False)[0]
        if result.masks is None or len(result.boxes) == 0:
            return _empty_pred()
        all_boxes  = result.boxes.xyxy.cpu().numpy()
        all_labels = result.boxes.cls.cpu().numpy().astype(int) + 1
        all_scores = result.boxes.conf.cpu().numpy()
        all_masks  = []
        for raw in result.masks.data.cpu().numpy():
            m = Image.fromarray(raw.astype(np.uint8))
            m = m.resize((orig_w, orig_h), Image.NEAREST)
            all_masks.append(np.array(m) > 0)
        boxes, labels, scores, masks = [], [], [], []
        for i in range(len(all_boxes)):
            lbl = int(all_labels[i])
            cls_name = SEG_CLASS_NAMES[lbl - 1] if 1 <= lbl <= len(SEG_CLASS_NAMES) else None
            if all_scores[i] < self._thr(cls_name or ''):
                continue
            boxes.append(all_boxes[i])
            labels.append(lbl)
            scores.append(all_scores[i])
            masks.append(all_masks[i])
        if not boxes:
            return _empty_pred()
        return {
            'boxes':  np.array(boxes,  dtype=np.float32),
            'labels': np.array(labels, dtype=np.int32),
            'scores': np.array(scores, dtype=np.float32),
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
                 threshold: float = 0.25, device: str = None,
                 class_thresholds: dict = None):
        if not TORCH_AVAILABLE:
            raise ImportError("torch requis : pip install torch torchvision")
        self.device    = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold = threshold
        self.class_thresholds = class_thresholds or {}
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

    def _thr(self, cls_name: str) -> float:
        return self.class_thresholds.get(cls_name, self.threshold)

    @_no_grad
    def predict(self, img_path: str, img: Image.Image = None) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks'}.
        img : image pré-traitée (ex. letterboxée) fournie par le Pipeline.
        """
        img    = (img or Image.open(img_path)).convert('RGB')
        tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
        out    = self.model(tensor)[0]
        sc_np  = out['scores'].cpu().numpy()
        lb_np  = out['labels'].cpu().numpy()
        keep   = torch.tensor([
            sc_np[i] >= self._thr(
                SEG_CLASS_NAMES[lb_np[i] - 1] if 1 <= lb_np[i] <= len(SEG_CLASS_NAMES) else ''
            )
            for i in range(len(sc_np))
        ])
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
    def predict(self, img_path: str, img: Image.Image = None) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks'}.
        img : image pré-traitée (ex. letterboxée) fournie par le Pipeline.
        """
        img = (img or Image.open(img_path)).convert('RGB')
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
                 conf_thr:    float = 0.25,
                 class_thresholds: dict = None):
        self.yolo_seg = yolo_seg
        self.maskrcnn = maskrcnn
        self.deeplab  = deeplab
        self.class_names = SEG_CLASS_NAMES
        self.nms_iou_thr = nms_iou_thr
        self.conf_thr    = conf_thr
        self.class_thresholds = class_thresholds or {}

        total = w_yolo + w_maskrcnn + w_deeplab + 1e-8
        self.w_yolo     = w_yolo    / total
        self.w_maskrcnn = w_maskrcnn / total
        self.w_deeplab  = w_deeplab / total

    def _thr(self, cls_name: str) -> float:
        return self.class_thresholds.get(cls_name, self.conf_thr)

    def predict(self, img_path: str, img: Image.Image = None) -> dict:
        """Exécute les 3 modèles et retourne la prédiction fusionnée.
        img : image pré-traitée (ex. letterboxée) transmise aux 3 sous-modèles
              pour que tous les masques soient dans le même espace de coordonnées.
        """
        pa = self.yolo_seg.predict(img_path, img=img)
        pb = self.maskrcnn.predict(img_path, img=img)
        pc = self.deeplab.predict(img_path, img=img)
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
        keep = np.array([
            fs[i] >= self._thr(
                SEG_CLASS_NAMES[fl[i] - 1] if 1 <= fl[i] <= len(SEG_CLASS_NAMES) else ''
            )
            for i in range(len(fs))
        ], dtype=bool)
        return {
            'boxes':  fb[keep],
            'labels': fl[keep],
            'scores': fs[keep],
            'masks':  [fused_masks[i] for i, k in enumerate(keep) if k],
        }


# =============================================================================
# MODÈLE 5 — FasterRCNNDetector (class/fasterrcnn_classification)
# =============================================================================

class FasterRCNNDetector:
    """Faster R-CNN (class/fasterrcnn_classification).

    label_offset=0 pour nadir  : FRCNN label 1 → global 1 (panneau_solaire)
    label_offset=1 pour oblique: FRCNN label 1 → global 2 (batiment_peint), …
    """

    def __init__(self, model_path: str, threshold: float = 0.25,
                 label_offset: int = 0, device: str = None,
                 class_thresholds: dict = None):
        # label_offset conservé pour compatibilité d'appel mais non utilisé
        if not TORCH_AVAILABLE:
            raise ImportError("torch requis : pip install torch torchvision")
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold = threshold
        self.class_thresholds = class_thresholds or {}

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        nc              = ckpt.get('num_classes', 2)
        sz              = ckpt.get('image_size', 640)
        self.classes    = ckpt.get('classes', [])   # ex. ['__background__', 'panneau_solaire', …]
        m    = fasterrcnn_resnet50_fpn(weights=None)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, nc)
        m.load_state_dict(ckpt['model_state_dict'])
        m.to(self.device)
        m.eval()
        self.model      = m
        self.image_size = sz
        print(f"   [OK] FasterRCNNDetector : {model_path}  classes={self.classes}  device={self.device})")

    def _thr(self, cls_name: str) -> float:
        return self.class_thresholds.get(cls_name, self.threshold)

    @_no_grad
    def predict(self, img_path: str, img: Image.Image = None) -> dict:
        """Retourne {'boxes', 'labels', 'scores', 'masks': []}.
        Le label global est déterminé par le nom de classe lu dans le checkpoint.
        """
        img = (img or Image.open(img_path)).convert('RGB')
        ow, oh = img.size
        t = (TF.to_tensor(img.resize((self.image_size, self.image_size)))
             .unsqueeze(0).to(self.device))
        out = self.model(t)[0]
        if len(out['boxes']) == 0:
            return _empty_pred()

        b = out['boxes'].cpu().numpy().copy()
        l = out['labels'].cpu().numpy()
        s = out['scores'].cpu().numpy()

        sx, sy = ow / self.image_size, oh / self.image_size
        b[:, 0] *= sx;  b[:, 2] *= sx
        b[:, 1] *= sy;  b[:, 3] *= sy

        boxes, labels, scores = [], [], []
        for i in range(len(b)):
            raw = int(l[i])
            cls_name = self.classes[raw] if raw < len(self.classes) else ''
            if not cls_name or cls_name == '__background__':
                continue
            if cls_name not in DET_CLASS_NAMES:
                continue
            global_label = DET_CLASS_NAMES.index(cls_name) + 1
            if s[i] < self._thr(cls_name):
                continue
            boxes.append(b[i])
            labels.append(global_label)
            scores.append(float(s[i]))

        if not boxes:
            return _empty_pred()
        return {
            'boxes':  np.array(boxes,  dtype=np.float32),
            'labels': np.array(labels, dtype=np.int32),
            'scores': np.array(scores, dtype=np.float32),
            'masks':  [],
        }


# =============================================================================
# FUSION DÉTECTION — DetFuser (YOLO + Faster R-CNN, Soft+WA-NMS)
# =============================================================================

class DetFuser:
    """Fusion YOLO + Faster R-CNN via Soft+WA-NMS.

    w_yolo=0.55 (YOLO converge plus vite), w_frcnn=0.45.
    Stocke les prédictions individuelles dans _last_yolo / _last_frcnn
    pour la visualisation comparative (save_detection_comparison).
    """

    def __init__(self, yolo: YOLODetector, frcnn: FasterRCNNDetector,
                 w_yolo: float = 0.55, w_frcnn: float = 0.45,
                 nms_iou_thr: float = 0.5, conf_thr: float = 0.25,
                 class_thresholds: dict = None):
        self.yolo        = yolo
        self.frcnn       = frcnn
        self.w_yolo      = w_yolo
        self.w_frcnn     = w_frcnn
        self.nms_iou_thr = nms_iou_thr
        self.conf_thr    = conf_thr
        self.class_thresholds = class_thresholds or {}
        self.class_names = DET_CLASS_NAMES
        self._last_yolo:   dict  = _empty_pred()
        self._last_frcnn:  dict  = _empty_pred()
        self._last_t_yolo:  float = 0.0
        self._last_t_frcnn: float = 0.0

    def _thr(self, global_label: int) -> float:
        cls_name = (DET_CLASS_NAMES[global_label - 1]
                    if 1 <= global_label <= len(DET_CLASS_NAMES) else '')
        return self.class_thresholds.get(cls_name, self.conf_thr)

    def predict(self, img_path: str, img: Image.Image = None) -> dict:
        """img : image pré-traitée (ex. letterboxée) fournie par le Pipeline.
        Si None, les modèles chargent depuis img_path directement.
        """
        t0 = time.perf_counter()
        raw_yolo = self.yolo.predict(img_path, img=img)
        self._last_t_yolo = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        raw_frcnn = self.frcnn.predict(img_path, img=img)
        self._last_t_frcnn = (time.perf_counter() - t0) * 1000.0

        self._last_yolo  = raw_yolo
        self._last_frcnn = raw_frcnn

        return self._fuse(raw_yolo, raw_frcnn)

    def _fuse(self, pa: dict, pb: dict) -> dict:
        # panneau_solaire (label=1) : YOLO seul, pas de fusion avec FRCNN
        _SOL = DET_CLASS_NAMES.index('panneau_solaire') + 1  # = 1

        def _mask_label(pred, lbl, keep_match: bool):
            if len(pred['boxes']) == 0:
                return _empty_pred()
            m = (pred['labels'] == lbl) if keep_match else (pred['labels'] != lbl)
            return {
                'boxes':  pred['boxes'][m],
                'labels': pred['labels'][m],
                'scores': pred['scores'][m],
                'masks':  [],
            }

        # YOLO panneau_solaire → gardé tel quel (seuil par classe)
        pa_sol  = _mask_label(pa, _SOL, keep_match=True)
        sol_thr = self._thr(_SOL)
        if len(pa_sol['boxes']) > 0:
            keep_sol = pa_sol['scores'] >= sol_thr
            pa_sol = {k: (v[keep_sol] if isinstance(v, np.ndarray) else v)
                      for k, v in pa_sol.items()}

        # Reste : fusion YOLO + FRCNN (panneau_solaire exclu des deux)
        pa_rest = _mask_label(pa, _SOL, keep_match=False)
        pb_rest = _mask_label(pb, _SOL, keep_match=False)

        merged = fuse_soft_wanms(
            pa_rest, pb_rest,
            w_a=self.w_yolo, w_b=self.w_frcnn,
            nms_iou_thr=self.nms_iou_thr,
            conf_thr=self.conf_thr,
        )
        fb, fl, fs = merged['boxes'], merged['labels'], merged['scores']
        keep = np.array([fs[i] >= self._thr(int(fl[i])) for i in range(len(fs))], dtype=bool)
        fused_rest = {'boxes': fb[keep], 'labels': fl[keep], 'scores': fs[keep], 'masks': []}

        # Concaténer panneau_solaire (YOLO) + reste fusionné
        if len(pa_sol['boxes']) == 0:
            return fused_rest
        if len(fused_rest['boxes']) == 0:
            return pa_sol
        return {
            'boxes':  np.concatenate([pa_sol['boxes'],  fused_rest['boxes']],  axis=0),
            'labels': np.concatenate([pa_sol['labels'], fused_rest['labels']]),
            'scores': np.concatenate([pa_sol['scores'], fused_rest['scores']]),
            'masks':  [],
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


def save_detection_comparison(img_path: str, det_fuser: 'DetFuser',
                               fused_pred: dict, out_path: str,
                               display: bool = False,
                               img: Image.Image = None) -> None:
    """Sauvegarde une image de comparaison 3 panneaux : YOLO | FRCNN | Fusionné.
    img : image pré-traitée (ex. letterboxée) à afficher ; si None, ouvre img_path.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [!] matplotlib manquant — comparaison non generee (pip install matplotlib)")
        return

    img_arr = np.array((img if img is not None else Image.open(img_path)).convert('RGB'))
    pred_yolo  = getattr(det_fuser, '_last_yolo',  _empty_pred())
    pred_frcnn = getattr(det_fuser, '_last_frcnn', _empty_pred())
    t_yolo     = getattr(det_fuser, '_last_t_yolo',  0.0)
    t_frcnn    = getattr(det_fuser, '_last_t_frcnn', 0.0)

    panels = [
        (pred_yolo,  f"YOLO  ({len(pred_yolo['boxes'])} déts,  {t_yolo:.0f} ms)"),
        (pred_frcnn, f"Faster R-CNN  ({len(pred_frcnn['boxes'])} déts,  {t_frcnn:.0f} ms)"),
        (fused_pred, f"Fusionné Soft+WA-NMS  ({len(fused_pred['boxes'])} déts)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (pred, title) in zip(axes, panels):
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis('off')
        for i in range(len(pred['boxes'])):
            lbl   = int(pred['labels'][i])
            name  = (DET_CLASS_NAMES[lbl - 1]
                     if 1 <= lbl <= len(DET_CLASS_NAMES) else f'cls{lbl}')
            score = float(pred['scores'][i])
            rgb   = DET_COLORS.get(name, (128, 128, 128))
            color = tuple(c / 255 for c in rgb)
            x1, y1, x2, y2 = [float(v) for v in pred['boxes'][i]]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                boxstyle='square,pad=0', lw=1.5,
                edgecolor=color, facecolor='none'))
            ax.text(x1, max(0.0, y1 - 3), f"{name} {score:.2f}",
                    fontsize=6, color='white',
                    bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor='none'))

    handles = [mpatches.Patch(color=tuple(c / 255 for c in rgb), label=name)
               for name, rgb in DET_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    if display:
        plt.show()
    plt.close(fig)


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
        lines.append("Analyse Production (YOLO + Mask R-CNN).")
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
            lines.append(f"Mask R-CNN a identifié : {', '.join(parts)}.")
        elif ens_dets is not None:
            lines.append("Mask R-CNN n'a détecté aucune zone de toiture.")
    return " ".join(lines)


def make_description(image_name: str, mode: str,
                     det_pred: dict,
                     ensemble_pred: dict = None,
                     yolo_class_names: List[str] = None) -> dict:
    """Génère le dict de description d'une image."""
    _det_names = yolo_class_names or DET_CLASS_NAMES
    det_dets  = _detection_list(det_pred, _det_names)
    ens_dets  = (_detection_list(ensemble_pred, SEG_CLASS_NAMES)
                 if ensemble_pred is not None else None)

    desc = {
        'image':            image_name,
        'mode':             mode,
        'timestamp':        datetime.now().isoformat(),
        'text_description': _text_summary(mode, det_dets, ens_dets),
        'detection': {
            'model':         'DetFuser (YOLO + Faster R-CNN, Soft+WA-NMS)',
            'total':         len(det_dets),
            'classes_found': list({d['class'] for d in det_dets}),
            'detections':    det_dets,
        },
    }
    if ens_dets is not None:
        # Union des masques par classe — évite de compter deux fois les pixels
        # qui se chevauchent entre plusieurs instances de la même classe.
        union_areas: dict = {}
        if ensemble_pred is not None:
            ens_masks  = ensemble_pred.get('masks', [])
            ens_labels = ensemble_pred.get('labels', [])
            for i, lbl in enumerate(ens_labels):
                if i >= len(ens_masks) or ens_masks[i] is None:
                    continue
                cls_name = (SEG_CLASS_NAMES[int(lbl) - 1]
                            if 1 <= int(lbl) <= len(SEG_CLASS_NAMES) else None)
                if cls_name is None:
                    continue
                m = np.asarray(ens_masks[i], dtype=bool)
                if cls_name in union_areas:
                    union_areas[cls_name] |= m
                else:
                    union_areas[cls_name] = m.copy()

        desc['segmentation'] = {
            'model':                   'MaskRCNNDetector (seg/mark_r_cnn)',
            'total':                   len(ens_dets),
            'classes_found':           list({d['class'] for d in ens_dets}),
            'union_area_px_by_class':  {cls: int(m.sum()) for cls, m in union_areas.items()},
            'detections':              ens_dets,
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
      Production* → YOLODetector, puis MaskRCNNDetector (segmentation)
                    → 2 images annotées (_yolo / _seg) + 1 JSON

    Args:
        yolo_oblique : YOLODetector pour les vues obliques (Snapshot)
        yolo_nadir   : YOLODetector pour les vues nadir (Production)
        ensemble     : MaskRCNNDetector pour la segmentation (optionnel)
        output_dir   : dossier racine de sortie
    """

    def __init__(self,
                 det_oblique,
                 det_nadir,
                 ensemble=None,
                 output_dir: str = 'output',
                 display: bool = False):
        self.det_oblique = det_oblique   # Snapshot (vue oblique)
        self.det_nadir   = det_nadir     # Production (vue nadir)
        self.ensemble = ensemble
        self.display  = display
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
        print(f"  Detection    : YOLO + Faster R-CNN (Soft+WA-NMS)")
        if self.ensemble:
            print(f"  Segmentation : Mask R-CNN")
        else:
            print(f"  Segmentation : desactivee")
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
        stem = Path(img_name).stem

        fused   = self.det_oblique.predict(img_path)
        _ep     = _empty_pred()
        n_yolo  = len(getattr(self.det_oblique, '_last_yolo',  _ep).get('boxes', []))
        n_frcnn = len(getattr(self.det_oblique, '_last_frcnn', _ep).get('boxes', []))
        n_fused = len(fused.get('boxes', []))
        print(f"               YOLO -> {n_yolo}  FRCNN -> {n_frcnn}  Fusionné -> {n_fused}")

        ann   = annotate_image(img, fused, DET_CLASS_NAMES, DET_COLORS)
        ann_p = os.path.join(self._ann_dir, img_name)
        ann.save(ann_p, quality=95)

        if isinstance(self.det_oblique, DetFuser):
            comp_p = os.path.join(self._ann_dir, f"{stem}_comparison.png")
            save_detection_comparison(img_path, self.det_oblique, fused,
                                      comp_p, display=self.display)
            print(f"               -> {comp_p}")

        desc   = make_description(img_name, 'snapshot', fused,
                                  yolo_class_names=DET_CLASS_NAMES)
        desc_p = os.path.join(self._desc_dir, stem + '.json')
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

        # Letterbox → 1738×1333 (sans déformation, bords blancs)
        # L'image letterboxée est utilisée pour tous les modèles et l'affichage final.
        lb_img, _pad_l, _pad_t, _scale = _letterbox(img, 1738, 1333)

        # Étape 1 : Détection fusionnée (YOLO nadir + Faster R-CNN nadir)
        det_pred = self.det_nadir.predict(img_path, img=lb_img)
        _ep      = _empty_pred()
        n_yolo   = len(getattr(self.det_nadir, '_last_yolo',  _ep).get('boxes', []))
        n_frcnn  = len(getattr(self.det_nadir, '_last_frcnn', _ep).get('boxes', []))
        n_fused  = len(det_pred.get('boxes', []))
        print(f"               YOLO -> {n_yolo}  FRCNN -> {n_frcnn}  Fusionné -> {n_fused}")

        det_ann  = annotate_image(lb_img, det_pred, DET_CLASS_NAMES, DET_COLORS)
        det_path = os.path.join(self._ann_dir, f"{stem}_det{ext}")
        det_ann.save(det_path, quality=95)
        print(f"               -> {det_path}")

        if isinstance(self.det_nadir, DetFuser):
            comp_p = os.path.join(self._ann_dir, f"{stem}_comparison.png")
            save_detection_comparison(img_path, self.det_nadir, det_pred,
                                      comp_p, display=self.display, img=lb_img)
            print(f"               -> {comp_p}")

        # Étape 2 : Segmentation Mask R-CNN (image originale, pas de letterbox)
        ens_pred = None
        if self.ensemble is not None:
            ens_pred = self.ensemble.predict(img_path)
            n_ens    = len(ens_pred.get('boxes', []))
            print(f"               Mask R-CNN -> {n_ens} instance(s)")

            ens_ann  = annotate_image(img, ens_pred, SEG_CLASS_NAMES, SEG_COLORS)
            ens_path = os.path.join(self._ann_dir, f"{stem}_seg{ext}")
            ens_ann.save(ens_path, quality=95)
            print(f"               -> {ens_path}")
        else:
            print(f"               Segmentation desactivee")

        desc   = make_description(img_name, 'production', det_pred, ens_pred,
                                  yolo_class_names=DET_CLASS_NAMES)
        desc_p = os.path.join(self._desc_dir, stem + '.json')
        with open(desc_p, 'w', encoding='utf-8') as f:
            json.dump(desc, f, indent=2, ensure_ascii=False)
        print(f"               -> {desc_p}")
        return desc
