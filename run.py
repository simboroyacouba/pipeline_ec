"""
pipeline/run.py — Point d'entrée CLI du pipeline toitures

Usage :
  python pipeline/run.py --images /chemin/vers/images --output ./output

  python pipeline/run.py \\
      --images /chemin/vers/images \\
      --yolo-model        /chemin/yolo_class/best.pt \\
      --yolo-seg-model    /chemin/yolo_seg/best.pt \\
      --maskrcnn-model    /chemin/maskrcnn/best_model.pth \\
      --deeplab-model     /chemin/deeplab/best_model.pth \\
      --output            ./output

Variables d'environnement (.env) :
  YOLO_DET_MODEL       YOLO détection  (class/yolo_classification)
  YOLO_SEG_MODEL       YOLO seg        (seg/yolo ou seg/yolo_26)
  MASKRCNN_MODEL       Mask R-CNN      (seg/mark_r_cnn)
  DEEPLAB_MODEL        DeepLabV3+      (seg/DeepLabV3)
  DEEPLAB_BACKBONE     resnet50 | resnet101  (défaut resnet50)
  SCORE_THRESHOLD      seuil confiance       (défaut 0.25)
  OUTPUT_DIR           dossier de sortie     (défaut ./output)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Permettre l'import depuis le dossier parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    _base = Path(__file__).resolve().parent.parent
    # Charger les .env des sous-modules en premier (priorité basse)
    for _sub in ['seg/mark_r_cnn', 'seg/yolo_26', 'seg/yolo', 'seg/DeepLabV3',
                 'class/yolo_classification']:
        _env = _base / _sub / '.env'
        if _env.exists():
            load_dotenv(_env)
    load_dotenv()  # .env du répertoire courant
    # Charger pipeline/.env en dernier → priorité maximale
    _pipeline_env = Path(__file__).resolve().parent / '.env'
    if _pipeline_env.exists():
        load_dotenv(_pipeline_env, override=True)
except ImportError:
    pass

from pipeline.pipeline import (
    YOLODetector, FasterRCNNDetector, DetFuser,
    MaskRCNNDetector,
    Pipeline,
    YOLO_AVAILABLE, TORCH_AVAILABLE,
)


# =============================================================================
# SEUILS PAR CLASSE
# =============================================================================

_DET_CLASSES = ['panneau_solaire', 'batiment_peint', 'batiment_non_enduit',
                'batiment_enduit', 'menuiserie_metallique']
_SEG_CLASSES = ['toiture_tole_ondulee', 'toiture_tole_bac', 'toiture_dalle']

_YOLO_CONF_VARS = {c: f'YOLO_CONF_{c.upper()}'  for c in _DET_CLASSES}
_FRCNN_CONF_VARS = {c: f'FRCNN_CONF_{c.upper()}' for c in _DET_CLASSES}
_FUSION_CONF_VARS = {c: f'CONF_{c.upper()}'      for c in _DET_CLASSES}
_SEG_CONF_VARS   = {c: f'CONF_{c.upper()}'       for c in _SEG_CLASSES}


def _load_thresholds(var_map: dict, default: float) -> dict:
    return {cls: float(os.getenv(var, default)) for cls, var in var_map.items()}


def _load_yolo_thresholds(default: float) -> dict:
    return _load_thresholds(_YOLO_CONF_VARS, default)

def _load_frcnn_thresholds(default: float) -> dict:
    return _load_thresholds(_FRCNN_CONF_VARS, default)

def _load_fusion_thresholds(default: float) -> dict:
    return {**_load_thresholds(_FUSION_CONF_VARS, default),
            **_load_thresholds(_SEG_CONF_VARS,    default)}


# =============================================================================
# AUTO-DÉTECTION DES MODÈLES
# =============================================================================

_BASE = Path(__file__).resolve().parent.parent


def _find(*paths) -> Optional[str]:
    """Retourne le premier chemin existant parmi ceux fournis, ou None."""
    for p in paths:
        if p and os.path.exists(str(p)):
            return str(p)
    return None


def _auto_yolo_oblique() -> Optional[str]:
    base = _BASE / 'class' / 'yolo_classification'
    return _find(
        os.getenv('YOLO_DET_OBLIQUE_MODEL'),
        os.getenv('YOLO_DET_MODEL'),
        base / 'runs/detect/oblique/train/weights/best.pt',
        base / 'runs/detect/train/weights/best.pt',
        base / 'output/best.pt',
    )


def _auto_yolo_nadir() -> Optional[str]:
    base = _BASE / 'class' / 'yolo_classification'
    return _find(
        os.getenv('YOLO_DET_NADIR_MODEL'),
        os.getenv('YOLO_DET_MODEL'),
        base / 'runs/detect/nadir/train/weights/best.pt',
        base / 'runs/detect/train/weights/best.pt',
        base / 'output/best.pt',
    )


def _auto_maskrcnn() -> Optional[str]:
    base = _BASE / 'seg' / 'mark_r_cnn'
    return _find(
        os.getenv('MASKRCNN_MODEL'),
        base / 'output/best_model.pth',
    )


def _auto_frcnn_oblique() -> Optional[str]:
    base = _BASE / 'class' / 'fasterrcnn_classification'
    return _find(
        os.getenv('FASTER_DET_OBLIQUE_MODEL'),
        base / 'runs/detect/train/oblique/fasterrcnn_oblique/best_model.pth',
    )


def _auto_frcnn_nadir() -> Optional[str]:
    base = _BASE / 'class' / 'fasterrcnn_classification'
    return _find(
        os.getenv('FASTER_DET_NADIR_MODEL'),
        base / 'runs/detect/train/nadir/fasterrcnn_nadir/best_model.pth',
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline toitures — Snapshot (YOLO) / Production (YOLO + Mask R-CNN)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--images', required=True,
        help='Dossier contenant les images\n(ou chemins séparés par des virgules)',
    )
    parser.add_argument(
        '--output', default=os.getenv('OUTPUT_DIR', './output'),
        help='Dossier de sortie (défaut : ./output)',
    )
    parser.add_argument(
        '--yolo-oblique-model', default=None,
        help='YOLO oblique best.pt  — Snapshot (class/yolo_classification)',
    )
    parser.add_argument(
        '--yolo-nadir-model', default=None,
        help='YOLO nadir best.pt    — Production (class/yolo_classification)',
    )
    parser.add_argument(
        '--frcnn-oblique-model', default=None,
        help='Faster R-CNN oblique best_model.pth  (class/fasterrcnn_classification)',
    )
    parser.add_argument(
        '--frcnn-nadir-model', default=None,
        help='Faster R-CNN nadir best_model.pth  (class/fasterrcnn_classification)',
    )
    parser.add_argument(
        '--maskrcnn-model', default=None,
        help='Mask R-CNN best_model.pth  (seg/mark_r_cnn)',
    )
    parser.add_argument(
        '--threshold', type=float,
        default=float(os.getenv('SCORE_THRESHOLD', '0.25')),
        help='Seuil de confiance global (défaut : 0.25)',
    )
    parser.add_argument(
        '--no-seg', action='store_true',
        help='Désactiver la segmentation Mask R-CNN',
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Ne pas afficher les comparaisons matplotlib (toujours sauvegardées)',
    )
    args = parser.parse_args()

    print('=' * 62)
    print('  Pipeline Toitures')
    print('  Snapshot   -> YOLO oblique + Faster R-CNN oblique (Soft+WA-NMS)')
    print('  Production -> YOLO nadir  + Faster R-CNN nadir  (Soft+WA-NMS)')
    print('               + Mask R-CNN segmentation (optionnel)')
    print('=' * 62)

    if not YOLO_AVAILABLE:
        print('\nERREUR : ultralytics non installe - pip install ultralytics')
        sys.exit(1)

    # ── Résolution des chemins modèles ────────────────────────────────────────
    oblique_path      = args.yolo_oblique_model  or _auto_yolo_oblique()
    nadir_path        = args.yolo_nadir_model    or _auto_yolo_nadir()
    frcnn_obl_path    = args.frcnn_oblique_model or _auto_frcnn_oblique()
    frcnn_nadir_path  = args.frcnn_nadir_model   or _auto_frcnn_nadir()
    maskrcnn_path     = args.maskrcnn_model      or _auto_maskrcnn()

    # ── YOLO obligatoires ────────────────────────────────────────────────────
    if not oblique_path:
        print('\nERREUR : YOLO oblique introuvable.')
        print('  Definir YOLO_DET_OBLIQUE_MODEL dans .env  ou  --yolo-oblique-model <chemin>')
        sys.exit(1)
    if not nadir_path:
        print('\nERREUR : YOLO nadir introuvable.')
        print('  Definir YOLO_DET_NADIR_MODEL dans .env  ou  --yolo-nadir-model <chemin>')
        sys.exit(1)

    yolo_thr   = _load_yolo_thresholds(args.threshold)
    frcnn_thr  = _load_frcnn_thresholds(args.threshold)
    fusion_thr = _load_fusion_thresholds(args.threshold)
    print(f'\nChargement des modeles (seuil global={args.threshold}) ...')

    # ── YOLO détecteurs ───────────────────────────────────────────────────────
    yolo_oblique = YOLODetector(oblique_path, threshold=args.threshold,
                                class_thresholds=yolo_thr)
    yolo_nadir   = YOLODetector(nadir_path,   threshold=args.threshold,
                                class_thresholds=yolo_thr)

    # ── Faster R-CNN + fusion DetFuser ────────────────────────────────────────
    det_oblique = det_nadir = None
    if TORCH_AVAILABLE:
        try:
            if frcnn_obl_path:
                frcnn_obl = FasterRCNNDetector(frcnn_obl_path, threshold=args.threshold,
                                               class_thresholds=frcnn_thr)
                det_oblique = DetFuser(yolo_oblique, frcnn_obl,
                                       conf_thr=args.threshold,
                                       class_thresholds=fusion_thr)
            else:
                print('\n[!] FRCNN oblique introuvable — detection oblique YOLO seul.')
                print('    Definir FASTER_DET_OBLIQUE_MODEL dans .env ou --frcnn-oblique-model')

            if frcnn_nadir_path:
                frcnn_nadir = FasterRCNNDetector(frcnn_nadir_path, threshold=args.threshold,
                                                 class_thresholds=frcnn_thr)
                det_nadir = DetFuser(yolo_nadir, frcnn_nadir,
                                     conf_thr=args.threshold,
                                     class_thresholds=fusion_thr)
            else:
                print('\n[!] FRCNN nadir introuvable — detection nadir YOLO seul.')
                print('    Definir FASTER_DET_NADIR_MODEL dans .env ou --frcnn-nadir-model')
        except Exception as exc:
            print(f'\n[!] Impossible de charger Faster R-CNN : {exc}')
            print('    Les detections seront effectuees avec YOLO uniquement.\n')
    else:
        print('\n[!] torch non installe - Faster R-CNN desactive (pip install torch torchvision).')

    # Fallback : si DetFuser non créé, utiliser YOLO seul comme wrapper minimal
    if det_oblique is None:
        det_oblique = yolo_oblique
    if det_nadir is None:
        det_nadir = yolo_nadir

    # ── Mask R-CNN segmentation (optionnel) ───────────────────────────────────
    ensemble = None
    if not args.no_seg and TORCH_AVAILABLE:
        if not maskrcnn_path:
            print('\n[!] Segmentation desactivee - MASKRCNN_MODEL introuvable.')
            print('    Definir MASKRCNN_MODEL dans .env ou utiliser --maskrcnn-model <chemin>')
        else:
            try:
                ensemble = MaskRCNNDetector(maskrcnn_path, threshold=args.threshold,
                                           class_thresholds=fusion_thr)
                print('   [OK] Mask R-CNN (segmentation) pret\n')
            except Exception as exc:
                print(f'\n[!] Impossible de charger Mask R-CNN : {exc}')
                print('    Les images Production seront traitees sans segmentation.\n')

    elif not TORCH_AVAILABLE and not args.no_seg:
        print('\n[!] torch non installe - segmentation desactivee (pip install torch torchvision).')

    # ── Exécution ─────────────────────────────────────────────────────────────
    pipeline = Pipeline(det_oblique, det_nadir, ensemble=ensemble,
                        output_dir=args.output, display=not args.no_display)

    if os.path.isdir(args.images):
        results = pipeline.process_directory(args.images)
    else:
        paths   = [p.strip() for p in args.images.split(',') if p.strip()]
        results = pipeline.process_images(paths)

    print(f'[OK] Termine - {len(results)} image(s) traitee(s) -> {args.output}')


if __name__ == '__main__':
    main()
