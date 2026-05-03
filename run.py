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
    YOLODetector, YOLOSegDetector, MaskRCNNDetector, DeepLabDetector,
    SoftWANMSEnsemble, Pipeline,
    YOLO_AVAILABLE, TORCH_AVAILABLE,
)


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


def _auto_yolo_seg() -> Optional[str]:
    for folder in ['yolo_26', 'yolo']:
        base = _BASE / 'seg' / folder
        p = _find(
            os.getenv('YOLO_SEG_MODEL'),
            base / 'runs/segment/output/train/weights/best.pt',
            base / 'runs/segment/train/weights/best.pt',
        )
        if p:
            return p
    return None


def _auto_maskrcnn() -> Optional[str]:
    base = _BASE / 'seg' / 'mark_r_cnn'
    return _find(
        os.getenv('MASKRCNN_MODEL'),
        base / 'output/best_model.pth',
    )


def _auto_deeplab() -> Optional[str]:
    for folder in ['DeepLabV3', 'deeplab']:
        base = _BASE / 'seg' / folder
        p = _find(
            os.getenv('DEEPLAB_MODEL'),
            base / 'output/best_model.pth',
        )
        if p:
            return p
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline toitures — Snapshot (YOLO) / Production (YOLO + Soft+WA-NMS)',
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
        '--yolo-seg-model', default=None,
        help='YOLO segmentation best.pt  (seg/yolo ou seg/yolo_26)',
    )
    parser.add_argument(
        '--maskrcnn-model', default=None,
        help='Mask R-CNN best_model.pth  (seg/mark_r_cnn)',
    )
    parser.add_argument(
        '--deeplab-model', default=None,
        help='DeepLabV3+ best_model.pth  (seg/DeepLabV3)',
    )
    parser.add_argument(
        '--backbone', default=os.getenv('DEEPLAB_BACKBONE', 'resnet50'),
        choices=['resnet50', 'resnet101'],
        help='Backbone DeepLabV3+ (défaut : resnet50)',
    )
    parser.add_argument(
        '--threshold', type=float,
        default=float(os.getenv('SCORE_THRESHOLD', '0.25')),
        help='Seuil de confiance global (défaut : 0.25)',
    )
    parser.add_argument(
        '--w-yolo',     type=float, default=1.0,
        help='Poids YOLO-seg dans l\'ensemble (défaut : 1.0)',
    )
    parser.add_argument(
        '--w-maskrcnn', type=float, default=1.0,
        help='Poids Mask R-CNN dans l\'ensemble (défaut : 1.0)',
    )
    parser.add_argument(
        '--w-deeplab',  type=float, default=0.8,
        help='Poids DeepLabV3+ dans l\'ensemble (défaut : 0.8)',
    )
    parser.add_argument(
        '--no-ensemble', action='store_true',
        help='Désactiver l\'ensemble — YOLO uniquement pour toutes les images',
    )
    args = parser.parse_args()

    print('=' * 62)
    print('  Pipeline Toitures')
    print('  Snapshot   -> YOLO oblique')
    print('  Production -> YOLO nadir  +  Soft+WA-NMS (YOLO-seg + MaskRCNN + DeepLab)')
    print('=' * 62)

    if not YOLO_AVAILABLE:
        print('\nERREUR : ultralytics non installe - pip install ultralytics')
        sys.exit(1)

    # ── Résolution des chemins modèles ────────────────────────────────────────
    oblique_path  = args.yolo_oblique_model or _auto_yolo_oblique()
    nadir_path    = args.yolo_nadir_model   or _auto_yolo_nadir()
    yolo_seg_path = args.yolo_seg_model     or _auto_yolo_seg()
    maskrcnn_path = args.maskrcnn_model     or _auto_maskrcnn()
    deeplab_path  = args.deeplab_model      or _auto_deeplab()

    # ── YOLO oblique (Snapshot) ───────────────────────────────────────────────
    if not oblique_path:
        print('\nERREUR : YOLO oblique introuvable.')
        print('  Definir YOLO_DET_OBLIQUE_MODEL dans .env  ou  --yolo-oblique-model <chemin>')
        sys.exit(1)

    # ── YOLO nadir (Production) ───────────────────────────────────────────────
    if not nadir_path:
        print('\nERREUR : YOLO nadir introuvable.')
        print('  Definir YOLO_DET_NADIR_MODEL dans .env  ou  --yolo-nadir-model <chemin>')
        sys.exit(1)

    print(f'\nChargement des modeles (seuil={args.threshold}) ...')
    yolo_oblique = YOLODetector(oblique_path, threshold=args.threshold)
    yolo_nadir   = YOLODetector(nadir_path,   threshold=args.threshold)

    # ── Ensemble Soft+WA-NMS (optionnel) ─────────────────────────────────────
    ensemble = None
    if not args.no_ensemble and TORCH_AVAILABLE:
        missing = []
        if not yolo_seg_path:
            missing.append('YOLO_SEG_MODEL')
        if not maskrcnn_path:
            missing.append('MASKRCNN_MODEL')
        if not deeplab_path:
            missing.append('DEEPLAB_MODEL')

        if missing:
            print(f'\n[!] Ensemble desactive - modeles manquants : {", ".join(missing)}')
            print('    Les images Production seront traitees avec YOLO uniquement.')
            print('    Definir ces variables dans .env ou utiliser --*-model')
        else:
            try:
                yolo_seg = YOLOSegDetector(yolo_seg_path, threshold=args.threshold)
                maskrcnn = MaskRCNNDetector(maskrcnn_path, threshold=args.threshold)
                deeplab  = DeepLabDetector(
                    deeplab_path, backbone=args.backbone)
                ensemble = SoftWANMSEnsemble(
                    yolo_seg, maskrcnn, deeplab,
                    w_yolo=args.w_yolo,
                    w_maskrcnn=args.w_maskrcnn,
                    w_deeplab=args.w_deeplab,
                    nms_iou_thr=0.5,
                    conf_thr=args.threshold,
                )
                print('   [OK] Ensemble Soft+WA-NMS pret\n')
            except Exception as exc:
                print(f'\n[!] Impossible de charger l\'ensemble : {exc}')
                print('    Les images Production seront traitees avec YOLO uniquement.\n')

    elif not TORCH_AVAILABLE and not args.no_ensemble:
        print('\n[!] torch non installe - ensemble desactive (pip install torch torchvision).')

    # ── Exécution ─────────────────────────────────────────────────────────────
    pipeline = Pipeline(yolo_oblique, yolo_nadir, ensemble=ensemble, output_dir=args.output)

    if os.path.isdir(args.images):
        results = pipeline.process_directory(args.images)
    else:
        paths   = [p.strip() for p in args.images.split(',') if p.strip()]
        results = pipeline.process_images(paths)

    print(f'[OK] Termine - {len(results)} image(s) traitee(s) -> {args.output}')


if __name__ == '__main__':
    main()
