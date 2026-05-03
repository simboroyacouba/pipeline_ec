"""
pipeline/app.py — Interface web Flask pour le pipeline toitures

Usage :
  conda run -n maskrcnn python pipeline/app.py
  puis ouvrir http://localhost:5000
"""

import os
import sys
import json
import uuid
import shutil
from pathlib import Path

from flask import (Flask, render_template, request, redirect,
                   url_for, send_from_directory)

# ── Env + sys.path ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

try:
    from dotenv import load_dotenv
    load_dotenv(_HERE / '.env', override=True)
except ImportError:
    pass

sys.path.insert(0, str(_ROOT))

from pipeline.pipeline import (
    YOLODetector, YOLOSegDetector, MaskRCNNDetector, DeepLabDetector,
    SoftWANMSEnsemble, Pipeline,
    YOLO_AVAILABLE, TORCH_AVAILABLE,
)

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

SESSIONS_DIR  = _HERE / 'output' / 'sessions'
ALLOWED_EXTS  = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ── Filtre Jinja2 : formatage FCFA ────────────────────────────────────────────
@app.template_filter('fcfa')
def fcfa_filter(value):
    if value is None:
        return '—'
    return f"{int(value):,}".replace(',', ' ') + ' FCFA'


# ── Chargement du pipeline (une seule fois) ───────────────────────────────────
_pipeline: Pipeline = None


def _load_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    oblique_path = os.getenv('YOLO_DET_OBLIQUE_MODEL')
    nadir_path   = os.getenv('YOLO_DET_NADIR_MODEL')
    if not oblique_path or not nadir_path:
        raise RuntimeError(
            'YOLO_DET_OBLIQUE_MODEL / YOLO_DET_NADIR_MODEL non définis dans .env')

    threshold = float(os.getenv('SCORE_THRESHOLD', '0.25'))
    yolo_oblique = YOLODetector(oblique_path, threshold=threshold)
    yolo_nadir   = YOLODetector(nadir_path,   threshold=threshold)

    ensemble = None
    seg_path = os.getenv('YOLO_SEG_MODEL')
    mrc_path = os.getenv('MASKRCNN_MODEL')
    dlb_path = os.getenv('DEEPLAB_MODEL')
    if TORCH_AVAILABLE and seg_path and mrc_path and dlb_path:
        try:
            yolo_seg = YOLOSegDetector(seg_path, threshold=threshold)
            maskrcnn = MaskRCNNDetector(mrc_path, threshold=threshold)
            deeplab  = DeepLabDetector(dlb_path,
                                       backbone=os.getenv('DEEPLAB_BACKBONE', 'resnet50'))
            ensemble = SoftWANMSEnsemble(yolo_seg, maskrcnn, deeplab,
                                          conf_thr=threshold)
            print('[OK] Ensemble Soft+WA-NMS charge')
        except Exception as exc:
            print(f'[!] Ensemble non disponible : {exc}')

    _pipeline = Pipeline(yolo_oblique, yolo_nadir,
                         ensemble=ensemble,
                         output_dir=str(_HERE / 'output'))
    return _pipeline


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run():
    files = request.files.getlist('images')
    valid = [f for f in files
             if f.filename and Path(f.filename).suffix.lower() in ALLOWED_EXTS]

    if not valid:
        return render_template('index.html',
                               error='Aucune image valide (jpg / png / tif).')

    session_id  = uuid.uuid4().hex[:12]
    upload_dir  = SESSIONS_DIR / session_id / 'uploads'
    output_dir  = SESSIONS_DIR / session_id / 'output'
    upload_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    saved = []
    for f in valid:
        dest = upload_dir / Path(f.filename).name
        f.save(dest)
        saved.append(str(dest))

    try:
        pipeline = _load_pipeline()
        pipeline.out_dir   = str(output_dir)
        pipeline._ann_dir  = str(output_dir / 'annotated')
        pipeline._desc_dir = str(output_dir / 'descriptions')
        os.makedirs(pipeline._ann_dir,  exist_ok=True)
        os.makedirs(pipeline._desc_dir, exist_ok=True)
        pipeline.process_images(saved)
    except Exception as exc:
        shutil.rmtree(str(SESSIONS_DIR / session_id), ignore_errors=True)
        return render_template('index.html', error=f'Erreur pipeline : {exc}')

    return redirect(url_for('results', session_id=session_id))


@app.route('/results/<session_id>')
def results(session_id):
    fiche_path = SESSIONS_DIR / session_id / 'output' / 'fiche_finale.json'
    if not fiche_path.exists():
        return render_template('index.html', error='Résultats introuvables.')

    with open(fiche_path, encoding='utf-8') as f:
        fiche = json.load(f)

    ann_dir = SESSIONS_DIR / session_id / 'output' / 'annotated'
    images  = sorted(p.name for p in ann_dir.glob('*')
                     if p.suffix.lower() in ALLOWED_EXTS)

    return render_template('results.html',
                           fiche=fiche,
                           session_id=session_id,
                           images=images)


@app.route('/files/<session_id>/<filename>')
def serve_file(session_id, filename):
    ann_dir = SESSIONS_DIR / session_id / 'output' / 'annotated'
    return send_from_directory(str(ann_dir), filename)


if __name__ == '__main__':
    print('Chargement des modeles...')
    try:
        _load_pipeline()
        print('[OK] Pipeline pret')
    except Exception as exc:
        print(f'[!] {exc}')
    app.run(debug=False, host='0.0.0.0', port=5000)
