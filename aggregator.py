"""
pipeline/aggregator.py — Génération du rapport final agrégé

Lit tous les JSON de descriptions/ et produit rapport_final.json avec :

  panneaux_solaires
    ├── total / confidence_moyenne / par_image

  superficies_toiture       (ensemble Production)
    └── par_type
          ├── classe_modele / type_couverture
          ├── superficie_px / superficie_m2
          └── categorie_resolue   ← via class_mapping + categories_batiments

  caracteristiques_snapshot (YOLO Snapshot)
    └── par_classe
          ├── materiaux_construction / amenagement_facade / menuiserie_facade
          └── categorie_resolue   ← idem

Résolution de catégorie :
  Pour chaque champ (type_couverture, materiaux_construction,
  amenagement_facade, menuiserie_facade), on calcule un score :
    +2  correspondance exacte de token
    +1  correspondance partielle (substring)
     0  valeur null dans categories_batiments (wildcard)
    -2  aucune correspondance
  La catégorie retenue est celle au score le plus élevé.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Optional


# =============================================================================
# CHARGEMENT DES RÉFÉRENTIELS
# =============================================================================

def _load_mapping() -> dict:
    path = os.getenv('CLASS_MAPPING_FILE',
                     str(Path(__file__).parent / 'class_mapping.json'))
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _load_categories() -> list:
    path = os.getenv('CATEGORIES_FILE',
                     str(Path(__file__).parent / 'categories_batiments.json'))
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding='utf-8') as f:
        return json.load(f).get('categories_batiments', [])


def _resolve(mapping: dict, caracteristique: str, classe_modele: str) -> str:
    """Retourne la valeur lisible pour une classe modèle via class_mapping.json."""
    val = mapping.get(caracteristique, {}).get(classe_modele)
    if val:
        return val
    env_key = f"DEFAULT_{caracteristique.upper()}"
    default = os.getenv(env_key)
    return default if default else classe_modele


# =============================================================================
# RÉSOLUTION DE CATÉGORIE
# =============================================================================

FIELDS = ['type_couverture', 'materiaux_construction',
          'amenagement_facade', 'menuiserie_facade']


def _tokenize(val) -> set:
    """Convertit une valeur (str, list, None) en ensemble de tokens normalisés.

    "Tôle bac"          → {"tôle bac"}
    ["Tôle", "Tuile"]   → {"tôle", "tuile"}
    "Bois ou métallique"→ {"bois", "métallique"}
    null                → set vide
    """
    if val is None:
        return set()
    if isinstance(val, list):
        tokens = set()
        for item in val:
            tokens.update(_tokenize(item))
        return tokens
    parts = [p.strip().lower() for p in str(val).split(' ou ')]
    return {p for p in parts if p}


def _field_score(detected_val, category_val) -> int:
    """Score de correspondance entre une valeur détectée et une valeur de catégorie.

    Returns:
         0  category_val est null (wildcard — ne pénalise pas)
        +2  correspondance exacte de token
        +1  correspondance partielle (un token est substring de l'autre)
        -2  aucune correspondance
    """
    if category_val is None:
        return 0

    d_tok = _tokenize(detected_val)
    c_tok = _tokenize(category_val)

    if not d_tok:
        return 0  # valeur détectée inconnue → neutre

    # Correspondance exacte
    if d_tok & c_tok:
        return 2

    # Correspondance partielle (substring)
    for d in d_tok:
        for c in c_tok:
            if d in c or c in d:
                return 1

    return -2  # aucune correspondance


def resolve_category(
    categories: list,
    type_couverture: Optional[str]       = None,
    materiaux_construction: Optional[str] = None,
    amenagement_facade: Optional[str]    = None,
    menuiserie_facade: Optional[str]     = None,
) -> dict:
    """Trouve la catégorie de categories_batiments.json la mieux assortie.

    Args:
        categories             : liste chargée depuis categories_batiments.json
        type_couverture        : valeur détectée (peut être None)
        materiaux_construction : valeur détectée (peut être None)
        amenagement_facade     : valeur détectée (peut être None)
        menuiserie_facade      : valeur détectée (peut être None)

    Returns:
        dict avec id, type_batiment, score, score_max, confiance (0–1)
    """
    detected = {
        'type_couverture':       type_couverture,
        'materiaux_construction': materiaux_construction,
        'amenagement_facade':    amenagement_facade,
        'menuiserie_facade':     menuiserie_facade,
    }

    best_score = None
    best_cat   = None
    scores     = []

    for cat in categories:
        score     = 0
        max_score = 0
        for field in FIELDS:
            cat_val = cat.get(field)
            det_val = detected.get(field)
            fs      = _field_score(det_val, cat_val)
            score  += fs
            if cat_val is not None:          # null = wildcard, ne compte pas
                max_score += 2               # score max possible pour ce champ
        scores.append((score, max_score, cat))
        if best_score is None or score > best_score:
            best_score = score
            best_cat   = cat

    if best_cat is None:
        return {'id': None, 'type_batiment': None, 'score': 0, 'confiance': 0.0}

    # Calculer un score max théorique (tous les champs non-null de la meilleure cat)
    max_possible = next(mx for sc, mx, c in scores if c is best_cat)

    confiance = round(best_score / max_possible, 3) if max_possible > 0 else 0.0

    return {
        'id':           best_cat['id'],
        'type_batiment': best_cat['type_batiment'],
        'type_couverture_ref':        best_cat.get('type_couverture'),
        'materiaux_construction_ref': best_cat.get('materiaux_construction'),
        'amenagement_facade_ref':     best_cat.get('amenagement_facade'),
        'menuiserie_facade_ref':      best_cat.get('menuiserie_facade'),
        'score':        best_score,
        'score_max':    max_possible,
        'confiance':    confiance,
    }


# =============================================================================
# AGGREGATE
# =============================================================================

def aggregate(descriptions_dir: str,
              output_path: Optional[str] = None,
              resolution_m_px: Optional[float] = None) -> dict:
    """Agrège tous les JSON de descriptions/ en un rapport final.

    Args:
        descriptions_dir : dossier contenant les JSON individuels
        output_path      : si fourni, sauvegarde le rapport à ce chemin
        resolution_m_px  : GSD en m/px (priorité sur RESOLUTION_M_PX du .env)
    """
    if resolution_m_px is None:
        _env = os.getenv('RESOLUTION_M_PX')
        resolution_m_px = float(_env) if _env else None

    mapping    = _load_mapping()
    categories = _load_categories()

    json_files = sorted(Path(descriptions_dir).glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"Aucun JSON trouve dans {descriptions_dir}")

    # ── Collecte ───────────────────────────────────────────────────────────────
    panneaux: List[dict]       = []
    toitures: dict             = defaultdict(list)
    snap_classes: dict         = defaultdict(list)
    stats = {'total': 0, 'snapshot': 0, 'production': 0}

    for jf in json_files:
        with open(jf, encoding='utf-8') as f:
            data = json.load(f)

        stats['total'] += 1
        mode     = data.get('mode', 'unknown')
        img_name = data.get('image', jf.stem)
        if mode == 'snapshot':
            stats['snapshot'] += 1
        elif mode == 'production':
            stats['production'] += 1

        for det in (data.get('detection') or data.get('yolo', {})).get('detections', []):
            cls  = det['class']
            conf = det['confidence']
            bbox = det['bbox']
            area = det.get('area_px', 0)
            if cls == 'panneau_solaire':
                panneaux.append({'image': img_name, 'mode': mode,
                                 'confidence': conf, 'bbox': bbox, 'area_px': area})
            if mode == 'snapshot' and cls != 'panneau_solaire':
                snap_classes[cls].append({'image': img_name, 'confidence': conf,
                                          'bbox': bbox, 'area_px': area})

        if mode == 'production':
            seg_data  = data.get('segmentation', {})
            union_px  = seg_data.get('union_area_px_by_class', {})
            # added_cls : première occurrence de chaque classe dans cette image
            # → reçoit l'aire union ; les suivantes reçoivent 0 (pas de double-comptage)
            added_cls: set = set()
            for det in seg_data.get('detections', []):
                cls = det['class']
                if cls in union_px:
                    mask_area = union_px[cls] if cls not in added_cls else 0
                else:
                    mask_area = det.get('mask_area_px', det.get('area_px', 0))
                added_cls.add(cls)
                toitures[cls].append({'image': img_name, 'confidence': det['confidence'],
                                      'bbox': det['bbox'], 'mask_area_px': mask_area})

    # ── Panneaux solaires ──────────────────────────────────────────────────────
    panneau_par_image: dict = defaultdict(list)
    for p in panneaux:
        panneau_par_image[p['image']].append(p)

    panneau_section = {
        'total': len(panneaux),
        'confidence_moyenne': (
            round(sum(p['confidence'] for p in panneaux) / len(panneaux), 4)
            if panneaux else 0.0),
        'par_image': [
            {'image': img, 'count': len(dets),
             'confidence_moyenne': round(
                 sum(d['confidence'] for d in dets) / len(dets), 4),
             'instances': dets}
            for img, dets in sorted(panneau_par_image.items())
        ],
    }

    # ── Superficies toiture ────────────────────────────────────────────────────
    def _px_to_m2(px: int) -> Optional[float]:
        if resolution_m_px is None:
            return None
        return round(px * resolution_m_px ** 2, 2)

    toiture_par_type = {}
    total_superficie = 0
    for cls, instances in sorted(toitures.items()):
        superficie = sum(i['mask_area_px'] for i in instances)
        total_superficie += superficie

        type_couv = _resolve(mapping, 'type_couverture', cls)

        # Résolution catégorie avec type_couverture connu, défaults pour le reste
        cat = resolve_category(
            categories,
            type_couverture       = type_couv,
            materiaux_construction= os.getenv('DEFAULT_MATERIAUX_CONSTRUCTION'),
            amenagement_facade    = os.getenv('DEFAULT_AMENAGEMENT_FACADE'),
            menuiserie_facade     = os.getenv('DEFAULT_MENUISERIE_FACADE'),
        )

        entry = {
            'classe_modele':    cls,
            'type_couverture':  type_couv,
            'nombre_zones':     len(instances),
            'superficie_px':    superficie,
            'categorie_resolue': cat,
            'instances':        instances,
        }
        m2 = _px_to_m2(superficie)
        if m2 is not None:
            entry['superficie_m2'] = m2
        toiture_par_type[cls] = entry

    toiture_section = {'totale_px': total_superficie}
    totale_m2 = _px_to_m2(total_superficie)
    if totale_m2 is not None:
        toiture_section['totale_m2']       = totale_m2
        toiture_section['resolution_m_px'] = resolution_m_px
    else:
        toiture_section['note'] = (
            'superficie en pixels — definir RESOLUTION_M_PX dans .env pour convertir en m2')
    toiture_section['par_type'] = toiture_par_type

    # ── Caractéristiques Snapshot ──────────────────────────────────────────────
    snap_section: dict = {}
    for cls, instances in sorted(snap_classes.items()):
        mat  = _resolve(mapping, 'materiaux_construction', cls)
        amen = _resolve(mapping, 'amenagement_facade',     cls)
        men  = _resolve(mapping, 'menuiserie_facade',      cls)

        # Résolution catégorie avec les 3 champs détectés, défaut pour type_couverture
        cat = resolve_category(
            categories,
            type_couverture       = os.getenv('DEFAULT_TYPE_COUVERTURE'),
            materiaux_construction= mat,
            amenagement_facade    = amen,
            menuiserie_facade     = men,
        )

        snap_section[cls] = {
            'classe_modele':          cls,
            'materiaux_construction': mat,
            'amenagement_facade':     amen,
            'menuiserie_facade':      men,
            'total':                  len(instances),
            'confidence_moyenne':     round(
                sum(i['confidence'] for i in instances) / len(instances), 4),
            'categorie_resolue':      cat,
            'instances':              instances,
        }

    # ── Résumé textuel ─────────────────────────────────────────────────────────
    lignes = []

    if panneaux:
        lignes.append(
            f"{len(panneaux)} panneau(x) solaire(s) detecte(s) "
            f"(confiance moy. {panneau_section['confidence_moyenne']:.0%}).")

    if toiture_par_type:
        parties = []
        for cls, v in toiture_par_type.items():
            label = v['type_couverture']
            cat_id = v['categorie_resolue'].get('id', '?')
            if 'superficie_m2' in v:
                parties.append(
                    f"{v['nombre_zones']} zone(s) {label} ({v['superficie_m2']:,.1f} m2, cat.{cat_id})")
            else:
                parties.append(
                    f"{v['nombre_zones']} zone(s) {label} ({v['superficie_px']:,} px, cat.{cat_id})")
        total_str = (f"{totale_m2:,.1f} m2 (GSD={resolution_m_px} m/px)"
                     if totale_m2 is not None else f"{total_superficie:,} px")
        lignes.append(f"Toitures : {', '.join(parties)}. Total : {total_str}.")

    if snap_section:
        parties = []
        for cls, v in snap_section.items():
            cat_id = v['categorie_resolue'].get('id', '?')
            parties.append(
                f"{v['total']} {v['amenagement_facade']} / {v['materiaux_construction']} (cat.{cat_id})")
        lignes.append(f"Batiments (Snapshot) : {', '.join(parties)}.")

    resume = ' '.join(lignes) if lignes else "Aucune detection."

    # ── Rapport final ──────────────────────────────────────────────────────────
    rapport = {
        'generated_at':       datetime.now().isoformat(),
        'source':             str(descriptions_dir),
        'class_mapping_file': str(os.getenv('CLASS_MAPPING_FILE',
                                  str(Path(__file__).parent / 'class_mapping.json'))),
        'categories_file':    str(os.getenv('CATEGORIES_FILE',
                                  str(Path(__file__).parent / 'categories_batiments.json'))),
        'images_analysees':   stats,
        'resume_textuel':     resume,
        'panneaux_solaires':          panneau_section,
        'superficies_toiture':        toiture_section,
        'caracteristiques_snapshot':  snap_section,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rapport, f, indent=2, ensure_ascii=False)
        print(f"   Rapport final -> {output_path}")

        # Générer automatiquement la fiche simplifiée à côté du rapport
        fiche_path = str(Path(output_path).parent / 'fiche_finale.json')
        fiche = generate_fiche(rapport, mapping=mapping)
        with open(fiche_path, 'w', encoding='utf-8') as f:
            json.dump(fiche, f, indent=2, ensure_ascii=False)
        print(f"   Fiche finale  -> {fiche_path}")

    return rapport


# =============================================================================
# FICHE FINALE SIMPLIFIÉE
# =============================================================================

def generate_fiche(rapport: dict, mapping: dict = None) -> dict:
    """Génère la fiche finale simplifiée depuis un rapport agrégé.

    Structure retournée :
    {
      "batiment": {
        "type_couverture":        valeur dominante (superficie max),
        "materiaux_construction": valeur dominante (mapping explicite prioritaire),
        "amenagement_facade":     valeur dominante (mapping explicite prioritaire),
        "menuiserie_facade":      valeur dominante (mapping explicite prioritaire),
        "superficie_m2":          total (ou total_px si pas de GSD),
        "categorie":              { id, type_batiment, confiance }
      },
      "amenagement": {
        "panneau_solaire": { "nombre": N, "confidence_moyenne": X }
      }
    }

    Pour chaque champ façade, priorité aux classes qui ont un mapping explicite
    dans class_mapping.json (pas le défaut). En cas d'ex-aequo → plus grand total.
    """
    if mapping is None:
        mapping = _load_mapping()

    toiture_pt = rapport.get('superficies_toiture', {}).get('par_type', {})
    snap        = rapport.get('caracteristiques_snapshot', {})
    panneaux    = rapport.get('panneaux_solaires', {})

    # ── Type de couverture dominant (superficie la plus grande) ───────────────
    dominant_toiture = None
    if toiture_pt:
        dominant_toiture = max(
            toiture_pt.values(),
            key=lambda v: v.get('superficie_m2', v.get('superficie_px', 0))
        )

    # ── Valeur dominante par champ — priorité mapping explicite ───────────────
    def _dominant_for_field(field: str) -> Optional[str]:
        """Retourne la valeur la plus représentée pour ce champ.

        Donne la priorité aux classes ayant un mapping explicite dans
        class_mapping.json pour ce champ (vs celles qui tombent sur le défaut).
        """
        field_map = mapping.get(field, {})
        explicit: dict = {}   # cls_name → (value, total) — mapping direct
        implicit: dict = {}   # cls_name → (value, total) — valeur par défaut

        for cls_name, cls_data in snap.items():
            val   = cls_data.get(field)
            count = cls_data.get('total', 0)
            if not val:
                continue
            if cls_name in field_map:
                explicit[cls_name] = (val, count)
            else:
                implicit[cls_name] = (val, count)

        pool = explicit if explicit else implicit
        if not pool:
            return None
        return max(pool.values(), key=lambda x: x[1])[0]

    # ── Construction de la fiche ───────────────────────────────────────────────
    type_couverture = (dominant_toiture.get('type_couverture')
                       if dominant_toiture else
                       os.getenv('DEFAULT_TYPE_COUVERTURE'))

    materiaux   = (_dominant_for_field('materiaux_construction')
                   or os.getenv('DEFAULT_MATERIAUX_CONSTRUCTION'))
    amenagement = (_dominant_for_field('amenagement_facade')
                   or os.getenv('DEFAULT_AMENAGEMENT_FACADE'))
    # menuiserie_metallique prime toujours sur les autres classes
    if 'menuiserie_metallique' in snap:
        menuiserie = _resolve(mapping, 'menuiserie_facade', 'menuiserie_metallique')
    else:
        menuiserie = (_dominant_for_field('menuiserie_facade')
                      or os.getenv('DEFAULT_MENUISERIE_FACADE'))

    # ── Catégorie : classe snapshot la plus représentée ───────────────────────
    dominant_snap = max(snap.values(), key=lambda v: v.get('total', 0)) if snap else None

    # Superficie : total m2 si disponible, sinon px
    sup = rapport.get('superficies_toiture', {})
    if 'totale_m2' in sup:
        superficie_val  = sup['totale_m2']
        superficie_unit = 'm2'
    else:
        superficie_val  = sup.get('totale_px', 0)
        superficie_unit = 'px'

    # Catégorie : depuis l'entrée dominante (toiture ou snapshot)
    cat_id       = None
    cat_info     = None
    valeur_m2_k  = None   # milliers FCFA/m²

    if dominant_toiture and 'categorie_resolue' in dominant_toiture:
        cr = dominant_toiture['categorie_resolue']
    elif dominant_snap and 'categorie_resolue' in dominant_snap:
        cr = dominant_snap['categorie_resolue']
    else:
        cr = None

    if cr:
        cat_id = cr.get('id')
        # Récupérer la valeur au m² depuis categories_batiments.json
        categories = _load_categories()
        for cat in categories:
            if cat.get('id') == cat_id:
                valeur_m2_k = cat.get('valeur_m2_kfcfa')
                break
        cat_info = {
            'id':            cat_id,
            'type_batiment': cr.get('type_batiment'),
            'confiance':     cr.get('confiance'),
        }
        if valeur_m2_k is not None:
            cat_info['valeur_m2_fcfa'] = valeur_m2_k * 1000

    # ── Estimation financière ─────────────────────────────────────────────────
    prix_panneau = int(os.getenv('PRIX_PANNEAU_SOLAIRE_FCFA', '75000'))
    nb_panneaux  = panneaux.get('total', 0)

    valeur_batiment_fcfa  = None
    valeur_panneaux_fcfa  = nb_panneaux * prix_panneau
    estimation_totale_fcfa = None

    if valeur_m2_k is not None and superficie_unit == 'm2':
        superficie_affichee    = round(superficie_val, 1)   # même arrondi que l'affichage
        valeur_batiment_fcfa   = round(superficie_affichee * valeur_m2_k * 1000)
        estimation_totale_fcfa = valeur_batiment_fcfa + valeur_panneaux_fcfa

    # ── Construction de la fiche ──────────────────────────────────────────────
    batiment = {
        'type_couverture':        type_couverture,
        'materiaux_construction': materiaux,
        'amenagement_facade':     amenagement,
        'menuiserie_facade':      menuiserie,
        f'superficie_{superficie_unit}': round(superficie_val, 1) if superficie_unit == 'm2' else superficie_val,
    }
    if cat_info:
        batiment['categorie'] = cat_info
    if valeur_batiment_fcfa is not None:
        batiment['valeur_totale_fcfa'] = valeur_batiment_fcfa

    # ── Aménagement ───────────────────────────────────────────────────────────
    panneau_entry = {
        'nombre':             nb_panneaux,
        'confidence_moyenne': panneaux.get('confidence_moyenne', 0.0),
        'prix_unitaire_fcfa': prix_panneau,
        'valeur_totale_fcfa': valeur_panneaux_fcfa,
    }

    fiche = {
        'generated_at': rapport.get('generated_at'),
        'batiment':     batiment,
        'amenagement':  {'panneau_solaire': panneau_entry},
    }
    if estimation_totale_fcfa is not None:
        fiche['estimation_totale_fcfa'] = estimation_totale_fcfa

    return fiche
