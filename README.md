# fer-meetings

Repositorio local y replicable para convertir el proyecto del ZIP en un piloto publicable: transferir modelos FER preentrenados a video de reuniones AMI y medir cuánto se recupera con agregación temporal, calibración ligera y adaptación clip-level sobre embeddings.

Target recomendado para arrancar:

- `hhoangphuoc/ami-av` en Hugging Face, porque es la variante de AMI más fácil de poner a correr primero.
- El protocolo con `Closeup*.avi` del corpus oficial sigue siendo la referencia más fiel, pero ya no es la opción más simple.

Configuración experimental recomendada para el paper:

- [configs/ami_av_cnn_vit.json](configs/ami_av_cnn_vit.json)
- `CNN`: `firedfrog/convnext-tiny-224-emotion-prediction`
- `ViT`: `mo-thecreator/vit-Facial-Expression-Recognition`

## Qué cambia frente al notebook original

El ZIP original mezcla notebooks de Colab, montaje de Drive, comparación de modelos y un video propio. Este repo lo reorienta a una pregunta más clara y defendible:

> ¿Qué tan bien transfiere un modelo FER entrenado fuera de dominio a reuniones AMI y cuánto se recupera con una corrección de muy bajo costo?

El flujo base queda reducido a:

1. Crear un manifiesto de clips desde segmentos AMI procesados (`ami-av`) o desde videos `Closeup` del corpus oficial.
2. Etiquetar un conjunto pequeño en `positive / neutral / negative`.
3. Ejecutar inferencia zero-shot con uno o varios modelos FER de Hugging Face.
4. Exportar probabilidades y embeddings por frame/clip.
5. Comparar `single-frame` vs `temporal smoothing`.
6. Opcionalmente calibrar con el split `dev`.
7. Entrenar probes clip-level y `attention pooling` sobre embeddings.

## Alcance experimental

Este repo está pensado para un piloto disciplinado, no para entrenar un sistema nuevo desde cero.

- `E0`: predicción del frame central del clip.
- `E1`: agregación temporal de varios frames por clip.
- `E2`: calibración ligera usando clips etiquetados del split `dev`.
- `E3`: `linear probe` o `HistGradientBoosting` sobre embedding promedio del clip.
- `E4`: `attention pooling` temporal sobre secuencias cortas de embeddings.

## Assets externos

- Modelo FER preentrenado: `HardlyHumans/Facial-expression-detection`
  - https://huggingface.co/HardlyHumans/Facial-expression-detection
- Dataset fuente en Hugging Face:
  - `Jeneral/fer-2013`
- Plantilla para comparación `CNN` vs `ViT`:
  - `DrGM/DrGM-ConvNeXt-V2L-Facial-Emotion-Recognition`
  - `trpakov/vit-face-expression`
- Corpus AMI oficial:
  - overview: https://groups.inf.ed.ac.uk/ami/corpus/
  - splits oficiales: https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
  - señales de audio/video y nombres `Closeup`: https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml
- Alternativa AMI procesada en Hugging Face:
  - `hhoangphuoc/ami-av`
- Si luego quieres reentrenar el source model con FER2013:
  - busca una copia en Hugging Face o usa tu copia local del ZIP original.

## Estructura

```text
configs/                 Configuración del piloto y mapeo de etiquetas
data/
  raw/                   Videos AMI descargados manualmente
  interim/               Manifiestos de clips
  annotations/           CSVs de etiquetado manual
docs/                    Outline de paper y notas metodológicas
results/                 Predicciones, métricas y tablas
src/fer_meetings/        Código del proyecto
tests/                   Pruebas ligeras de lógica
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 0. Descargar el dataset fuente desde Hugging Face

```bash
fer-fetch-hf-data \
  --dataset-id Jeneral/fer-2013 \
  --output-dir data/raw/hf/fer2013 \
  --snapshot-only \
  --bytes-column img_bytes \
  --label-column labels
```

Esto descarga el snapshot del repo del dataset en Hugging Face, exporta `FER2013` por split y escribe `metadata.csv` por cada uno.

Si usas el corpus oficial, pon los videos AMI en `data/raw/ami/` o en cualquier subcarpeta dentro de `data/raw/`.

### 1. Construir el manifiesto de clips

#### Opción recomendada: `ami-av`

Si ya tienes los segmentos `mp4` extraídos localmente:

```bash
fer-build-ami-av-manifest \
  --metadata-csv data/raw/hf/ami_av_meta/ami-segments-info.csv \
  --video-root data/raw/hf/ami_av/video_segments/original_videos \
  --output data/interim/ami_manifest.csv
```

#### Opción alternativa: `Closeup*.avi` del corpus oficial

```bash
fer-build-manifest \
  --config configs/pilot_fast.json \
  --raw-dir data/raw \
  --output data/interim/ami_manifest.csv
```

### 2. Crear la plantilla de etiquetado

```bash
fer-make-label-template \
  --manifest data/interim/ami_manifest.csv \
  --output data/annotations/ami_gold_labels.csv
```

Rellena manualmente la columna `gold_label` con `negative`, `neutral` o `positive`.

Para una versión de calidad de publicación, usa preferentemente estos campos:

- `rater_1_label`
- `rater_2_label`
- `adjudicated_label`
- `exclude_from_gold`

El campo `gold_label` puede seguir usándose como etiqueta final directa, pero para reportar acuerdo interanotador conviene resolverlo después con `fer-compute-interrater`.

Si quieres revisar clips más rápido antes de etiquetar, genera un paquete local con thumbnails y visor HTML:

```bash
fer-build-annotation-pack \
  --manifest data/interim/ami_av_manifest.csv \
  --predictions results/ami_av_pilot_convnext/predictions.csv \
  --output-dir results/ami_av_pilot_convnext/annotation_pack
```

Eso crea:

- `annotation_pack/annotation_sheet.csv`
- `annotation_pack/index.html`
- `annotation_pack/thumbnails/*.jpg`

### 2b. Correr el experimento completo por fases

Fase 1, antes de etiquetar:

```bash
fer-run-experiment \
  --config configs/ami_av_cnn_vit.json \
  --manifest data/interim/ami_av_manifest.csv \
  --output-dir results/ami_av_experiment \
  --phase prelabels \
  --frames-per-clip 3
```

Eso deja:

- `results/ami_av_experiment/predictions.csv`
- `results/ami_av_experiment/frame_details.csv`
- `results/ami_av_experiment/clip_features.csv`
- `results/ami_av_experiment/annotation_pack/annotation_sheet.csv`
- `results/ami_av_experiment/annotation_pack/index.html`

Fase 2, después de llenar `gold_label`:

```bash
fer-run-experiment \
  --config configs/ami_av_cnn_vit.json \
  --manifest data/interim/ami_av_manifest.csv \
  --output-dir results/ami_av_experiment \
  --phase postlabels
```

Eso ejecuta:

- evaluación zero-shot y temporally pooled
- modelos clip-level
- tablas y figuras del paper
- resolución de `clip_labels.csv`
- `interrater_agreement.csv`
- `scenario_splits.json`
- reportes de publicación en `reports/`

## Flujo de publicación recomendado

Una vez tengas la hoja de anotación completa, estos son los artefactos que dejan el estudio en una forma más publicable:

```bash
fer-compute-interrater \
  --labels results/ami_av_experiment/annotation_pack/annotation_sheet.csv \
  --output-dir results/ami_av_experiment

fer-build-scenario-splits \
  --manifest data/interim/ami_av_manifest.csv \
  --output results/ami_av_experiment/scenario_splits.json

fer-generate-reporting \
  --config configs/ami_av_cnn_vit.json \
  --manifest data/interim/ami_av_manifest.csv \
  --labels results/ami_av_experiment/clip_labels.csv \
  --pilot-dir results/ami_av_experiment \
  --clip-model-dir results/ami_av_experiment/clip_models \
  --output-dir results/ami_av_experiment/reports
```

Eso añade:

- `clip_labels.csv`
- `interrater_agreement.csv`
- `interrater_pairs.csv` cuando existan dos anotadores
- `scenario_splits.json`
- `reports/model_registry.yaml`
- `reports/experiment_card.md`
- `reports/data_sheet.md`
- `reports/limitations_and_ethics.md`
- `reports/reproducibility_checklist.md`

### 3. Correr el piloto zero-shot

Configuración recomendada para arrancar en esta máquina:

- `configs/pilot_fast_convnext.json`
- modelo: `firedfrog/convnext-tiny-224-emotion-prediction`
- target: `data/interim/ami_av_manifest.csv`

```bash
fer-run-pilot \
  --config configs/pilot_fast_convnext.json \
  --manifest data/interim/ami_av_manifest.csv \
  --output results/pilot/predictions.csv \
  --frame-details-output results/pilot/frame_details.csv \
  --clip-features-output results/pilot/clip_features.csv
```

### 4. Evaluar

```bash
fer-evaluate \
  --predictions results/pilot/predictions.csv \
  --labels data/annotations/ami_gold_labels.csv \
  --output-dir results/pilot \
  --fit-calibrator
```

### 5. Entrenar modelos clip-level sobre embeddings

```bash
fer-train-clip-models \
  --clip-features results/pilot/clip_features.csv \
  --labels data/annotations/ami_gold_labels.csv \
  --output-dir results/pilot/clip_models
```

### 6. Generar tablas y figuras del paper

```bash
fer-make-paper-assets \
  --pilot-dir results/pilot \
  --clip-model-dir results/pilot/clip_models \
  --output-dir results/paper_assets
```

Los artefactos principales quedan en `results/pilot/`:

- `predictions.csv`
- `frame_details.csv`
- `clip_features.csv`
- `metrics.json`
- `metrics.csv`
- `per_class_metrics.csv`
- `labeled_predictions.csv`
- `summary.md`
- `confusion_matrices.csv`
- `clip_labels.csv`
- `interrater_agreement.csv`
- `scenario_splits.json`

Y para la línea clip-level:

- `clip_models/clip_model_metrics.json`
- `clip_models/clip_model_metrics.csv`
- `clip_models/clip_model_per_class_metrics.csv`
- `clip_models/clip_model_confusion_matrices.csv`
- `clip_models/clip_model_predictions.csv`
- `clip_models/clip_model_summary.md`

Y para el manuscrito:

- `../paper_assets/tables/main_model_comparison.{csv,md}`
- `../paper_assets/tables/main_per_class_metrics.{csv,md}`
- `../paper_assets/tables/clip_model_comparison.{csv,md}`
- `../paper_assets/tables/clip_model_per_class_metrics.{csv,md}`
- `../paper_assets/tables/dataset_summary.{csv,md}`
- `../paper_assets/tables/label_distribution.{csv,md}`
- `../paper_assets/figures/main_test_macro_f1.png`
- `../paper_assets/figures/main_test_balanced_accuracy.png`
- `../paper_assets/figures/clip_models_macro_f1.png`
- `../paper_assets/figures/label_distribution.png`
- `../paper_assets/figures/confusion_*.png`

Y para la capa editorial/publicable:

- `../reports/model_registry.yaml`
- `../reports/experiment_card.md`
- `../reports/data_sheet.md`
- `../reports/limitations_and_ethics.md`
- `../reports/reproducibility_checklist.md`

## Protocolo de una tarde

La configuración incluida en [`configs/pilot_fast.json`](configs/pilot_fast.json) crea aproximadamente 100 clips:

- `dev`: `ES2003a.Closeup1.avi`, `ES2003a.Closeup2.avi`
- `test`: `ES2004a.Closeup1.avi`, `ES2004a.Closeup2.avi`
- `3s` por clip
- `12s` entre clips
- `25` clips por video

Con eso tienes:

- suficiente material para escribir un pilot study;
- un `dev` pequeño para calibración;
- un `test` separado para reportar el resultado final;
- una base limpia para comparar `CNN` vs `ViT`.

## Notas de implementación

- El detector facial por defecto usa `OpenCV Haar Cascade`, no MediaPipe, para bajar fricción.
- Si no detecta cara en un frame, hace fallback al frame completo para no romper el pipeline.
- El mapeo por defecto reduce las salidas del modelo a tres clases usando [`configs/label_map_3class.json`](configs/label_map_3class.json).
- `fer-run-pilot` acepta múltiples backbones desde la configuración `models` o repitiendo `--model-id`.
- Si exportas `clip_features.csv`, el pipeline guarda embeddings por frame, embedding promedio del clip y señales temporales simples.
- `fer-fetch-hf-data` puede materializar datasets de Hugging Face con columnas de imagen o video. En este repo ya deja resuelto el dataset fuente `FER2013`.
- `fer-build-ami-av-manifest` crea un `dev/test` determinista por `meeting_id` a partir de segmentos `ami-av` de 3 a 5 segundos.
- La calibración es deliberadamente simple: una regresión logística multinomial entrenada sobre probabilidades del split `dev`.
- Los modelos clip-level supervisados entrenan en `split=dev` y se evalúan en `split=test`.
- A fecha de esta revisión, no encontré una copia utilizable de **AMI close-up video** en Hugging Face equivalente a las cámaras `Closeup*.avi`; el mirror más cercano encontrado es `hhoangphuoc/ami-av`, que publica segmentos `mp4` procesados de AMI para AVSR. Si quieres mantener exactamente el protocolo `Closeup*.avi`, sigue siendo necesario descargar el corpus oficial; si aceptas segmentos AMI procesados, ese mirror sí puede servir como target alternativo.

## Comandos cortos con `make`

```bash
make fetch-fer
make fetch-ami-av-subset
make manifest-ami-av
make manifest
make labels
make annotation-pack
make publication-labels
make pilot
make pilot-ami-av-fast
make experiment-ami-av-prelabels
make experiment-ami-av-postlabels
make evaluate
make clip-models
make paper-assets
make publication-reports
make test
```

El primer flujo que ya quedó validado extremo a extremo en este repo es:

```bash
make fetch-ami-av-subset
make manifest-ami-av
fer-make-label-template \
  --manifest data/interim/ami_av_manifest.csv \
  --output data/annotations/ami_av_gold_labels.csv
make pilot-ami-av-fast
fer-build-annotation-pack \
  --manifest data/interim/ami_av_manifest.csv \
  --predictions results/ami_av_pilot_convnext/predictions.csv \
  --output-dir results/ami_av_pilot_convnext/annotation_pack
```

Eso deja estas salidas listas:

- `results/ami_av_pilot_convnext/predictions.csv`
- `results/ami_av_pilot_convnext/frame_details.csv`
- `results/ami_av_pilot_convnext/clip_features.csv`
- `results/ami_av_pilot_convnext/annotation_pack/index.html`

## Subir a GitHub

```bash
git init -b main
git add .
git commit -m "Initialize FER-to-AMI pilot repo"
gh repo create <tu-repo> --private --source=. --push
```

No subas `data/raw`, `data/interim`, `data/annotations` ni `results/`; ya están ignorados por `.gitignore`.
