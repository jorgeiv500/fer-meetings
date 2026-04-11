# Resultados versionados

Este directorio conserva solo los artefactos que ayudan a inspeccionar, reproducir y escribir el experimento sin subir datos descargados ni dumps pesados regenerables.

## Se versiona

- `annotation_pack/`: paquete HTML para revisar etiquetas y validaciones humanas
- `clip_labels.csv`: etiquetas consolidadas por clip
- `metrics.csv`, `metrics.json`, `confusion_matrices.csv`, `per_class_metrics.csv`: resumen de evaluación principal
- `clip_models/`: métricas y resúmenes compactos de adaptación a nivel de clip
- `paper_assets/`: figuras y tablas curadas para el manuscrito
- `reports/`: fichas de experimento, datos, ética y reproducibilidad
- `summary.md`: resumen rápido del run canónico

## No se versiona

- videos y snapshots descargados desde Hugging Face
- `frame_details.csv` y `clip_features.csv`
- dumps completos de predicciones frame-level o clip-level
- curvas exhaustivas `PR/ROC` y archivos `.zip` duplicados del mismo contenido

## Cómo regenerar lo omitido

Desde la raíz del repo:

```bash
make fetch-fer
make fetch-ami-av-subset
make manifest-ami-av
make experiment-prelabels
make experiment-postlabels
```

La configuración canónica usada para esta corrida es `configs/ami_av_cnn_vit_publication.json`.
