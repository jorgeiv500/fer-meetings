# fer-meetings

Estudio reproducible de transferencia de reconocimiento afectivo facial desde modelos FER estándar hacia video de reuniones AMI, con foco en comparación entre `CNN` y `Vision Transformer` bajo cambio de dominio.

## Qué evalúa este repo

La evaluación correcta aquí no es `ViT` en aislamiento. El protocolo final contrasta:

- un backbone `CNN`: `firedfrog/convnext-tiny-224-emotion-prediction`
- un backbone `ViT`: `mo-thecreator/vit-Facial-Expression-Recognition`

ambos bajo exactamente las mismas familias experimentales:

- `single-frame`
- `temporal pooling` (`smoothed`, `vote`)
- `calibración` en `dev`
- `clip-level adaptation` sobre embeddings congelados (`mean_embedding_logreg`, `mean_embedding_hgb`, `attention_pooling`)
- `hybrid fusion` entre `CNN` y `ViT`:
  - ensemble probabilístico (`mean`, `entropy-weighted`)
  - fusión de embeddings a nivel de clip (`cnn_vit_fusion`)
  - adaptación ligera `CORAL` y `MMD` sobre embeddings de clip

Eso permite responder si la mejora viene de la arquitectura, del pooling temporal o de la adaptación ligera de clip. Para este paper, ese contraste es bastante más defendible que “probar solo ViT” o añadir una arquitectura nueva sin una hipótesis clara.

## Configuración canónica

- Config: [configs/ami_av_cnn_vit_publication.json](configs/ami_av_cnn_vit_publication.json)
- Manifest final: [data/interim/ami_av_manifest_publication.csv](data/interim/ami_av_manifest_publication.csv)
- Corrida final: [results/ami_av_publication](results/ami_av_publication)
- Assets finales: [results/ami_av_publication/paper_assets](results/ami_av_publication/paper_assets)
- Estado para escritura: [docs/paper_progress.md](docs/paper_progress.md)

## Reproducibilidad y artefactos versionados

El repositorio conserva el código, la configuración, el manifiesto final y un conjunto curado de resultados listos para inspección y escritura. Los datos descargados, videos y dumps pesados regenerables no se versionan.

Se conservan en Git:

- código fuente, tests y configuración
- `data/interim/ami_av_manifest_publication.csv`
- `results/ami_av_publication/annotation_pack`
- `results/ami_av_publication/paper_assets`
- `results/ami_av_publication/reports`
- métricas compactas y etiquetas consolidadas del run canónico

No se suben al repositorio:

- videos AMI descargados
- snapshots completos de datasets
- `frame_details.csv`, `clip_features.csv` y dumps completos de predicción
- zips duplicados de resultados

Todo lo omitido se puede regenerar con los comandos del flujo recomendado y la configuración canónica.

## Flujo recomendado

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 1. Datos

Descargar `FER2013` desde Hugging Face:

```bash
make fetch-fer
```

Descargar subconjunto utilizable de `ami-av`:

```bash
make fetch-ami-av-subset
make manifest-ami-av
```

### 2. Corrida prelabels

```bash
make experiment-prelabels
```

Eso deja:

- `predictions.csv`
- `frame_details.csv`
- `clip_features.csv`
- `annotation_pack/index.html`
- `annotation_pack/annotation_sheet.csv`

### 3. Anotación humana

Abre:

- [results/ami_av_publication/annotation_pack/index.html](results/ami_av_publication/annotation_pack/index.html)

La interfaz permite:

- asignar `gold_label` por clip
- visualizar `humano 1`, `humano 2`, `adjudicado` y `acuerdo`
- filtrar por `split`, `labeled/unlabeled` y texto
- guardar una copia CSV actualizada desde el navegador

Después de exportar, reemplaza el archivo del proyecto:

- [results/ami_av_publication/annotation_pack/annotation_sheet.csv](results/ami_av_publication/annotation_pack/annotation_sheet.csv)

### 4. Corrida postlabels

```bash
make experiment-postlabels
```

Eso ejecuta:

- resolución de labels
- `scenario_splits`
- evaluación principal
- modelos clip-level
- tablas y figuras
- reportes reproducibles

## Salidas finales para paper

Tablas principales:

- [main_model_comparison.csv](results/ami_av_publication/paper_assets/tables/main_model_comparison.csv)
- [clip_model_comparison.csv](results/ami_av_publication/paper_assets/tables/clip_model_comparison.csv)
- [dataset_summary.csv](results/ami_av_publication/paper_assets/tables/dataset_summary.csv)
- [label_distribution.csv](results/ami_av_publication/paper_assets/tables/label_distribution.csv)
- [interrater_summary.csv](results/ami_av_publication/paper_assets/tables/interrater_summary.csv)

Figuras principales:

- [main_test_macro_f1.png](results/ami_av_publication/paper_assets/figures/main_test_macro_f1.png)
- [main_test_scorecard.png](results/ami_av_publication/paper_assets/figures/main_test_scorecard.png)
- [clip_models_macro_f1.png](results/ami_av_publication/paper_assets/figures/clip_models_macro_f1.png)
- [label_distribution.png](results/ami_av_publication/paper_assets/figures/label_distribution.png)
- [interrater_overview.png](results/ami_av_publication/paper_assets/figures/interrater_overview.png)
- [selected_confusions.png](results/ami_av_publication/paper_assets/figures/selected_confusions.png)

Reportes:

- [experiment_card.md](results/ami_av_publication/reports/experiment_card.md)
- [data_sheet.md](results/ami_av_publication/reports/data_sheet.md)
- [limitations_and_ethics.md](results/ami_av_publication/reports/limitations_and_ethics.md)
- [reproducibility_checklist.md](results/ami_av_publication/reports/reproducibility_checklist.md)

## Interpretación metodológica

Para este proyecto, la mejor forma de evaluar `ViT` no es “ponerlo solo” ni “añadir algo nuevo” sin hipótesis. El diseño fuerte es:

1. comparar `ViT` contra un `CNN` competitivo bajo el mismo pipeline;
2. medir qué cambia entre `single-frame`, `pooling` y `clip-level adaptation`;
3. usar `macro-F1` y `balanced accuracy` como métricas centrales;
4. tratar el problema como cambio de dominio y representación, no como `SOTA` de FER.

Si luego quieres extender el paper, la siguiente mejora razonable no es una `GNN` forzada; sería uno de estos dos:

- análisis explícito de `domain shift` en el espacio de embeddings (`CNN` vs `ViT`)
- un mejor bloque temporal sobre embeddings congelados

## Criterio de poda de resultados

El directorio `paper_assets` ahora es un paquete curado de resultados para escritura. Ahí solo quedan las tablas y figuras que sí entran en la narrativa del paper:

- comparación principal en test
- scorecard compacto de métricas
- ranking de adaptación a nivel de clip
- distribución de etiquetas
- acuerdo entre evaluadores
- panel pequeño de matrices de confusión

Los archivos crudos del pipeline (`predictions.csv`, `metrics.json`, `confusion_matrices.csv`, `clip_features.csv`, etc.) se conservan en `results/ami_av_publication` para reproducibilidad, pero ya no se duplican como assets de paper si no aportan claridad visual o narrativa.

## Comandos cortos

```bash
make fetch-fer
make fetch-ami-av-subset
make manifest-ami-av
make experiment-prelabels
make experiment-postlabels
make test
```
