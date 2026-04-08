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

Eso permite responder si la mejora viene de la arquitectura, del pooling temporal o de la adaptación ligera de clip. Para este paper, ese contraste es bastante más defendible que “probar solo ViT” o añadir una arquitectura nueva sin una hipótesis clara.

## Configuración canónica

- Config: [configs/ami_av_cnn_vit_publication.json](configs/ami_av_cnn_vit_publication.json)
- Manifest final: [data/interim/ami_av_manifest_publication.csv](data/interim/ami_av_manifest_publication.csv)
- Corrida final: [results/ami_av_publication](results/ami_av_publication)
- Assets finales: [results/ami_av_publication/paper_assets](results/ami_av_publication/paper_assets)

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

## Salidas finales

Tablas principales:

- [main_model_comparison.csv](results/ami_av_publication/paper_assets/tables/main_model_comparison.csv)
- [clip_model_comparison.csv](results/ami_av_publication/paper_assets/tables/clip_model_comparison.csv)
- [dataset_summary.csv](results/ami_av_publication/paper_assets/tables/dataset_summary.csv)
- [label_distribution.csv](results/ami_av_publication/paper_assets/tables/label_distribution.csv)

Figuras principales:

- [main_test_macro_f1.png](results/ami_av_publication/paper_assets/figures/main_test_macro_f1.png)
- [main_test_balanced_accuracy.png](results/ami_av_publication/paper_assets/figures/main_test_balanced_accuracy.png)
- [clip_models_macro_f1.png](results/ami_av_publication/paper_assets/figures/clip_models_macro_f1.png)
- [label_distribution.png](results/ami_av_publication/paper_assets/figures/label_distribution.png)

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

## Comandos cortos

```bash
make fetch-fer
make fetch-ami-av-subset
make manifest-ami-av
make experiment-prelabels
make experiment-postlabels
make test
```
