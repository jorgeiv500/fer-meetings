# Paper Progress

## Estado actual

- Corrida canónica: [results/latest_publication](../results/latest_publication)
- Dataset anotado: `100` clips AMI close-up
- Splits: `50 dev / 50 test`
- Familias fuente: `CNN`, `ViT`
- Extensiones evaluadas: `temporal pooling`, `calibración`, `clip-level adaptation`, `hybrid fusion`

## Hallazgos consolidados

- Mejor resultado principal en `test`: `ViT | Single frame`
  - `Macro-F1 = 0.4552`
  - `Balanced accuracy = 0.4472`
- Mejor resultado de adaptación a nivel de clip: `Fusion | Clip LogReg`
  - `Macro-F1 = 0.6244`
  - `Balanced accuracy = 0.6944`
- Acuerdo humano:
  - `100` clips doblemente evaluados
  - `observed agreement = 0.7700`
  - `Cohen's kappa = 0.6175`

## Lectura metodológica

- El backbone `ViT` es el mejor cero-shot en `test`, pero la mejora grande aparece al pasar a adaptación supervisada ligera a nivel de clip.
- El `CNN` puro se degrada mucho en cero-shot, aunque mejora de forma sustancial después de calibración.
- La fusión `CNN+ViT` es más valiosa como representación de clip que como ensemble probabilístico simple.
- Las matrices de confusión muestran que el cuello de botella sigue siendo separar `neutral` de los extremos, especialmente para los métodos zero-shot.

## Paquete visual final del paper

### Figuras core

1. [main_test_macro_f1.png](../results/ami_av_publication/paper_assets/figures/main_test_macro_f1.png)
2. [main_test_scorecard.png](../results/ami_av_publication/paper_assets/figures/main_test_scorecard.png)
3. [clip_models_macro_f1.png](../results/ami_av_publication/paper_assets/figures/clip_models_macro_f1.png)
4. [label_distribution.png](../results/ami_av_publication/paper_assets/figures/label_distribution.png)
5. [interrater_overview.png](../results/ami_av_publication/paper_assets/figures/interrater_overview.png)
6. [selected_confusions.png](../results/ami_av_publication/paper_assets/figures/selected_confusions.png)

### Tablas core

1. [main_model_comparison.csv](../results/ami_av_publication/paper_assets/tables/main_model_comparison.csv)
2. [clip_model_comparison.csv](../results/ami_av_publication/paper_assets/tables/clip_model_comparison.csv)
3. [dataset_summary.csv](../results/ami_av_publication/paper_assets/tables/dataset_summary.csv)
4. [label_distribution.csv](../results/ami_av_publication/paper_assets/tables/label_distribution.csv)
5. [interrater_summary.csv](../results/ami_av_publication/paper_assets/tables/interrater_summary.csv)

## Qué ya no entra en el paper-facing package

- curvas `PR/ROC` con demasiadas series
- todas las `confusion_*.png` por método
- tablas auxiliares que repetían per-class metrics sin aportar foco narrativo
- históricos de entrenamiento que no sostienen un claim principal del paper

Los outputs crudos siguen en `results/ami_av_publication` para análisis reproducible, pero no forman parte del set editorial.

## Pendientes para la escritura

1. Añadir una figura esquemática del pipeline completo.
2. Seleccionar 3-4 casos cualitativos de éxito/fallo/mejora por pooling.
3. Decidir si los desacuerdos entre evaluadores se adjudican antes del corte final del manuscrito.
4. Redactar la sección de error analysis con apoyo en `selected_confusions.png`.
5. Convertir `docs/paper_outline.md` en un draft de secciones con captions y mensajes por figura.

## Orden editorial sugerido

### Cuerpo principal

1. `dataset_summary.csv/md`
2. `label_distribution.png`
3. `interrater_summary.csv/md`
4. `interrater_overview.png`
5. `main_test_macro_f1.png`
6. `main_test_scorecard.png`
7. `main_model_comparison.csv/md`
8. `clip_models_macro_f1.png`
9. `clip_model_comparison.csv/md`
10. `selected_confusions.png`

### Mensaje narrativo por bloque

- **Datos y anotación**: el corpus es pequeño pero usable, y el acuerdo humano respalda la tarea.
- **Resultados principales**: el cambio de dominio degrada fuerte el zero-shot y no basta con mirar una sola métrica.
- **Adaptación de clip**: la mayor recuperación de desempeño aparece al usar embeddings de clip y adaptación ligera.
- **Análisis de error**: la frontera `neutral` vs extremos sigue siendo el punto más frágil.

## Estado de carpetas de resultados

- En `results` solo queda una corrida canónica: [results/ami_av_publication](../results/ami_av_publication)
- [results/latest_publication](../results/latest_publication) y `latest_paper_assets` son symlinks al run actual
- No se encontraron carpetas adicionales de corridas anteriores para borrar
