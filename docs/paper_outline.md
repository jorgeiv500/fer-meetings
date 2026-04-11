# Paper Outline

## Título base

Cross-Domain Transfer of Facial Affect Recognition to Meeting Video: Comparing CNN and Vision Transformer Representations on the AMI Corpus

## Pregunta de investigación

¿Qué tan severa es la degradación fuera de dominio cuando modelos FER preentrenados se transfieren desde datasets faciales estándar a clips close-up de reuniones AMI, cómo cambia esa degradación entre backbones `CNN` y `Vision Transformer`, y cuánto se puede recuperar con agregación temporal y adaptación supervisada ligera a nivel de clip?

## Hipótesis principal

1. El rendimiento `single-frame` cae de forma marcada en AMI para cualquier backbone.
2. La agregación temporal mejora estabilidad y `macro-F1` frente a la predicción de un solo frame.
3. Las representaciones congeladas de clip son más útiles que las probabilidades frame a frame crudas para adaptación de bajo costo.
4. La diferencia entre `CNN` y `ViT` no debe asumirse a priori; el paper debe evaluar si la ventaja proviene de la arquitectura, de la calibración o de la representación temporal.

## Framing metodológico recomendado

No vender el trabajo como "aplicamos un modelo FER a AMI".

Venderlo como:

1. estudio de **generalización de dominio**;
2. comparación entre **familias de representación visual** (`CNN` vs `ViT`);
3. estudio de **agregación temporal de bajo costo** para valencia observable a nivel de clip;
4. evaluación de **adaptación ligera y reproducible** en un dataset objetivo pequeño.

## Qué ya soporta el proyecto

El repo actual ya implementa un piloto reproducible:

- manifiesto de clips AMI;
- inferencia frame-level con modelo Hugging Face;
- agregación temporal simple;
- evaluación contra `gold_label`;
- calibración ligera en `dev`.

Además, la clase `HfEmotionClassifier` ya usa `AutoImageProcessor` y `AutoModelForImageClassification`, así que puede cargar tanto modelos `CNN` como modelos `ViT` sin rediseñar el pipeline principal.

## Reenfoque técnico del proyecto

### Línea A. Baselines de transferencia directa

- `A0`: `single-frame` con frame central
- `A1`: `mean pooling` de probabilidades
- `A2`: `majority vote`
- `A3`: `temperature scaling` o calibración multinomial en `dev`

Esto preserva la lógica del repo actual y funciona como línea base obligatoria.

### Línea B. Comparación entre familias de backbone

Comparar al menos dos extractores fuente:

- `CNN-FER`: por ejemplo `ResNet` o `EfficientNet` entrenado para expresión facial
- `ViT-FER`: `ViT` o `DeiT` afinado para clasificación de emoción facial

Lo importante no es solo comparar accuracy final, sino analizar:

- robustez fuera de dominio;
- confianza/calibración;
- estabilidad temporal;
- sensibilidad a pose, blur, oclusión y baja expresividad.

### Línea C. Adaptación ligera a nivel de clip

A partir de embeddings congelados por frame:

- `C0`: `Logistic Regression` sobre embedding promedio del clip
- `C1`: `HistGradientBoosting` o `XGBoost` sobre features agregadas del clip
- `C2`: `Linear probe` sobre embedding pooled

Esto es más fuerte para la revista que quedarse solo en etiquetas frame-level, porque introduce análisis de representación sin requerir fine-tuning pesado.

### Línea D. Método más interesante y todavía defendible

La extensión más interesante dentro del alcance del proyecto no es una GNN forzada, sino una de estas dos:

1. **Temporal attention pooling / temporal transformer pequeño**
   - entrada: embeddings de varios frames del clip
   - salida: una sola predicción de valencia del clip
   - valor: modela expresividad sutil y estabilidad temporal mejor que promedio simple

2. **Análisis explícito de cambio de dominio en el espacio de representación**
   - medir separabilidad `FER2013 vs AMI` en embeddings `CNN` y `ViT`
   - usar métricas como distancia entre centroides, `MMD`, `CORAL loss` solo como análisis, o un clasificador de dominio
   - valor: convierte el paper en uno de pattern representation bajo domain shift, más alineado con la revista

Si hay que elegir solo una extensión, la mejor combinación de interés y factibilidad es:

- `CNN` vs `ViT`
- `mean pooling` vs `attention pooling`
- `linear probe` sobre embeddings de clip

## Propuesta de familias experimentales

### Familia 1. Zero-shot directo

- `E0-CNN`: single frame
- `E0-ViT`: single frame

### Familia 2. Agregación temporal sin entrenamiento

- `E1-CNN`: mean pooling / vote
- `E1-ViT`: mean pooling / vote

### Familia 3. Adaptación ligera supervisada

- `E2-CNN`: calibración y/o `linear probe` sobre clip embeddings
- `E2-ViT`: calibración y/o `linear probe` sobre clip embeddings

### Familia 4. Agregación temporal aprendida

- `E3-CNN`: temporal attention pooling sobre embeddings
- `E3-ViT`: temporal attention pooling sobre embeddings

## Recomendación de scope final

Para no romper el carácter de piloto, el paper debería cerrar con este núcleo:

1. dos backbones fuente: `CNN` y `ViT`;
2. tres niveles de inferencia:
   - frame único;
   - pooling temporal simple;
   - pooling temporal aprendido o `linear probe` sobre clip embeddings;
3. outcome humano de valencia en tres clases;
4. evaluación agrupada por reunión/escenario;
5. análisis de error por calidad visual y expresividad sutil.

No recomiendo como foco principal:

- GNN sobre clips;
- siete emociones discretas;
- fine-tuning completo del backbone en AMI;
- multimodalidad audio-video en esta primera versión.

Eso abriría demasiado el scope y debilitaría la claridad del paper.

## Paquete visual recomendado para la versión actual

### Figuras ya listas

1. `main_test_macro_f1.png`
   ranking horizontal claro de métodos principales en `test`
2. `main_test_scorecard.png`
   heatmap compacto con `Macro-F1`, `Balanced Accuracy` y `Accuracy`
3. `clip_models_macro_f1.png`
   ranking de adaptación supervisada a nivel de clip
4. `label_distribution.png`
   distribución de etiquetas por split
5. `interrater_overview.png`
   acuerdo entre evaluadores humanos
6. `selected_confusions.png`
   panel pequeño de matrices de confusión para los modelos representativos más fuertes

### Tablas ya listas

1. `dataset_summary.csv/md`
2. `interrater_summary.csv/md`
3. `main_model_comparison.csv/md`
4. `clip_model_comparison.csv/md`
5. `label_distribution.csv/md`

### Figuras que todavía conviene producir manualmente

1. Pipeline completo del experimento
2. Ejemplos visuales del cambio de dominio `FER2013` vs `AMI`
3. Casos cualitativos de acierto, fallo y mejora por agregación temporal

### Qué se decidió sacar del paquete editorial

- curvas `ROC/PR` con demasiadas series en una sola figura
- baterías completas de `confusion_*.png` por método
- gráficos con nombres de método demasiado largos en el eje x
- tablas auxiliares que repetían resultados sin ayudar a la historia central

La regla editorial ahora es: cada figura debe contestar una sola pregunta y seguir siendo legible sin leer el código del experimento.

## Resultados que sí serían publicables

- demostrar cuantitativamente que el dominio de reuniones rompe el supuesto FER clásico;
- mostrar si `ViT` ofrece una representación más robusta o no;
- mostrar que el pooling temporal simple ya recupera señal útil;
- mostrar si una adaptación ligera sobre embeddings supera claramente a la predicción zero-shot;
- reportar calibración y estabilidad, no solo accuracy.

## Resultados negativos que también sirven

Si `ViT` no supera a `CNN`, eso sigue siendo publicable si se demuestra que:

- el dataset objetivo es pequeño;
- la expresividad es sutil;
- la ganancia depende más del pooling temporal que de la arquitectura;
- el dominio objetivo favorece señales locales y faciales de baja resolución donde el `CNN` sigue siendo competitivo.

Si el temporal attention pooling no mejora al promedio simple, eso también es útil:

- implica que la mayor parte de la señal recuperable ya está en una agregación estable de bajo costo;
- fortalece el framing de método ligero y reproducible.

## Cambios concretos que conviene hacer en el código

1. Permitir múltiples `model_id` en configuración y registrar la familia (`cnn` / `vit`).
2. Exportar embeddings por frame además de probabilidades.
3. Guardar features agregadas de clip para entrenamiento ligero.
4. Separar claramente:
   - inferencia del backbone;
   - pooling temporal;
   - adaptación supervisada de clip;
   - evaluación y calibración.
5. Mantener un paquete de `paper_assets` curado:
   - pocas figuras
   - etiquetas cortas
   - variedad visual real
   - cero outputs redundantes en el directorio editorial

## Estado de avance

- `annotation_pack` ya soporta `humano 1`, `humano 2`, `adjudicado` y `acuerdo`
- el set actual tiene `100` clips doblemente evaluados
- el mejor zero-shot en `test` sigue siendo `ViT | Single frame`
- el mejor resultado global del proyecto es `Fusion | Clip LogReg`
- la siguiente prioridad de escritura ya no es más benchmarking, sino narrativa del paper y selección cualitativa

## Claim final recomendado

Este paper debe posicionarse como un estudio reproducible de transferencia de reconocimiento afectivo facial bajo cambio de dominio, centrado en la comparación entre representaciones `CNN` y `ViT` y en el valor real de la agregación temporal ligera para inferencia de valencia observable en video de reuniones.

## Claim que debes evitar

No afirmar:

- que el sistema reconoce estados emocionales profundos;
- que el modelo es listo para despliegue real;
- que `ViT` o cualquier arquitectura es superior de forma universal;
- que una mejora marginal implica comprensión contextual de la reunión.

## Orden recomendado del manuscrito

### Introducción

Abrir con tres ideas:

1. los modelos FER entrenados en datasets estándar sufren degradación fuerte al pasar a video de reuniones;
2. no está claro si el beneficio potencial de `ViT` sobre `CNN` sobrevive ese cambio de dominio;
3. una agregación temporal ligera y una adaptación supervisada pequeña pueden recuperar parte de la señal sin fine-tuning pesado.

No meter figuras aquí. La introducción debe vender el problema y la pregunta.

### Datos y anotación

Usar primero:

1. `dataset_summary.csv/md`
2. `label_distribution.png`
3. `interrater_summary.csv/md`
4. `interrater_overview.png`

Mensaje de la sección:

- el dataset es pequeño pero controlado y doblemente evaluado;
- la distribución por split es legible y suficiente para un piloto;
- el acuerdo humano es razonable, así que el cuello de botella no es solo ruido de anotación.

Captions base:

- **Figura. `label_distribution.png`**: Distribución de etiquetas de valencia en `dev` y `test`, mostrando un balance moderado entre clases y tamaños homogéneos por split.
- **Figura. `interrater_overview.png`**: Resumen del acuerdo entre los dos evaluadores humanos, con conteos de acuerdo/desacuerdo y métricas globales de consistencia.
- **Tabla. `dataset_summary.csv/md`**: Resumen del conjunto AMI close-up usado en el estudio, con número de clips, splits y cobertura de anotación.
- **Tabla. `interrater_summary.csv/md`**: Métricas de acuerdo entre evaluadores sobre los clips doblemente anotados.

### Protocolo experimental

Describir aquí las tres familias:

1. transferencia directa `single-frame`;
2. agregación temporal ligera sin entrenamiento fuerte;
3. adaptación supervisada a nivel de clip con embeddings congelados.

No hace falta una figura nueva si no está lista todavía. Si luego haces el diagrama del pipeline, este sería su lugar.

### Resultados principales

Orden recomendado:

1. `main_test_macro_f1.png`
2. `main_test_scorecard.png`
3. `main_model_comparison.csv/md`

Mensaje de la sección:

- comparar primero familias de método;
- después mostrar que la lectura no cambia al mirar varias métricas;
- cerrar con la tabla para dejar valores exactos y significancia práctica.

Captions base:

- **Figura. `main_test_macro_f1.png`**: Ranking de los métodos principales en `test` según `Macro-F1`, destacando la degradación fuera de dominio y las diferencias entre `CNN` y `ViT`.
- **Figura. `main_test_scorecard.png`**: Comparación compacta de `Macro-F1`, `Balanced Accuracy` y `Accuracy` para los métodos principales en `test`.
- **Tabla. `main_model_comparison.csv/md`**: Resultados cuantitativos completos de los métodos principales en el split `test`.

### Resultados de adaptación a nivel de clip

Orden recomendado:

1. `clip_models_macro_f1.png`
2. `clip_model_comparison.csv/md`

Mensaje de la sección:

- esta es la parte donde aparece la mejora importante del proyecto;
- conviene enfatizar que el mayor salto no viene del backbone solo, sino de la representación de clip y la adaptación ligera.

Captions base:

- **Figura. `clip_models_macro_f1.png`**: Desempeño de los modelos supervisados a nivel de clip, mostrando que la adaptación ligera sobre embeddings supera claramente a la transferencia directa.
- **Tabla. `clip_model_comparison.csv/md`**: Comparación detallada de los modelos de adaptación a nivel de clip y fusión entre backbones.

### Análisis de error

Usar:

1. `selected_confusions.png`

Mensaje de la sección:

- el error dominante es la separación entre `neutral` y los extremos;
- esta figura debe apoyar una discusión corta, no una galería extensa de matrices.

Caption base:

- **Figura. `selected_confusions.png`**: Matrices de confusión de modelos representativos, usadas para identificar patrones de error recurrentes bajo cambio de dominio.

## Secuencia final de assets en el paper

Si el manuscrito queda corto, la secuencia más limpia es:

1. `label_distribution.png`
2. `interrater_overview.png`
3. `main_test_macro_f1.png`
4. `main_test_scorecard.png`
5. `clip_models_macro_f1.png`
6. `selected_confusions.png`

Y las tablas:

1. `dataset_summary.csv/md`
2. `interrater_summary.csv/md`
3. `main_model_comparison.csv/md`
4. `clip_model_comparison.csv/md`

Con eso el paper mantiene una historia simple: datos y acuerdo, resultados principales, mejora por adaptación de clip, y análisis de error.
