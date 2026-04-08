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

## Tablas e imágenes planificadas desde el inicio

### Main text

- `6` figuras
- `5` tablas

### Appendix / supplementary

- `4` figuras
- `2` tablas

### Figuras principales sugeridas

1. Pipeline completo del experimento
2. Ejemplos del cambio de dominio `FER2013` vs `AMI`
3. Protocolo de anotación y construcción del outcome
4. Comparación principal `CNN` vs `ViT` y variantes temporales
5. Calibration / confusion / robustness en multipanel
6. Casos cualitativos: acierto, fallo y mejora por agregación temporal

### Tablas principales sugeridas

1. Descripción del dataset anotado
2. Acuerdo interanotador y consolidación
3. Benchmark principal de modelos
4. Sensibilidad por longitud de clip, pooling y mapeo a valencia
5. Error analysis por subgrupos y condiciones visuales

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
5. Añadir métricas de robustez:
   - `macro-F1`
   - `balanced accuracy`
   - `ECE` o `Brier`
   - métricas por subgrupo de calidad facial

## Claim final recomendado

Este paper debe posicionarse como un estudio reproducible de transferencia de reconocimiento afectivo facial bajo cambio de dominio, centrado en la comparación entre representaciones `CNN` y `ViT` y en el valor real de la agregación temporal ligera para inferencia de valencia observable en video de reuniones.

## Claim que debes evitar

No afirmar:

- que el sistema reconoce estados emocionales profundos;
- que el modelo es listo para despliegue real;
- que `ViT` o cualquier arquitectura es superior de forma universal;
- que una mejora marginal implica comprensión contextual de la reunión.
