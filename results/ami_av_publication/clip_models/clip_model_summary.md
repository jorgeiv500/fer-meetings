# Clip-Level Model Summary

| model | family | method | n_clips | accuracy | balanced_accuracy | macro_f1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| convnext_tiny_emotion | cnn | attention_pooling | 50 | 0.6600 | 0.6806 | 0.6181 |
| convnext_tiny_emotion | cnn | mean_embedding_hgb | 50 | 0.4400 | 0.4139 | 0.3622 |
| convnext_tiny_emotion | cnn | mean_embedding_logreg | 50 | 0.4800 | 0.5000 | 0.4331 |
| vit_face_expression | vit | attention_pooling | 50 | 0.6400 | 0.6528 | 0.5972 |
| vit_face_expression | vit | mean_embedding_hgb | 50 | 0.5400 | 0.5472 | 0.4616 |
| vit_face_expression | vit | mean_embedding_logreg | 50 | 0.6200 | 0.6722 | 0.5920 |
