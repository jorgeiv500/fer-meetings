| model | family | scope | method | n_clips | accuracy | balanced_accuracy | macro_f1 | auroc_ovr | auprc_macro | brier_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_tiny_emotion | cnn | test | single_frame | 50 | 0.2600 | 0.3889 | 0.1959 | 0.6870 | 0.5173 | 0.3487 |
| convnext_tiny_emotion | cnn | test | smoothed | 50 | 0.2200 | 0.3667 | 0.1576 | 0.7019 | 0.5398 | 0.3625 |
| convnext_tiny_emotion | cnn | test | smoothed_calibrated | 50 | 0.5200 | 0.5028 | 0.3723 | 0.7914 | 0.5937 | 0.1932 |
| convnext_tiny_emotion | cnn | test | vote | 50 | 0.2400 | 0.3778 | 0.1772 |  |  |  |
| vit_face_expression | vit | test | single_frame | 50 | 0.6600 | 0.4472 | 0.4552 | 0.6795 | 0.5810 | 0.1767 |
| vit_face_expression | vit | test | smoothed | 50 | 0.6400 | 0.4056 | 0.3911 | 0.6947 | 0.5981 | 0.1671 |
| vit_face_expression | vit | test | smoothed_calibrated | 50 | 0.5800 | 0.4333 | 0.4479 | 0.6834 | 0.5960 | 0.1874 |
| vit_face_expression | vit | test | vote | 50 | 0.6400 | 0.4361 | 0.4439 |  |  |  |
| cnn_vit_entropy_ensemble | hybrid | test | single_frame | 50 | 0.5400 | 0.5139 | 0.3860 | 0.7767 | 0.6119 | 0.2010 |
| cnn_vit_entropy_ensemble | hybrid | test | smoothed | 50 | 0.5600 | 0.5556 | 0.4070 | 0.7778 | 0.6065 | 0.2155 |
| cnn_vit_mean_ensemble | hybrid | test | single_frame | 50 | 0.5400 | 0.5139 | 0.3860 | 0.8012 | 0.6700 | 0.1966 |
| cnn_vit_mean_ensemble | hybrid | test | smoothed | 50 | 0.5600 | 0.5556 | 0.4070 | 0.8061 | 0.6621 | 0.1999 |
