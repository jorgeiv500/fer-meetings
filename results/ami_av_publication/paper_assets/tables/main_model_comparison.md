| model | family | scope | method | n_clips | accuracy | balanced_accuracy | macro_f1 | auroc_ovr | auprc_macro | brier_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_tiny_emotion | cnn | all | single_frame | 100 | 0.3400 | 0.3791 | 0.2305 | 0.6483 | 0.4806 | 0.3075 |
| convnext_tiny_emotion | cnn | all | smoothed | 100 | 0.3100 | 0.3595 | 0.1948 | 0.6472 | 0.4978 | 0.3125 |
| convnext_tiny_emotion | cnn | all | vote | 100 | 0.3200 | 0.3660 | 0.2072 |  |  |  |
| convnext_tiny_emotion | cnn | dev | single_frame | 50 | 0.4200 | 0.3651 | 0.2470 | 0.6514 | 0.4860 | 0.2663 |
| convnext_tiny_emotion | cnn | dev | smoothed | 50 | 0.4000 | 0.3492 | 0.2166 | 0.6514 | 0.5159 | 0.2624 |
| convnext_tiny_emotion | cnn | dev | vote | 50 | 0.4000 | 0.3492 | 0.2180 |  |  |  |
| convnext_tiny_emotion | cnn | test | single_frame | 50 | 0.2600 | 0.3889 | 0.1959 | 0.6870 | 0.5173 | 0.3487 |
| convnext_tiny_emotion | cnn | test | smoothed | 50 | 0.2200 | 0.3667 | 0.1576 | 0.7019 | 0.5398 | 0.3625 |
| convnext_tiny_emotion | cnn | test | smoothed_calibrated | 50 | 0.5200 | 0.5028 | 0.3723 | 0.7914 | 0.5937 | 0.1932 |
| convnext_tiny_emotion | cnn | test | vote | 50 | 0.2400 | 0.3778 | 0.1772 |  |  |  |
| vit_face_expression | vit | all | single_frame | 100 | 0.6100 | 0.4794 | 0.4774 | 0.7087 | 0.5881 | 0.1954 |
| vit_face_expression | vit | all | smoothed | 100 | 0.5900 | 0.4547 | 0.4402 | 0.7571 | 0.6463 | 0.1811 |
| vit_face_expression | vit | all | vote | 100 | 0.5800 | 0.4454 | 0.4331 |  |  |  |
| vit_face_expression | vit | dev | single_frame | 50 | 0.5600 | 0.5035 | 0.4790 | 0.7218 | 0.6002 | 0.2141 |
| vit_face_expression | vit | dev | smoothed | 50 | 0.5400 | 0.4860 | 0.4525 | 0.7967 | 0.6798 | 0.1952 |
| vit_face_expression | vit | dev | vote | 50 | 0.5200 | 0.4526 | 0.4081 |  |  |  |
| vit_face_expression | vit | test | single_frame | 50 | 0.6600 | 0.4472 | 0.4552 | 0.6795 | 0.5810 | 0.1767 |
| vit_face_expression | vit | test | smoothed | 50 | 0.6400 | 0.4056 | 0.3911 | 0.6947 | 0.5981 | 0.1671 |
| vit_face_expression | vit | test | smoothed_calibrated | 50 | 0.5800 | 0.4333 | 0.4479 | 0.6834 | 0.5960 | 0.1874 |
| vit_face_expression | vit | test | vote | 50 | 0.6400 | 0.4361 | 0.4439 |  |  |  |
