# Pilot Summary

| model | family | scope | method | n_clips | accuracy | balanced_accuracy | macro_f1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| cnn_vit_entropy_ensemble | hybrid | all | single_frame | 100 | 0.5600 | 0.4996 | 0.4170 |
| cnn_vit_entropy_ensemble | hybrid | all | smoothed | 100 | 0.6300 | 0.5686 | 0.4712 |
| cnn_vit_entropy_ensemble | hybrid | dev | single_frame | 50 | 0.5800 | 0.4871 | 0.4291 |
| cnn_vit_entropy_ensemble | hybrid | dev | smoothed | 50 | 0.7000 | 0.5873 | 0.5200 |
| cnn_vit_entropy_ensemble | hybrid | test | single_frame | 50 | 0.5400 | 0.5139 | 0.3860 |
| cnn_vit_entropy_ensemble | hybrid | test | smoothed | 50 | 0.5600 | 0.5556 | 0.4070 |
| cnn_vit_mean_ensemble | hybrid | all | single_frame | 100 | 0.5600 | 0.5054 | 0.4183 |
| cnn_vit_mean_ensemble | hybrid | all | smoothed | 100 | 0.6400 | 0.5752 | 0.4774 |
| cnn_vit_mean_ensemble | hybrid | dev | single_frame | 50 | 0.5800 | 0.4887 | 0.4274 |
| cnn_vit_mean_ensemble | hybrid | dev | smoothed | 50 | 0.7200 | 0.6032 | 0.5337 |
| cnn_vit_mean_ensemble | hybrid | test | single_frame | 50 | 0.5400 | 0.5139 | 0.3860 |
| cnn_vit_mean_ensemble | hybrid | test | smoothed | 50 | 0.5600 | 0.5556 | 0.4070 |
| convnext_tiny_emotion | cnn | all | single_frame | 100 | 0.3400 | 0.3791 | 0.2305 |
| convnext_tiny_emotion | cnn | all | smoothed | 100 | 0.3100 | 0.3595 | 0.1948 |
| convnext_tiny_emotion | cnn | all | vote | 100 | 0.3200 | 0.3660 | 0.2072 |
| convnext_tiny_emotion | cnn | dev | single_frame | 50 | 0.4200 | 0.3651 | 0.2470 |
| convnext_tiny_emotion | cnn | dev | smoothed | 50 | 0.4000 | 0.3492 | 0.2166 |
| convnext_tiny_emotion | cnn | dev | vote | 50 | 0.4000 | 0.3492 | 0.2180 |
| convnext_tiny_emotion | cnn | test | single_frame | 50 | 0.2600 | 0.3889 | 0.1959 |
| convnext_tiny_emotion | cnn | test | smoothed | 50 | 0.2200 | 0.3667 | 0.1576 |
| convnext_tiny_emotion | cnn | test | smoothed_calibrated | 50 | 0.5200 | 0.5028 | 0.3723 |
| convnext_tiny_emotion | cnn | test | vote | 50 | 0.2400 | 0.3778 | 0.1772 |
| vit_face_expression | vit | all | single_frame | 100 | 0.6100 | 0.4794 | 0.4774 |
| vit_face_expression | vit | all | smoothed | 100 | 0.5900 | 0.4547 | 0.4402 |
| vit_face_expression | vit | all | vote | 100 | 0.5800 | 0.4454 | 0.4331 |
| vit_face_expression | vit | dev | single_frame | 50 | 0.5600 | 0.5035 | 0.4790 |
| vit_face_expression | vit | dev | smoothed | 50 | 0.5400 | 0.4860 | 0.4525 |
| vit_face_expression | vit | dev | vote | 50 | 0.5200 | 0.4526 | 0.4081 |
| vit_face_expression | vit | test | single_frame | 50 | 0.6600 | 0.4472 | 0.4552 |
| vit_face_expression | vit | test | smoothed | 50 | 0.6400 | 0.4056 | 0.3911 |
| vit_face_expression | vit | test | smoothed_calibrated | 50 | 0.5800 | 0.4333 | 0.4479 |
| vit_face_expression | vit | test | vote | 50 | 0.6400 | 0.4361 | 0.4439 |

## Notes

- Calibration for convnext_tiny_emotion used multinomial logistic regression over smoothed 3-class probabilities from split=dev.
- Calibration for vit_face_expression used multinomial logistic regression over smoothed 3-class probabilities from split=dev.
- Hybrid CNN+ViT ensemble rows were added with mean-probability fusion and entropy-weighted fusion over paired clip probabilities.
