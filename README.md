# VisRec: A Semi-Supervised Approach to Radio Interferometric Data Reconstruction (AAAI-25)

### Abstract

Radio telescopes produce visibility data about celestial objects, but these data are sparse and noisy. As a result, images created on raw visibility data are of low quality. Recent studies have used deep learning models to reconstruct visibility data to get cleaner images. However, these methods rely on a substantial amount of labeled training data, which requires significant labeling effort from radio astronomers. Addressing this challenge, we propose VisRec, a model-agnostic semi-supervised learning approach to the reconstruction of visibility data. Specifically, VisRec consists of both a supervised learning module and an unsupervised learning module. In the supervised learning module, we introduce a set of data augmentation functions to produce diverse training examples. In comparison, the unsupervised learning module in VisRec augments unlabeled data and uses reconstructions from non-augmented visibility data as pseudo-labels for training. This hybrid approach allows VisRec to effectively leverage both labeled and unlabeled data. This way, VisRec performs well even when labeled data is scarce. Our evaluation results show that VisRec outperforms all baseline methods in reconstruction quality, robustness against common observation perturbation, and generalizability to different telescope configurations.

---

### Dataset

To run the demos, please download public dataset presented by Wu et al.[1]. 

Then modify the related path in the code.

## NF model

#### Setup the conda environment

Set up the conda environment using the `requirements.txt` file.


#### Modify the model, dataset path within the bash script.

#### Train Model

Run the `train_model.sh` script from command line:

```
sh ./train_model.sh
```

#### Inference using the trained model

Modify the paths in `eval_model.sh` script.

Run the `eval_model.sh` script from command line:

```
sh ./eval_model.sh
```

The results will be saved in the `'../test_res1'` folder, including visibility reconstruction and resultant image.

Evaluate the results with SSIM, PSNR, LFD:

```
python metrics.py
```

---

## CNN model

#### Modify the model, datapath path parameter within the bash script.

#### Train Model

Modify the paths in `train_model.sh` script. Also, please modify the paths in `galaxy.py` `dataset_setting.py` and `myfft.py`

Run the `train_model.sh` script from command line:

```
sh ./train_model.sh
```

#### Inference using the trained model

Modify the paths in `eval_model.sh` script.

Run the `eval_model.sh` script from command line:

```
sh ./eval_model.sh
```

Evaluate the results with SSIM, PSNR, LFD:

```
python metrics.py
```

---

## Reference

[1] Wu, Benjamin, et al. "Neural Interferometry: Image Reconstruction from Astronomical Interferometers using Transformer-Conditioned Neural Fields." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 36. No. 3. 2022.