# README

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