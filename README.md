# Clinical outcome and Deep learning imaging characteristics of patients treated by radio-chemotherapy for a “molecular” glioblastoma
________________
This repository contains the source code associated to the proposed paper: **Clinical outcome and Deep learning imaging characteristics of patients treated by radio-chemotherapy for a “molecular” glioblastoma**.

To use this repository, it is highly recommended to create a dedicated python 3.10 environment.

For instance, in you may run in your terminal:

```
$ conda create python=3.10 --name gliomol
$ conda activate gliomol
```

Then, to install all requirements:
````
$ cd gliomol/
$ pip install -r requirements.txt
````

## Preprocessing
______

For pre-processing, we use the [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit).

Your data must be organized as follows:
````
data/
| - LGG/
|   - patient_1/
|       - flair.nii.gz
|   - patient_2/
|       - flair.nii.gz
| - mGB/
|   - patient_1/
|       - flair.nii.gz
...
````

Then use the following script:

````
$ python registered.py
````

## Dataset creation
_______
To create segmentation mask you can either use the BraTS Toolkit Segmentor or use your own segmenter train on FLAIR data only.

Here we choose to use our own model, available at `pretrained_encoders/segmenter.pth`

To extract segmentation mask manually as well as radiomics with [pyRadiomics](https://pyradiomics.readthedocs.io/en/latest/); please refer to the jupyter notebook `mask_and_radiomics_extraction.ipynb`

This yields a `radiomics.csv`file containing all the necessary information for radiomics modeling.

Finally, to create the dataframe for Deep Learning experiments containing the label, patient ID, path to the FLAIR sequence, and path to the tensor containing the tumor area (64 x 64 x 64), simply enter the following in the CLI:

`````
$ python create_data.py
`````

## Modeling
_______
### Radiomics-based
Once you have your `radiomics.csv` file, refer to the `radiomics_modeling.ipynb` notebook to train the radiomics models.


### Deep Learning
To conduct deep learning experiments, you can use the train.py file. Adjust the parser arguments according to the desired experiment. In particular, use the frozen argument to freeze weights and the tumor argument to choose between training on the entire MRI or focusing on the tumor area.

````
$ python train.py --tumor False
````

## Logging
_____
All the useful and necessary information for the trainings and the realization of the paper have been logged directly on [Weights and Biases](https://wandb.ai/). Therefore, it is also possible to use wandb as a logger with this repository. To do so, you have to add your entity name in the training command line `--entity`.