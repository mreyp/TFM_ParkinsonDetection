# Voice-Based Parkinson’s Disease Detection through Neural Networks and GAN-Driven Data Augmentation 

This repository is used to store all the files used in the development of the project: **Voice-Based Parkinson’s Disease Detection through Neural Networks and GAN-Driven Data Augmentation**.

This project has been developed as a Master's Thesis by Marta Rey Paredes, for the Master's Degree in Artificial Intelligence (MUIA) at the Polytechnic University of Madrid. The associated document can be found in the [Archivo Digital UPM](https://oa.upm.es). 

The five models presented are copies of existing repositories, modified to adapt them for this project.

The data used for the development of this work is not included in this repository, due to privacy constraints related to the subjects who participated in the study of the donated database. 

The project is divided into six subfolders:

* preparacion: includes the notebooks used for preprocessing the data, extracting visual aids, and performing the 5-fold cross-validation technique.
* BigVSAN: includes the necessary files to train, generate and evaluate the sound files. Pretrained models are also stored. Modified from [https://github.com/sony/bigvsan](https://github.com/sony/bigvsan)
* ResNet. Modified from [https://github.com/hfawaz/dl-4-tsc](https://github.com/hfawaz/dl-4-tsc)
* LSTM_FCN. Modified from [https://github.com/flaviagiammarino/lstm-fcn-pytorch](https://github.com/flaviagiammarino/lstm-fcn-pytorch)
* InceptionTime. Modified from [https://github.com/hfawaz/InceptionTime](https://github.com/hfawaz/InceptionTime)
* CDIL_CNN. Modified from [https://github.com/LeiCheng-no/CDIL-CNN](https://github.com/LeiCheng-no/CDIL-CNN)

## Usage
To use any of the models:

* Download or clone the repository
* Move to the directory of the model you want to use
* Any of the models can be used by simply opening the ```.ipynb```associated file and changing the input/output directories
* You may have to change the input data inside the ```main.py```file of each model


## Acknowledgements
We are thankful to Prof. Orozco Arroyave and GITA research group of the University of Antioquia (Colombia) for allowing to use the PCGita database.

This research is part of the R\&D\&I projects PID2021-122209OB-C31 and PID2021-122209OB-C32 funded by MCIN/AEI/10.13039/501100011033 and ERDF A way of making Europe.

## Citing
The author gives permission for any kind of work in which the code can be used partially or completely in any project that may be necessary, thanking the acknowledgement of her authorship.
