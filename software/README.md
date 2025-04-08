Original github page and code:  https://github.com/uvaydovd/spectrum_sensing_stitching.git

# Envrionment

To generate the signal bank, use the data generator, or train the multilabel model with self-attention used in the paper import this environment from yml:

```
conda env create -n multilabel --file multilabel.yml
conda activate multilabel
```

To evaluate onnx version of model (version we use for real-time inference) import this environment from yml:

```
conda env create -n eval_DL --file eval_DL.yml
conda activate eval_DL
```


# Code

```multilabel.py``` - Use this file to train or test the customized unet multilabel model with self attention

```
usage: multilabel.py [-h] [-m] [-ts] [-vs] [-d]

options:
  -h, --help         show this help message and exit
  -m , --Mode        specify training or testing (default: train)
  -ts , --TrainSet   filepath of training set (default:
                     ./train.h5)
  -vs , --ValSet     filepath of valiation set (default:
                     ./val.h5)
  -d , --Device      specify the gpu device, -1 means cpu (default: -1)

  
example: python multilabel.py -m train -ts train.h5 -vs test.h5 -d 0
```



```eval_DL_onnx.py``` - Use this file to run the model on raw IQs. The code expects the IQs in the form of a binary file, similiar to the output file that GNU Radio generates from a file sink, specifically the binary file contains a series of two 32-bit floating point numbers, one each for the real and imaginary components.

```
usage: eval_DL_onnx.py [-h] [--id_gpu ID_GPU] [--samp_rate SAMP_RATE] [--input INPUT] [--model MODEL] [--normalize NORMALIZE]

GPU and Model Specifications

optional arguments:
  -h, --help                show this help message and exit
  --id_gpu ID_GPU           specify which gpu to use.
  --samp_rate SAMP_RATE     specifies the sampling rate in MHz, for now must be multiple of 25 MHz
  --input INPUT             specifies the IQ samples bin file to be fed to the network
  --model MODEL             specifies the model filepath

example: python eval_DL_onnx.py --id_gpu 0 --samp_rate 25 --input ./test_data.bin --model ./multilabel.onnx
```
