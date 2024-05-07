## Installation

You need to have a Python 3.10 version installed:
```bash
py -V:3.10 -m venv <NAME>
<NAME>/Scripts/activate
```

You need to have Anaconda installed:
```bash
conda create -n nemo python=3.10.11 (>=python38)
conda activate nemo
```

**Requirements**

Conda:
```bash
pytorch
torchvision
torchaudio 
pytorch-cuda=11.8 -c pytorch -c nvidia
```

Pip:
```bash
Cython
nemo_toolkit #`all`, `asr` and `nlp` gave errors
pytorch_lightning
hydra-core
librosa
transformers
sentencepiece
inflect
webdataset
lhotse
pyannote.audio
editdistance
jiwer
cuda-python>=12.3
```

**Notes**
1. The packages `youtokentome` and `pynini` made a lot of problems due to wrong build wheels, compilining errors etc. I had to remove them from the requirements.
2. The version of nemo had some import errors, so I needed to manually change the imports in the code (it's mostly about the `aed_multitask_models.py` file).