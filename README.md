# ANAGEN

Here are the steps of running ANAGEN on the NAS-BENCH-360 DeepSEA task:

1.Install Packages
conda create --name ANAGEN
conda activate ANAGEN
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
pip install scipy sklearn tqdm ml-collections h5py requests
git clone https://github.com/mkhodak/relax relax
cd relax && pip install -e .
pip install ray

2.Download Data
Download the NAS-BENCH-360 DeepSEA data from https://drive.google.com/file/d/1wTCnpF91GYA043w8H14Q8riTLJNUGxGJ/view?usp=share_link

3.Write dataloaders 

4.Run DASH
cd DASH/src
sh run.sh

5.Run ASHA
cd ASHA
python asha.py 

6.Retrain
python pretrain_embedder.py

7.Analysis
cd Analysis
The codes for generating plots are in data_analysis.ipynb and Analysis.ipynb
