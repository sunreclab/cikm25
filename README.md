# Dual-disentangle Framework for Diversified Sequential  Recommendation

This is the pytorch implementation of our paper at CIKM 2025:  
Dual-disentangle Framework for Diversified Sequential  Recommendation  
Jingtong Liu, Haoran Zhang, Jiangzhou Deng, Junpeng Guo

## Environment Settings
Python version: 3.12.3   
Pytorch version: 2.6.0

## Dataset
The datasets can be downloaded from the following links:  
KuaiRec:  https://kuairec.com/  
Tenrec:  https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html  
MIND:  https://msnews.github.io/

## Usage
### prepare data
Please download raw datasets first and put the downloaded data folder in the 'preprocess/'. Then you can get preprocessed data by using the script  
```
python preprocess_[dataset name].py
```
### train model
```
python main.py  [hyper-para list]
```
