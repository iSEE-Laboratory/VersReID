# VersReID
Official implementation of Paper ''A Versatile Framework for Multi-scene Person Re-identification''.

![](./assets/main_figure.png)
## News
2024/3/19: Our arxiv paper can be found [here](https://arxiv.org/abs/2403.11121)

2024/3/16: Code and Pre-trained model are released. Check [Baidu Drive](https://pan.baidu.com/s/1TopJ37U9ZlmxQ2-HyP9kqw?pwd=pami) for the pre-trained model (key: pami)


## Datasets
Please visit the following link to download the dataset.
- [Market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501/data) 
- [MSMT17](http://www.pkuvmc.com/dataset.html) 
- [MLR-CUHK03](https://pan.baidu.com/s/1hMQZq0LAPhIl5RQ_EDDiFg?pwd=pami), key: pami
- [Cele-ReID](https://github.com/Huang-3/Celeb-reID)
- [PRCC](https://www.isee-ai.cn/%7Eyangqize/clothing.html)
- [Occ-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset)
- [SYSU-mm01](https://www.isee-ai.cn/project/RGBIRReID.html)

Once you download the datasets, make sure to modify the dataset's root manually [here](https://github.com/iSEE-Laboratory/VersReID/blob/main/configs/ReID-Bank.yml#L55) and [here](https://github.com/iSEE-Laboratory/VersReID/blob/main/configs/V-Branch.yml#L58). 

## Environments
Please follow [TransReID](https://github.com/damo-cv/TransReID) to configure the running environment. 

We provide our used package list in [full-environment.txt](https://github.com/iSEE-Laboratory/VersReID/blob/main/full-environment.txt) for a reference.


## Run
- Download the pre-trained model at [Baidu Drive](https://pan.baidu.com/s/1TopJ37U9ZlmxQ2-HyP9kqw?pwd=pami), the key is pami. 

- Create a directory named ckpts and then put the downloaded model into it, or you can modify the MODEL.PRETRAIN_PATH in ```./bash/run_VersReID.sh``` to your own pre-trained model path.

- To reproduce the results in our paper, just simply run this script:
```
bash ./bash/run_VersReID.sh
```
You can modify the configs by yourself to explore more settings.

__Note__: Training the ReID-Bank requires ~20G GPU Memory, V-Branch requires ~30G GPU Memory. We highly recommend use cuda 10.2 for better reproducibility. Moreover, the code does not support multi-GPU training currently. 

If you have any problem, feel free to open an issue or contact me :-)

## Acknowledgement
- This repository is heavily based on [TransReID](https://github.com/damo-cv/TransReID), many thanks to the authors.

- If you find this repo helpful, please consider citing us:
```
@article{zheng2024versreid,
  title = {A Versatile Framework for Multi-scene Person Re-identification},
  author = {Zheng, Wei-Shi and Yan, Junkai and Peng, Yi-Xing},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2024}
}
```
