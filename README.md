# MORE
Official code implementation of ECCV 2022 paper: ["MORE: Multi-Order RElation Mining for Dense Captioning in 3D Scenes"](https://arxiv.org/abs/2203.05203). 

This paper aims at progressively encode first- and multi-order spatial relations within graphs in a recently proposed novel task -- 3D dense captioning
![framework of MORE](/Images/MORE.png)

## Data and Setup steps
We suggest readers refer to data and setup preparation steps of [Scan2Cap](https://github.com/daveredrum/Scan2Cap).

## Training and Evaluation
We provide training scripts on multi-GPUs, which can facilitate users to train our model faster. To train MORE with different painted point features (e.g., xyz, xyz+normal, xyz+normal+rgb, xyz+normal+multiview), please run
```
CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $num_of_gpus --master_port $port_id scripts/train_ddp.py --use_color --use_normal --num_graph_steps 1 --graph_module SLGC --graph_mode spatial_layout_conv --stamp MORE_normal_rgb --decoder_module OTAG --lr 1e-4
```
The above command gives an example of using ```xyz+normal+rgb``` as input point features. For using other settings of point features, please flexibly add or remove args including ```--use_normal, --use_color, --use_rgb```. Please change the ```--num_graph_steps``` to 2 when using multiview features as:
```
CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $num_of_gpus --master_port $port_id scripts/train_ddp.py --use_multiview --use_normal --num_graph_steps 2 --graph_module SLGC --graph_mode spatial_layout_conv --stamp MORE_normal_multiview --decoder_module OTAG --lr 1e-4
```
To evaluate the caption performances, please run
```
 python scripts/eval.py --folder $ckpt_folder_name --use_color --use_normal --num_graph_steps 1 --num_locals 10 --eval_caption --min_iou 0.5 --graph_module SLGC --graph_mode spatial_layout_conv --decoder_module OTAG
```
Note that arguments must match ones for training.

## Performances
|  model   | C@0.5IoU  | B-4@0.5IoU | M@0.5IoU | R@0.5IoU
|  :----:  | :----:  |  :----:  |  :----:  |  :----:  |
| MORE<sub>rgb  | 38.98 | 23.01 | 21.65 | 44.33 |
| MORE<sub>mul  | 40.94 | 22.93 | 21.66 | 44.42 |

Since the caption metrics are not stable, we recommend you to save the checkpoints at every epoch and load the checkpoint with higher validation performance during testing for better performance.

## Citation
If you found our project helpful, please kindly cite our paper via:
```
@article{jiao2022more,
  title={MORE: Multi-Order RElation Mining for Dense Captioning in 3D Scenes},
  author={Jiao, Yang and Chen, Shaoxiang and Jie, Zequn and Chen, Jingjing and Ma, Lin and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2203.05203},
  year={2022}
}
```
## Acknowledgement
We sincerely thank the authors of [Scan2Cap](https://github.com/daveredrum/Scan2Cap) for open sourcing their data and code. Part of the code in our project are from Scan2Cap.
