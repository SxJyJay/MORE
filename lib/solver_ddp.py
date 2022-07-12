'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.distributed as dist

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.loss_helper import get_scene_cap_loss
from lib.eval_helper import eval_cap
from utils.eta import decode_eta
from lib.pointnet2.pytorch_utils import BNMomentumScheduler


ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_cap_loss: {train_cap_loss}
[loss] train_ori_loss: {train_ori_loss}
[loss] train_dist_loss: {train_dist_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_vote_loss: {train_vote_loss}
[loss] train_box_loss: {train_box_loss}
[sco.] train_cap_acc: {train_cap_acc}
[sco.] train_ori_acc: {train_ori_acc}
[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_pred_ious: {train_pred_ious}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

# EPOCH_REPORT_TEMPLATE = """
# ---------------------------------summary---------------------------------
# [train] train_bleu-1: {train_bleu_1}
# [train] train_bleu-2: {train_bleu_2}
# [train] train_bleu-3: {train_bleu_3}
# [train] train_bleu-4: {train_bleu_4}
# [train] train_cider: {train_cider}
# [train] train_rouge: {train_rouge}
# [train] train_meteor: {train_meteor}
# [val]   val_bleu-1: {val_bleu_1}
# [val]   val_bleu-2: {val_bleu_2}
# [val]   val_bleu-3: {val_bleu_3}
# [val]   val_bleu-4: {val_bleu_4}
# [val]   val_cider: {val_cider}
# [val]   val_rouge: {val_rouge}
# [val]   val_meteor: {val_meteor}
# """

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[val]   val_bleu-1: {val_bleu_1}
[val]   val_bleu-2: {val_bleu_2}
[val]   val_bleu-3: {val_bleu_3}
[val]   val_bleu-4: {val_bleu_4}
[val]   val_cider: {val_cider}
[val]   val_rouge: {val_rouge}
[val]   val_meteor: {val_meteor}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[best]  bleu-1: {bleu_1}
[best]  bleu-2: {bleu_2}
[best]  bleu-3: {bleu_3}
[best]  bleu-4: {bleu_4}
[best]  cider: {cider}
[best]  rouge: {rouge}
[best]  meteor: {meteor}
"""

class Solver():
    def __init__(self, model, device, config, dataset, dataloader, optimizer, stamp, val_step=10, 
    detection=True, caption=True, orientation=False, distance=False, use_tf=True,
    lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None,
    criterion="meteor", checkpoint_best=None):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.device = device
        self.config = config
        self.dataset = dataset
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step

        self.detection = detection
        self.caption = caption
        self.orientation = orientation
        self.distance = distance
        self.use_tf = use_tf

        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.criterion = criterion
        self.checkpoint_best = checkpoint_best

        self.best = {
            "epoch": 0,
            "bleu-1": -float("inf"),
            "bleu-2": -float("inf"),
            "bleu-3": -float("inf"),
            "bleu-4": -float("inf"),
            "cider": -float("inf"),
            "rouge": -float("inf"),
            "meteor": -float("inf"),
            "sum": -float("inf")
        } if checkpoint_best == None else checkpoint_best

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }
        
        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate**(int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = (len(self.dataloader["eval"]["train"]) + len(self.dataloader["eval"]["val"])) \
             * (self._total_iter["train"] / self.val_step)
        
        for epoch_id in range(epoch):
            try:
                if dist.get_rank() == 0:
                    self._log("epoch {} starting...".format(epoch_id + 1))

                # feed
                self.dataloader["train"].sampler.set_epoch(epoch) 
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                if dist.get_rank() == 0 and epoch_id > 0:
                    self._log("saving epoch_{} models...\n".format(epoch_id))
                    model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                    torch.save(self.model.module.state_dict(), os.path.join(model_root, "epoch_{}_model.pth".format(epoch_id)))
                    

                # update lr scheduler
                if self.lr_scheduler:
                    if dist.get_rank() == 0:
                        print("update learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()))
                    self.lr_scheduler.step()

                # update bn scheduler
                if self.bn_scheduler:
                    if dist.get_rank() == 0:
                        print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _reduce_tensor(inp, world_size, average=False):
        """
        Reduce the loss from all the process so 
        that process with rank 0 has average result.
        """
        with torch.no_grad():
            reduced_inp = inp
            dist.reduce(reduced_inp, dst=0)
            if average: 
                reduced_inp = reduced_inp / world_size
        return reduced_inp

    def _reset_log(self, phase):
        if phase == "train":
            self.log[phase] = {
                # info
                "forward": [],
                "backward": [],
                "eval": [],
                "fetch": [],
                "iter_time": [],
                # loss (float, not torch.cuda.FloatTensor)
                "loss": [],
                "cap_loss": [],
                "ori_loss": [],
                "dist_loss": [],
                "objectness_loss": [],
                "vote_loss": [],
                "box_loss": [],
                # scores (float, not torch.cuda.FloatTensor)
                "lang_acc": [],
                "cap_acc": [],
                "ori_acc": [],
                "obj_acc": [],
                "pred_ious": [],
                "neg_ratio": [],
                "pos_ratio": [],
                "neg_ratio": [],
                # for eval
                "bleu-1": [],
                "bleu-2": [],
                "bleu-3": [],
                "bleu-4": [],
                "cider": [],
                "rouge": [],
                "meteor": []
            }
        else:
            self.log[phase] = {
                "bleu-1": [],
                "bleu-2": [],
                "bleu-3": [],
                "bleu-4": [],
                "cider": [],
                "rouge": [],
                "meteor": []
            }

    def _dump_log(self, phase, is_eval=False):
        if phase == "train" and not is_eval:
            log = {
                "loss": ["loss", "cap_loss", "ori_loss", "dist_loss", "objectness_loss", "vote_loss", "box_loss"],
                "score": ["lang_acc", "cap_acc", "ori_acc", "obj_acc", "pred_ious", "pos_ratio", "neg_ratio"]
            }
            for key in log:
                for item in log[key]:
                    if self.log[phase][item]:
                        self._log_writer[phase].add_scalar(
                            "{}/{}".format(key, item),
                            np.mean([v for v in self.log[phase][item]]),
                            self._global_iter_id
                        )

        # eval
        if is_eval:
            log = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "cider", "rouge", "meteor"]
            for key in log:
                if self.log[phase][key]:
                    self._log_writer[phase].add_scalar(
                        "eval/{}".format(key),
                        self.log[phase][key],
                        self._global_iter_id
                    )

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict, use_tf=self.use_tf)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        data_dict = get_scene_cap_loss(
            data_dict=data_dict, 
            device=self.device,
            config=self.config, 
            weights=self.dataset["train"].weights,
            detection=self.detection,
            caption=self.caption,
            orientation=self.orientation,
            distance=self.distance
        )

        # store loss
        self._running_log["cap_loss"] = data_dict["cap_loss"]
        self._running_log["ori_loss"] = data_dict["ori_loss"]
        self._running_log["dist_loss"] = data_dict["dist_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["loss"] = data_dict["loss"]

        # store eval
        self._running_log["cap_acc"] = data_dict["cap_acc"].item()
        self._running_log["ori_acc"] = data_dict["ori_acc"].item()
        self._running_log["pred_ious"] = data_dict["pred_ious"].item()
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()


    def _eval(self, phase):
        if self.caption:
            bleu, cider, rouge, meteor = eval_cap(
                model=self.model,
                device=self.device,
                dataset=self.dataset["eval"][phase],
                dataloader=self.dataloader["eval"][phase],
                phase=phase,
                folder=self.stamp,
                use_tf=False,
                max_len=CONF.TRAIN.MAX_DES_LEN,
                force=True,
                min_iou=CONF.EVAL.MIN_IOU_THRESHOLD
            )

            # dump
            self.log[phase]["bleu-1"] = bleu[0][0]
            self.log[phase]["bleu-2"] = bleu[0][1]
            self.log[phase]["bleu-3"] = bleu[0][2]
            self.log[phase]["bleu-4"] = bleu[0][3]
            self.log[phase]["cider"] = cider[0]
            self.log[phase]["rouge"] = rouge[0]
            self.log[phase]["meteor"] = meteor[0]
        else:
            self.log[phase]["bleu-1"] = 0
            self.log[phase]["bleu-2"] = 0
            self.log[phase]["bleu-3"] = 0
            self.log[phase]["bleu-4"] = 0
            self.log[phase]["cider"] = 0
            self.log[phase]["rouge"] = 0
            self.log[phase]["meteor"] = 0

    def _feed(self, dataloader, phase, epoch_id, is_eval=False):
        # switch mode
        if is_eval:
            self._set_phase("val") 
        else:   
            self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        # enter mode
        if not is_eval:
            for data_dict in dataloader:
                # move to cuda
                for key in data_dict:
                    # data_dict[key] = data_dict[key].cuda()
                    data_dict[key] = data_dict[key].to(self.device)

                # initialize the running loss
                self._running_log = {
                    # loss
                    "loss": 0,
                    "cap_loss": 0,
                    "ori_loss": 0,
                    "dist_loss": 0,
                    "objectness_loss": 0,
                    "vote_loss": 0,
                    "box_loss": 0,
                    # acc
                    "lang_acc": 0,
                    "cap_acc": 0,
                    "ori_acc": 0,
                    "obj_acc": 0,
                    "pred_ious": 0,
                    "pos_ratio": 0,
                    "neg_ratio": 0,
                }

                # load
                self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

                # forward
                start = time.time()
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                start = time.time()
                self._backward()
                self.log[phase]["backward"].append(time.time() - start)
                
                # eval
                start = time.time()
                # self._eval(data_dict)
                self.log[phase]["eval"].append(time.time() - start)

                # record log
                self.log[phase]["loss"].append(self._running_log["loss"].item())
                self.log[phase]["cap_loss"].append(self._running_log["cap_loss"].item())
                self.log[phase]["ori_loss"].append(self._running_log["ori_loss"].item())
                self.log[phase]["dist_loss"].append(self._running_log["dist_loss"].item())
                self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
                self.log[phase]["vote_loss"].append(self._running_log["vote_loss"].item())
                self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())

                self.log[phase]["lang_acc"].append(self._running_log["lang_acc"])
                self.log[phase]["cap_acc"].append(self._running_log["cap_acc"])
                self.log[phase]["ori_acc"].append(self._running_log["ori_acc"])
                self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
                self.log[phase]["pred_ious"].append(self._running_log["pred_ious"])
                self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
                self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])             

                # report
                if phase == "train":
                    iter_time = self.log[phase]["fetch"][-1]
                    iter_time += self.log[phase]["forward"][-1]
                    iter_time += self.log[phase]["backward"][-1]
                    iter_time += self.log[phase]["eval"][-1]
                    self.log[phase]["iter_time"].append(iter_time)
                    if (self._global_iter_id + 1) % self.verbose == 0:
                        self._train_report(epoch_id)

                    # evaluation
                    # if (self._global_iter_id+1)>=2000 and (self._global_iter_id+1) % self.val_step == 0 and dist.get_rank() == 0:
                    if self._global_iter_id % self.val_step == 0 and dist.get_rank() == 0:
                        # save model according different steps
                        # if dist.get_rank() == 0:
                        #     self._log("saving {}_{} models...\n".format(epoch_id, self._global_iter_id))
                        #     model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                        #     torch.save(self.model.module.state_dict(), os.path.join(model_root, "epoch_{}_iter_{}_model.pth".format(epoch_id,self._global_iter_id + 1)))
                        # print("evaluating on train...")
                        # self._feed(self.dataloader["eval"]["train"], "train", epoch_id, True)
                        # self._dump_log("train", True)
                        
                        # val
                        print("evaluating on val...")
                        self._feed(self.dataloader["eval"]["val"], "val", epoch_id, True)
                        self._dump_log("val", True)


                        self._set_phase("train")
                        self._epoch_report(epoch_id)

                    # dump log
                    if self._global_iter_id != 0: self._dump_log("train")
                    self._global_iter_id += 1
        else:
            self._eval(phase)

            cur_criterion = self.criterion
            if cur_criterion == "sum":
                metrics = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "cider", "rouge", "meteor"]
                cur_best = np.sum([np.mean(self.log[phase][m]) for m in metrics])
            else:
                cur_best = np.mean(self.log[phase][cur_criterion])
            
            if phase == "val" and cur_best > self.best[cur_criterion]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))

                self.best["epoch"] = epoch_id + 1
                self.best["bleu-1"] = self.log[phase]["bleu-1"]
                self.best["bleu-2"] = self.log[phase]["bleu-2"]
                self.best["bleu-3"] = self.log[phase]["bleu-3"]
                self.best["bleu-4"] = self.log[phase]["bleu-4"]
                self.best["cider"] = self.log[phase]["cider"]
                self.best["rouge"] = self.log[phase]["rouge"]
                self.best["meteor"] = self.log[phase]["meteor"]
                self.best["sum"] = cur_best

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.module.state_dict(), os.path.join(model_root, "model.pth"))

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best": self.best
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        if dist.get_rank() == 0:
            self._log("saving last models...\n")
            model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            torch.save(self.model.module.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        
        num_train_iter_left = self._total_iter["train"] - self._global_iter_id - 1
        eta_sec = num_train_iter_left * mean_train_time
        
        num_val_times = num_train_iter_left // self.val_step
        eta_sec += len(self.dataloader["eval"]["train"]) * num_val_times * mean_est_val_time
        eta_sec += len(self.dataloader["eval"]["val"]) * num_val_times * mean_est_val_time
        
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_cap_loss=round(np.mean([v for v in self.log["train"]["cap_loss"]]), 5),
            train_ori_loss=round(np.mean([v for v in self.log["train"]["ori_loss"]]), 5),
            train_dist_loss=round(np.mean([v for v in self.log["train"]["dist_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_cap_acc=round(np.mean([v for v in self.log["train"]["cap_acc"]]), 5),
            train_ori_acc=round(np.mean([v for v in self.log["train"]["ori_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_pred_ious=round(np.mean([v for v in self.log["train"]["pred_ious"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        if dist.get_rank() == 0:
            self._log(iter_report)

    def _epoch_report(self, epoch_id):
        if dist.get_rank() == 0:
            self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            # train_bleu_1=round(self.log["train"]["bleu-1"], 5),
            # train_bleu_2=round(self.log["train"]["bleu-2"], 5),
            # train_bleu_3=round(self.log["train"]["bleu-3"], 5),
            # train_bleu_4=round(self.log["train"]["bleu-4"], 5),
            # train_cider=round(self.log["train"]["cider"], 5),
            # train_rouge=round(self.log["train"]["rouge"], 5),
            # train_meteor=round(self.log["train"]["meteor"], 5),
            val_bleu_1=round(self.log["val"]["bleu-1"], 5),
            val_bleu_2=round(self.log["val"]["bleu-2"], 5),
            val_bleu_3=round(self.log["val"]["bleu-3"], 5),
            val_bleu_4=round(self.log["val"]["bleu-4"], 5),
            val_cider=round(self.log["val"]["cider"], 5),
            val_rouge=round(self.log["val"]["rouge"], 5),
            val_meteor=round(self.log["val"]["meteor"], 5)
        )
        if dist.get_rank() == 0:
            self._log(epoch_report)
    
    def _best_report(self):
        if dist.get_rank() == 0:
            self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            bleu_1=round(self.best["bleu-1"], 5),
            bleu_2=round(self.best["bleu-2"], 5),
            bleu_3=round(self.best["bleu-3"], 5),
            bleu_4=round(self.best["bleu-4"], 5),
            cider=round(self.best["cider"], 5),
            rouge=round(self.best["rouge"], 5),
            meteor=round(self.best["meteor"], 5)
        )
        if dist.get_rank() == 0:
            self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
