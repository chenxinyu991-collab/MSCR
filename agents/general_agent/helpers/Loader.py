import logging
import os

import torch
import torch._dynamo
import torch.nn as nn
import torch.optim as optim
import wandb
from colorama import Fore

from models.MCR_Models import *
from models.SthSth_models import *
# from models.SyntheticData_Model import *
from utils.schedulers.no_scheduler import No_Scheduler
from utils.schedulers.warmup_scheduler import WarmupScheduler

TORCHDYNAMO_VERBOSE = 1
TORCH_LOGS = "+dynamo"

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

logger = logging.getLogger("torch._dynamo.symbolic_convert:")
logger.setLevel(logging.WARNING)

logger = logging.getLogger("torch._dynamo.output_graph:")
logger.setLevel(logging.WARNING)


class Loader:

    def __init__(self, agent):
        self.agent = agent

    def load_pretrained_models(self):
        if "pretrained_model" in self.agent.config.model:
            if self.agent.config.model.pretrained_model["use"] and not self.agent.config.model.load_ongoing:
                if self.agent.accelerator.ismainprocess:
                    self.agent.logger.info("Loading pretrained model from file {}".format(self.agent.config.model.pretrained_model["dir"]))
                checkpoint = torch.load(self.agent.config.model.pretrained_model["dir"], weights_only=False)
                self.agent.model.load_state_dict(checkpoint["model_state_dict"])
                # self.agent.best_model.load_state_dict(checkpoint["best_model_state_dict"])

    def _freeze_encoders(self, config_model, model):
        for enc in range(len(config_model.get("encoders", []))):
            enc_args = config_model.encoders[enc].get("args", {})
            if enc_args.get("freeze_encoder", False):
                if hasattr(model, "enc_{}".format(enc)):
                    self.agent.logger.info("Freezing encoder enc_{}".format(enc))
                    for p in getattr(model, "enc_{}".format(enc)).parameters():
                        p.requires_grad = False
                    for m in getattr(model, "enc_{}".format(enc)).modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.track_running_stats = False

            if "encoders" in config_model.encoders[enc]:
                for enc_i in range(len(config_model.encoders)):
                    self._freeze_encoders(
                        config_model=config_model.encoders[enc_i],
                        model=getattr(model, "enc_{}".format(enc_i)),
                    )

    def keep_only_trainable_params(self, list_of_params):
        params = []
        for p in list_of_params:
            if isinstance(p, list):
                for p_i in p:
                    if p_i.requires_grad:
                        params.append(p_i)
                continue
            else:
                if p.requires_grad:
                    params.append(p)
        return params

    def load_models_n_optimizer(self):

        enc = self.load_encoder(enc_args=self.agent.config.model.get("encoders", []))
        model_class = globals()[self.agent.config.model.model_class]

        if "save_base_dir" in self.agent.config.model and "swin_backbone" in self.agent.config.model.args:
            self.agent.config.model.args.swin_backbone = os.path.join(
                self.agent.config.model.save_base_dir,
                self.agent.config.model.args.swin_backbone,
            )

        if "save_base_dir" in self.agent.config.model and "pretraining_paths" in self.agent.config.model.args:
            self.agent.config.model.args.pretraining_paths = {
                i: os.path.join(
                    self.agent.config.model.save_base_dir,
                    self.agent.config.model.args.pretraining_paths[i],
                )
                for i in self.agent.config.model.args.pretraining_paths
            }

        self.agent.model = model_class(encs=enc, args=self.agent.config.model.args)
        self._freeze_encoders(config_model=self.agent.config.model, model=self.agent.model)
        self.agent.model.cuda()
        print("MODEL STRUCTURE:")
        print(self.agent.model)

        if self.agent.config.model.get("compile_model", False):
            self.agent.model = torch.compile(self.agent.model, backend="eager")

        self._my_numel(self.agent.model, verbose=True)

        if self.agent.config.optimizer.type == "Adam":
            ind_opt = self.agent.config.optimizer.get("indepentent_params", False)
            all_param_names = [name for name, _ in self.agent.model.named_parameters()]
            if ind_opt is not False:
                list_of_params = []
                for i, key in enumerate(ind_opt):
                    if hasattr(self.agent.model, key):
                        this_param_names = ["{}.{}".format(key, name) for name, _ in getattr(self.agent.model, key).named_parameters()]
                        all_param_names = [name for name in all_param_names if name not in this_param_names]
                        list_of_params.append(
                            {
                                "params": getattr(self.agent.model, key).parameters(),
                                "lr": ind_opt[key]["learning_rate"],
                                "weight_decay": ind_opt[key]["weight_decay"],
                            }
                        )
                list_of_params.append(
                    {
                        "params": [p for n_i in all_param_names for name, p in self.agent.model.named_parameters() if name == n_i],
                        "lr": self.agent.config.optimizer.learning_rate,
                        "weight_decay": self.agent.config.optimizer.weight_decay,
                    }
                )
                list_of_params = self.keep_only_trainable_params(list_of_params)
                self.agent.optimizer = optim.Adam(
                    list_of_params,
                    lr=self.agent.config.optimizer.learning_rate,
                    betas=(
                        self.agent.config.optimizer.beta1,
                        self.agent.config.optimizer.beta2,
                    ),
                    eps=1e-07,
                    weight_decay=self.agent.config.optimizer.weight_decay,
                )
            else:
                list_of_params = self.keep_only_trainable_params(self.agent.model.parameters())
                self.agent.optimizer = optim.Adam(
                    list_of_params,
                    lr=self.agent.config.optimizer.learning_rate,
                    betas=(
                        self.agent.config.optimizer.beta1,
                        self.agent.config.optimizer.beta2,
                    ),
                    eps=1e-07,
                    weight_decay=self.agent.config.optimizer.weight_decay,
                )
        elif self.agent.config.optimizer.type == "SGD":
            self.agent.optimizer = optim.SGD(
                self.agent.model.parameters(),
                lr=self.agent.config.optimizer.learning_rate,
                weight_decay=self.agent.config.optimizer.weight_decay,
                momentum=self.agent.config.optimizer.momentum,
            )
        elif self.agent.config.optimizer.type == "Adadelta":
            self.agent.optimizer = optim.Adadelta(
                self.agent.model.parameters(),
                lr=self.agent.config.optimizer.learning_rate,
                rho=0.9,
                eps=1e-06,
                weight_decay=self.agent.config.optimizer.weight_decay,
            )
        elif self.agent.config.optimizer.type == "Adaw":
            ind_opt = self.agent.config.optimizer.get("indepentent_params", False)
            if ind_opt is not False:
                list_of_params = []
                for i, key in enumerate(ind_opt):
                    name = "mod{}_{}_model".format(i, key)
                    if hasattr(self.agent.model, name):
                        list_of_params.append(
                            {
                                "params": getattr(self.agent.model, name).parameters(),
                                "lr": ind_opt[key]["learning_rate"],
                                "weight_decay": ind_opt[key]["weight_decay"],
                            }
                        )
                if hasattr(self.agent.model, "classifier"):
                    list_of_params.append(
                        {
                            "params": self.agent.model.classifier.parameters(),
                            "lr": self.agent.config.optimizer.learning_rate,
                            "weight_decay": self.agent.config.optimizer.weight_decay,
                        }
                    )
                self.agent.optimizer = optim.AdamW(
                    list_of_params,
                    lr=self.agent.config.optimizer.learning_rate,
                    weight_decay=self.agent.config.optimizer.weight_decay,
                )
            else:
                self.agent.optimizer = optim.AdamW(
                    self.agent.model.parameters(),
                    lr=self.agent.config.optimizer.learning_rate,
                    weight_decay=self.agent.config.optimizer.weight_decay,
                )
        self.load_pretrained_models()

    def load_best_model(self):

        file_name = self.agent.config.model.save_dir
        if "data_split" in self.agent.config.dataset and self.agent.config.dataset.data_split.get("split_method", False) == "patients_folds":
            file_name = file_name.format(self.agent.config.dataset.data_split.fold)

        if "save_base_dir" in self.agent.config.model:
            file_name = os.path.join(self.agent.config.model.save_base_dir, file_name)

        if os.path.exists(file_name):
            prev_checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)
            if "best_model_state_dict" in prev_checkpoint:
                prev_checkpoint["best_model_state_dict"] = {key.replace("module.", ""): value for key, value in prev_checkpoint["best_model_state_dict"].items()}

                self.agent.model.load_state_dict(prev_checkpoint["best_model_state_dict"])
                logging.info("Loaded best model from {}".format(file_name))
            else:
                logging.info("No best model found in {}".format(file_name))
        else:
            logging.info("No file found in {}".format(file_name))

    def _my_numel(self, m: torch.nn.Module, only_trainable: bool = False, verbose=True):

        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        model_total_params = sum(p.numel() for p in unique)
        if verbose and self.agent.accelerator.is_main_process:
            self.agent.logger.info("Total number of trainable parameters are: {}".format(model_total_params))

        return model_total_params

    
    def get_scheduler(self):
        
        # 简化引用
        config = self.agent.config
        optimizer = self.agent.optimizer
        
        # ============================================================
        # 1. CyclicLR
        # ============================================================
        if config.scheduler.type == "cyclic":
            after_scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=config.optimizer.learning_rate,
                max_lr=config.scheduler.max_lr,
                cycle_momentum=False,
            )
            self.agent.scheduler = WarmupScheduler(
                optimizer=optimizer,
                base_lr=config.optimizer.learning_rate,
                n_warmup_steps=config.scheduler.warm_up_steps,
                after_scheduler=after_scheduler,
            )

        # ============================================================
        # 2. Cosine Annealing (核心修改部分)
        # ============================================================
        elif config.scheduler.type == "cosanneal":
            
            # --- [Part A] 获取每个 Epoch 的步数 (Batch Count) ---
            # 自动探测 AVE_Dataloader 的内部属性
            dl_obj = self.agent.data_loader
            steps_per_epoch = None
            
            possible_names = ['train_loader', 'train_data_loader', 'train_dl']
            for name in possible_names:
                if hasattr(dl_obj, name):
                    steps_per_epoch = len(getattr(dl_obj, name))
                    break
            
            if steps_per_epoch is None:
                # 尝试直接 len(对象)
                try:
                    steps_per_epoch = len(dl_obj)
                except:
                    print("\n[DEBUG ERROR] 无法找到 train_loader。dl_obj 的属性有:", dir(dl_obj))
                    raise AttributeError("无法在 Data Loader 中找到 train_loader，请检查代码。")

            # --- [Part B] 获取总 Epoch 数 (全配置文件适配逻辑) ---
            total_epochs = None

            # 优先级 1: 命令行传入的 config.epochs (最高优先级)
            if hasattr(config, 'epochs') and config.epochs is not None:
                total_epochs = config.epochs
            
            # 优先级 2: Early Stopping 中的 max_epoch (AVE Config 用这个)
            elif hasattr(config, 'early_stopping') and hasattr(config.early_stopping, 'max_epoch'):
                total_epochs = config.early_stopping.max_epoch
            
            # 优先级 3: MCR/Collab 特有的 ending_epoch (MCR Config 用这个)
            # 路径: model -> args -> bias_infusion -> ending_epoch
            elif (hasattr(config, 'model') and 
                  hasattr(config.model, 'args') and 
                  hasattr(config.model.args, 'bias_infusion') and 
                  hasattr(config.model.args.bias_infusion, 'ending_epoch')):
                total_epochs = config.model.args.bias_infusion.ending_epoch

            # 优先级 4: Model Args 中的 epochs
            elif hasattr(config, 'model') and hasattr(config.model, 'args') and hasattr(config.model.args, 'epochs'):
                total_epochs = config.model.args.epochs

            # 保底默认值
            else:
                print("[Warning] 未能在 Config 中找到 Epoch 设置，默认使用 50")
                total_epochs = 50 

            # --- [Part C] 最终检查 & 计算 ---
            # 如果是 Unimodal 配置里的 1000，这里给个提示
            if total_epochs > 100:
                print(f"[Info] 检测到 Total Epochs = {total_epochs}。如果是意料之外的（如配置文件写死1000），请在命令行用 --epochs 50 覆盖。")

            total_steps = total_epochs * steps_per_epoch
            warmup_steps = config.scheduler.warm_up_steps
            decay_steps = total_steps - warmup_steps
            
            if decay_steps <= 0:
                decay_steps = 1 # 防止报错

            # --- [Part D] 定义 Scheduler ---
            after_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=decay_steps, 
                eta_min=0 
            )

            self.agent.scheduler = WarmupScheduler(
                optimizer=optimizer,
                base_lr=config.optimizer.learning_rate,
                n_warmup_steps=warmup_steps,
                after_scheduler=after_scheduler,
            )

        # ============================================================
        # 3. ReduceLROnPlateau
        # ============================================================
        elif config.scheduler.type == "reducerlonplatau":
            after_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=config.scheduler.factor,
                patience=config.scheduler.patience,
                verbose=True,
            )
            self.agent.scheduler = WarmupScheduler(
                optimizer=optimizer,
                base_lr=config.optimizer.learning_rate,
                n_warmup_steps=config.scheduler.warm_up_steps,
                after_scheduler=after_scheduler,
            )

        # ============================================================
        # 4. StepLR
        # ============================================================
        elif config.scheduler.type == "stepLR":
            after_scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=config.scheduler.lr_decay_step,
                gamma=config.scheduler.lr_decay_ratio,
            )
            self.agent.scheduler = WarmupScheduler(
                optimizer=optimizer,
                base_lr=config.optimizer.learning_rate,
                n_warmup_steps=config.scheduler.warm_up_steps,
                after_scheduler=after_scheduler,
            )

        # ============================================================
        # 5. MultiStepLR
        # ============================================================
        elif config.scheduler.type == "MultistepLR":
            after_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=config.scheduler.milestones,
                gamma=config.scheduler.lr_decay_ratio,
            )
            self.agent.scheduler = WarmupScheduler(
                optimizer=optimizer,
                base_lr=config.optimizer.learning_rate,
                n_warmup_steps=config.scheduler.warm_up_steps,
                after_scheduler=after_scheduler,
            )

        # ============================================================
        # 6. Fallback
        # ============================================================
        else:
            # 假设有一个默认的 No_Scheduler 或者 pass
            pass

    def load(self):

        if "save_base_dir" in self.agent.config.model:
            file_name = os.path.join(self.agent.config.model.save_base_dir, self.agent.config.model.save_dir)
        else:
            file_name = self.agent.config.model.save_dir

        if not os.path.exists(file_name):
            return False

        checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)

        if self.agent.accelerator.is_main_process:
            self.agent.logger.info("Loading checkpoint: {}".format(file_name))

        checkpoint["model_state_dict"] = {key.replace("module.", ""): value for key, value in checkpoint["model_state_dict"].items()}

        self.agent.model.load_state_dict(checkpoint["model_state_dict"])
        # self.agent.best_model.load_state_dict(checkpoint["best_model_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.agent.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "training_dataloder_generator_state" in checkpoint:
            self.agent.data_loader.train_loader.generator.set_state(checkpoint["training_dataloder_generator_state"])
        if "logs" in checkpoint:
            self.agent.logs = checkpoint["logs"]

        if hasattr(checkpoint, "metrics"):
            self.agent.data_loader.load_metrics_ongoing(checkpoint["metrics"])

        if hasattr(self.agent.logs, "weights"):
            self.agent.data_loader.weights = self.agent.logs["weights"]
            self.agent.weights = self.agent.data_loader.weights
            self.agent.logger.info("Loaded loss weights are:", self.agent.weights)

        for step in self.agent.logs["train_logs"]:
            wandb.log(
                {
                    "train": self.agent.logs["train_logs"][step],
                    "val": self.agent.logs["val_logs"][step],
                },
                step=step,
            )
            for i, lr in enumerate(self.agent.logs["train_logs"][step]["learning_rate"]):
                wandb.log(
                    {"lr": lr, "val": self.agent.logs["val_logs"][step]},
                    step=i + step - self.agent.config.early_stopping.validate_every,
                )

        self.agent.loss = nn.CrossEntropyLoss()

        message = ""
        if "step" in self.agent.logs["best_logs"]:
            message += Fore.WHITE + "The best in step: {} so far \n".format(int(self.agent.logs["best_logs"]["step"] / self.agent.config.early_stopping.validate_every))

            if "loss" in self.agent.logs["best_logs"]:
                for i, v in self.agent.logs["best_logs"]["loss"].items():
                    message += Fore.RED + "{} : {:.6f} ".format(i, v)
            if "acc" in self.agent.logs["best_logs"]:
                for i, v in self.agent.logs["best_logs"]["acc"].items():
                    message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)
            if "f1" in self.agent.logs["best_logs"]:
                for i, v in self.agent.logs["best_logs"]["f1"].items():
                    message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, v * 100)
            if "k" in self.agent.logs["best_logs"]:
                for i, v in self.agent.logs["best_logs"]["k"].items():
                    message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)

        if self.agent.accelerator.is_main_process:
            self.agent.logger.info("Checkpoint loaded successfully")
            self.agent.logger.info(message)

    def load_encoder(self, enc_args):
        encs = []
        for num_enc in range(len(enc_args)):
            enc_class = globals()[enc_args[num_enc]["model"]]
            args = enc_args[num_enc]["args"]

            if "encoders" in enc_args[num_enc]:
                enc_enc = self.load_encoder(enc_args[num_enc]["encoders"])
                enc = enc_class(encs=enc_enc, args=args)
            else:
                enc = enc_class(args=args, encs=[])

            pretrained_encoder_args = enc_args[num_enc].get("pretrainedEncoder", {"use": False})
            if pretrained_encoder_args["use"]:

                file_path = pretrained_encoder_args.get("dir", "")
                if "save_base_dir" in self.agent.config.model:
                    file_path = os.path.join(self.agent.config.model.save_base_dir, file_path)
                # === 加入这行 ===
                print(f"\n[DEBUG CHECK] Loading model from: {file_path}\n")
                # ================
                checkpoint = torch.load(file_path, weights_only=False)
                if "encoder_state_dict" in checkpoint:
                    missing_keys, unexpected_keys = enc.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
                    if missing_keys:
                        print(f"Missing keys in state_dict: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys in state_dict: {unexpected_keys}")

                elif "best_model_state_dict" in checkpoint:
                    logging.info("Loading enc best model state dict from {}".format(file_path))
                    if "VaVL" not in enc_args[num_enc]["model"]:
                        print("Replacing module")
                        checkpoint["best_model_state_dict"] = {key.replace("module.", ""): value for key, value in checkpoint["best_model_state_dict"].items()}
                    missing_keys, unexpected_keys = enc.load_state_dict(checkpoint["best_model_state_dict"], strict=False)
                    if missing_keys:
                        print(f"Missing keys in state_dict: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys in state_dict: {unexpected_keys}")

            encs.append(enc)
        return encs
