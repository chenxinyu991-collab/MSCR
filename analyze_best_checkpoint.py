import argparse
import json
from copy import deepcopy

import torch
from easydict import EasyDict

from agents.general_agent.agent import Agent


def deep_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_config(args):
    default_cfg = load_json(args.default_config)
    cfg = load_json(args.config)

    merged = deep_update(deepcopy(default_cfg), cfg)

    # common CLI overrides
    if args.fold is not None:
        if "dataset" in merged and "data_split" in merged["dataset"]:
            merged["dataset"]["data_split"]["fold"] = args.fold

    if args.lr is not None:
        merged["optimizer"]["learning_rate"] = args.lr

    if args.wd is not None:
        merged["optimizer"]["weight_decay"] = args.wd

    if args.batch_size is not None:
        merged["training_params"]["batch_size"] = args.batch_size
        merged["training_params"]["test_batch_size"] = args.batch_size

    if args.l is not None:
        merged["model"]["args"]["bias_infusion"]["l"] = args.l

    if args.num_samples is not None:
        merged["model"]["args"]["bias_infusion"]["num_samples"] = args.num_samples

    if args.reg_by is not None:
        merged["model"]["args"]["bias_infusion"]["reg_by"] = args.reg_by

    if args.contr_coeff is not None:
        merged["model"]["args"]["bias_infusion"]["contr_coeff"] = args.contr_coeff

    if args.lib is not None:
        merged["model"]["args"]["bias_infusion"]["lib"] = args.lib

    if args.multil is not None:
        if "multi_loss" in merged["model"]["args"]:
            if "multi_supervised_w" in merged["model"]["args"]["multi_loss"]:
                for key in merged["model"]["args"]["multi_loss"]["multi_supervised_w"]:
                    if key != "combined":
                        merged["model"]["args"]["multi_loss"]["multi_supervised_w"][key] = args.multil

    return EasyDict(merged)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--default_config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint (.pth.tar)")

    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--l", type=float, default=None)
    parser.add_argument("--multil", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--reg_by", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--contr_coeff", type=float, default=None)
    parser.add_argument("--lib", type=float, default=None)
    parser.add_argument("--test_only", action="store_true", help="Only run test set analysis")
    parser.add_argument("--val_only", action="store_true", help="Only run val set analysis")

    return parser.parse_args()


def load_checkpoint_to_model(agent, ckpt_path):
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # common possible checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # maybe the checkpoint itself is already a state_dict
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format")

    # If model is wrapped by accelerator/DDP, load into unwrapped model
    try:
        model_to_load = agent.accelerator.unwrap_model(agent.model)
    except Exception:
        model_to_load = agent.model

    missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)

    print("[INFO] Checkpoint loaded.")
    print("[INFO] Missing keys:", missing)
    print("[INFO] Unexpected keys:", unexpected)


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)

    # Build agent
    agent = Agent(config)

    # Prepare model/dataloader for accelerator
    agent.accelerate_components()

    # Manually load the specified checkpoint
    load_checkpoint_to_model(agent, args.ckpt)

    # Run validation / test only
    if not args.test_only:
        print("[INFO] Running validation set...")
        agent.validator_tester.validate(test_set=False)
        val_metrics = agent.evaluators.val_evaluator.evaluate()
        agent.monitor_n_saver.print_valid_results(val_metrics, -1)

    if not args.val_only:
        print("[INFO] Running test set...")
        agent.validator_tester.validate(test_set=True)
        test_metrics = agent.evaluators.test_evaluator.evaluate()
        agent.monitor_n_saver.print_valid_results(test_metrics, -1, test=True)