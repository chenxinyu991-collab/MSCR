from collections import defaultdict
import json
import math
import os

import torch


def _task_name(task):
    if task in ("reg", "regression"):
        return "regression"
    if task in ("classification", "cls"):
        return "classification"
    raise ValueError(f"Unsupported task for conditional recovery: {task}")


def _reg_matrix(value, name):
    value = value.detach().cpu()
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty [N], [N,1], or [N,D] tensor.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return value.to(torch.float64)


def compute_conditional_recovery(pred_combined, unimodal_predictions, labels,
                                 task="classification", tau_err=1.0):
    """Return full-dataset counts, global-percent matrix and conditional recovery.

    unimodal_predictions: ordered dict, e.g. {'c': c, 'g': g, 'f': f}.
    Column index = sum(correct_i * 2**i). Column 0 is all modalities wrong.
    No eligible samples -> recovery_percent=None (not an invented zero).
    For vector regression, correctness uses mean absolute error over output dims.
    """
    task = _task_name(task)
    names = list(unimodal_predictions)
    if len(names) not in (2, 3):
        raise ValueError("Conditional recovery requires two or three unimodal predictions.")
    if task == "regression":
        tau_err = float(tau_err)
        if not math.isfinite(tau_err) or tau_err < 0:
            raise ValueError("tau_err must be finite and nonnegative.")
        target = _reg_matrix(labels, "labels")

        def correct(pred):
            pred = _reg_matrix(pred, "prediction")
            if pred.shape != target.shape:
                raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}.")
            return (pred - target).abs().mean(dim=1) <= tau_err

        criterion = "absolute_error" if target.shape[1] == 1 else "mean_absolute_error"
    else:
        target = labels.detach().cpu()
        if target.ndim == 2 and target.shape[1] == 1:
            target = target[:, 0]
        if target.ndim != 1 or target.numel() == 0:
            raise ValueError("Classification labels must be nonempty [N] or [N,1].")
        if not torch.isfinite(target).all() or not torch.equal(target, target.long().to(target.dtype)):
            raise ValueError("Classification labels must be finite integer class indices.")

        def correct(pred):
            pred = pred.detach().cpu()
            if pred.ndim != 2 or pred.shape[0] != target.shape[0] or pred.shape[1] < 1:
                raise ValueError("Classification predictions must have shape [N,C].")
            if not torch.isfinite(pred).all():
                raise ValueError("Classification predictions contain NaN or infinity.")
            if (target < 0).any() or (target >= pred.shape[1]).any():
                raise ValueError("Classification labels are outside the class index range.")
            return pred.argmax(dim=1) == target

        criterion = "argmax"

    joint_correct = correct(pred_combined)
    modality_correct = torch.stack([correct(unimodal_predictions[k]) for k in names])
    n = joint_correct.numel()
    column = torch.zeros(n, dtype=torch.long)
    for i in range(len(names)):
        column += modality_correct[i].long() * (1 << i)
    num_columns = 1 << len(names)
    counts = torch.bincount(joint_correct.long() * num_columns + column,
                            minlength=2 * num_columns).reshape(2, num_columns)
    percentages = counts.double() * (100.0 / n)
    # Integer counts are pooled over the dataset before division.
    eligible = int(counts[:, 0].sum())
    recovered = int(counts[1, 0])
    columns = (["both_wrong", "only_m1_correct", "only_m2_correct", "both_correct"]
               if len(names) == 2 else
               ["all_wrong", "only_m1_correct", "only_m2_correct", "m1_m2_correct",
                "only_m3_correct", "m1_m3_correct", "m2_m3_correct", "all_correct"])
    return {
        "task": task,
        "criterion": criterion,
        "tau_err": tau_err if task == "regression" else None,
        "modalities": names,
        "rows": ["mm_wrong", "mm_correct"],
        "columns": columns,
        "column_correctness": [[bool(mask & (1 << i)) for i in range(len(names))]
                               for mask in range(num_columns)],
        "sample_count": n,
        "eligible_count": eligible,
        "recovered_count": recovered,
        "recovery_percent": 100.0 * recovered / eligible if eligible else None,
        "matrix_counts": counts.tolist(),
        "matrix_percent": percentages.tolist(),
        "joint_correct_percent": 100.0 * int(joint_correct.sum()) / n,
    }


def _legacy_result(report, return_percent):
    matrix = report["matrix_percent"] if return_percent else report["matrix_counts"]
    stats = {row: dict(zip(report["columns"], values))
             for row, values in zip(report["rows"], matrix)}
    return stats, matrix


def analyze_multimodal_error(pred_combined, pred_m1, pred_m2, labels,
                             return_percent=True, pred_m3=None):
    """Backward-compatible classification wrapper; optional third modality."""
    predictions = {"m1": pred_m1, "m2": pred_m2}
    if pred_m3 is not None:
        predictions["m3"] = pred_m3
    report = compute_conditional_recovery(pred_combined, predictions, labels, task="classification")
    return _legacy_result(report, return_percent)


def analyze_multimodal_error_regression(pred_combined, pred_m1, pred_m2, labels,
                                        tau_err=1.0, return_percent=True, pred_m3=None):
    """Absolute-error correctness; optional third modality. Default tau is 1.0."""
    predictions = {"m1": pred_m1, "m2": pred_m2}
    if pred_m3 is not None:
        predictions["m3"] = pred_m3
    report = compute_conditional_recovery(pred_combined, predictions, labels,
                                          task="regression", tau_err=tau_err)
    return _legacy_result(report, return_percent)


class Validator_Tester:
    def __init__(self, agent):
        self.agent = agent
        self.multi_supervised = False
        self.task_name = _task_name(agent.config.get("task", "classification"))
        self.valtest_step_func = ("valtest_one_step_regression" if self.task_name == "regression"
                                 else "valtest_one_step")
        self.this_valtest_step_func = getattr(self, self.valtest_step_func)
        self.last_error_analysis = None
        self._get_loss_weights()

    def _recovery_modalities(self, predictions, selected):
        if selected is None:
            configured = self.agent.config.model.args.get("recovery_modalities")
            selected = list(configured) if configured is not None else (
                ["c", "g", "f"] if "f" in predictions else ["c", "g"])
        if len(selected) not in (2, 3) or len(set(selected)) != len(selected) or "combined" in selected:
            raise ValueError("recovery_modalities must name two or three distinct unimodal outputs.")
        missing = [key for key in ["combined", *selected] if key not in predictions]
        if missing:
            raise KeyError(f"Missing recovery predictions: {missing}; refusing to drop a modality.")
        return selected

    def _gather_recovery(self, predictions, labels):
        accelerator = self.agent.accelerator
        payload = (predictions, labels)
        if hasattr(accelerator, "gather_for_metrics"):
            # Unlike raw gather, this removes distributed final-batch padding
            # when the loader is prepared by Accelerate.
            return accelerator.gather_for_metrics(payload)
        if getattr(accelerator, "num_processes", 1) != 1:
            raise RuntimeError("Distributed recovery requires accelerator.gather_for_metrics.")
        return payload

    def _write_recovery(self, report):
        if not getattr(self.agent.accelerator, "is_main_process", True):
            return
        save_dir = self.agent.config.model.save_base_dir
        os.makedirs(save_dir, exist_ok=True)
        record = {"epoch": int(self.agent.logs["current_epoch"]),
                  "step": int(self.agent.logs["current_step"]), **report}
        with open(os.path.join(save_dir, "conditional_recovery_test.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        rate = report["recovery_percent"]
        rate_text = "undefined (no all-wrong samples)" if rate is None else f"{rate:.6f}%"
        with open(os.path.join(save_dir, "multimodal_error_analysis_test.txt"), "a", encoding="utf-8") as handle:
            handle.write("=" * 80 + "\n")
            handle.write(f"Epoch: {record['epoch']}, Step: {record['step']}\n")
            handle.write(f"Mode: {report['task']}; criterion: {report['criterion']}; tau_err: {report['tau_err']}\n")
            handle.write(f"Modalities (m1,m2[,m3]): {report['modalities']}\n")
            handle.write(f"Rows: {report['rows']}\nCols: {report['columns']}\n")
            handle.write(f"Sample count: {report['sample_count']}\n")
            handle.write(f"Matrix (counts): {report['matrix_counts']}\n")
            handle.write(f"Matrix (% of ALL samples): {report['matrix_percent']}\n")
            handle.write(f"All-unimodal-wrong count: {report['eligible_count']}\n")
            handle.write(f"Recovered count: {report['recovered_count']}\n")
            handle.write(f"Conditional recovery: {rate_text}\n")
            handle.write(f"Joint correct rate over all samples: {report['joint_correct_percent']:.6f}%\n")

    def validate(self, test_set=False):
        self.agent.model.eval()
        self.last_error_analysis = None
        evaluator = self.agent.evaluators.test_evaluator if test_set else self.agent.evaluators.val_evaluator
        loader = self.agent.data_loader.test_loader if test_set else self.agent.data_loader.valid_loader
        evaluator.reset()
        selected = None
        all_predictions = defaultdict(list)
        all_labels = []
        with torch.no_grad():
            for served_dict in loader:
                if isinstance(served_dict, tuple):
                    served_dict = {
                        "data": {"c": served_dict[0][0], "f": served_dict[0][1], "g": served_dict[0][2]},
                        "label": served_dict[3].squeeze(dim=1),
                    }
                    if self.task_name == "classification":
                        served_dict["label"] = torch.where(served_dict["label"] == -1, 0, served_dict["label"])
                step = self.this_valtest_step_func(served_dict)

                # Keep the existing evaluator path; gather only its tensor payload.
                payload = {k: v for k, v in step.items() if k != "process_flag"}
                gathered = self.agent.accelerator.gather(payload)
                gathered["process_flag"] = bool(step.get("process_flag", True))
                if gathered["process_flag"]:
                    evaluator.process(gathered)

                if test_set:
                    selected = self._recovery_modalities(step["pred"], selected)
                    predictions, labels = self._gather_recovery(
                        {k: step["pred"][k] for k in ["combined", *selected]}, step["label"],
                    )
                    for key, value in predictions.items():
                        all_predictions[key].append(value.detach().cpu())
                    all_labels.append(labels.detach().cpu())

        if not test_set:
            return None
        if not all_labels:
            raise ValueError("No test batches were available for conditional recovery.")
        predictions = {key: torch.cat(values, dim=0) for key, values in all_predictions.items()}
        labels = torch.cat(all_labels, dim=0)
        report = compute_conditional_recovery(
            predictions["combined"], {key: predictions[key] for key in selected}, labels,
            task=self.task_name,
            tau_err=self.agent.config.model.args.get("recovery_tau_err", 1.0),
        )
        self.last_error_analysis = report
        self._write_recovery(report)
        return report

    def _prepare_data(self, served_dict):
        device = getattr(self.agent.accelerator, "device", None)
        if device is None:
            device = getattr(self.agent, "device", "cuda" if torch.cuda.is_available() else "cpu")
        data = {view: value.to(device) for view, value in served_dict["data"].items()
                if torch.is_tensor(value)}
        data.update({view: value.float() for view, value in data.items() if isinstance(view, int)})
        return data, served_dict["label"].to(device)

    def valtest_one_step(self, served_dict):
        data, label = self._prepare_data(served_dict)
        label = label.reshape(-1).long()
        self.agent.model.eval()
        output = self.agent.model(data, label=label)
        total_loss = label.new_zeros((), dtype=torch.float32)
        losses = {}
        if self.agent.config.model.args.get("validation_loss_w", {"combined": 1}):
            supervised = self.agent.config.model.args.multi_loss.multi_supervised_w
            for key, pred in output["preds"].items():
                if supervised.get(key, 0) != 0:
                    loss = self.agent.loss(pred, label)
                    total_loss = total_loss + self.w_loss[key] * loss
                    losses[f"ce_loss_{key}"] = loss.detach()
        losses["total"] = total_loss.detach()
        return {"loss": losses, "pred": {k: v.detach() for k, v in output["preds"].items()},
                "features": output["features"], "label": label, "process_flag": label.shape[0] > 0}

    def valtest_one_step_regression(self, served_dict):
        data, label = self._prepare_data(served_dict)
        if label.ndim == 0:
            label = label.unsqueeze(0)
        if label.ndim == 2 and label.shape[1] == 1:
            label = label[:, 0]
        output = self.agent.model(data)
        joint = output["preds"]["combined"]
        # Preserve evaluation L1 loss, but forbid accidental [B,1] vs [B] broadcasting.
        pred_matrix = joint.unsqueeze(1) if joint.ndim == 1 else joint
        target_matrix = label.unsqueeze(1) if label.ndim == 1 else label
        if pred_matrix.shape != target_matrix.shape:
            raise ValueError("Regression prediction/label shape mismatch in validation loss.")
        loss = torch.nn.functional.l1_loss(pred_matrix, target_matrix).detach()
        return {"loss": {"l1_loss_combined": loss, "total": loss},
                "pred": {k: v.detach() for k, v in output["preds"].items()},
                "label": label, "process_flag": label.shape[0] > 0}

    def _get_loss_weights(self):
        self.w_loss = defaultdict(int)
        self.w_loss["total"] = 1
        self.w_loss.update(self.agent.config.model.args.get("validation_loss_w", {"combined": 1}))
