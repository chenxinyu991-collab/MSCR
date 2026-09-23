from collections import defaultdict

import torch
import os
from tqdm import tqdm


def analyze_multimodal_error(pred_combined, pred_m1, pred_m2, labels, return_percent=True):
    """
    Classification version.

    Group samples by unimodal correctness and check whether the multimodal model is correct.

    Columns:
        0: both unimodals wrong
        1: only modality 1 correct
        2: only modality 2 correct
        3: both unimodals correct

    Rows:
        0: multimodal wrong
        1: multimodal correct
    """
    if len(labels.shape) > 1:
        labels = labels.flatten()

    mm_pred = torch.argmax(pred_combined, dim=1)
    m1_pred = torch.argmax(pred_m1, dim=1)
    m2_pred = torch.argmax(pred_m2, dim=1)

    m1_correct = (m1_pred == labels)
    m2_correct = (m2_pred == labels)
    mm_correct = (mm_pred == labels)

    stats = {
        "mm_wrong": {
            "both_wrong": 0,
            "only_m1_correct": 0,
            "only_m2_correct": 0,
            "both_correct": 0,
        },
        "mm_correct": {
            "both_wrong": 0,
            "only_m1_correct": 0,
            "only_m2_correct": 0,
            "both_correct": 0,
        }
    }

    for i in range(labels.size(0)):
        if (not m1_correct[i]) and (not m2_correct[i]):
            col = "both_wrong"
        elif m1_correct[i] and (not m2_correct[i]):
            col = "only_m1_correct"
        elif (not m1_correct[i]) and m2_correct[i]:
            col = "only_m2_correct"
        else:
            col = "both_correct"

        row = "mm_correct" if mm_correct[i] else "mm_wrong"
        stats[row][col] += 1

    total = labels.size(0)

    if return_percent:
        for row in stats:
            for col in stats[row]:
                stats[row][col] = 100.0 * stats[row][col] / total

    matrix = [
        [
            stats["mm_wrong"]["both_wrong"],
            stats["mm_wrong"]["only_m1_correct"],
            stats["mm_wrong"]["only_m2_correct"],
            stats["mm_wrong"]["both_correct"],
        ],
        [
            stats["mm_correct"]["both_wrong"],
            stats["mm_correct"]["only_m1_correct"],
            stats["mm_correct"]["only_m2_correct"],
            stats["mm_correct"]["both_correct"],
        ],
    ]

    return stats, matrix


def analyze_multimodal_error_regression(pred_combined, pred_m1, pred_m2, labels, tau_err=0.5, return_percent=True):
    """
    Regression version.

    We define correctness by whether the absolute prediction error is below a threshold.

    Columns:
        0: both unimodals wrong
        1: only modality 1 correct
        2: only modality 2 correct
        3: both unimodals correct

    Rows:
        0: multimodal wrong
        1: multimodal correct

    Args:
        pred_combined: [B] / [B,1] / [B,D]
        pred_m1:       [B] / [B,1] / [B,D]
        pred_m2:       [B] / [B,1] / [B,D]
        labels:        [B] / [B,1] / [B,D]
        tau_err:       threshold for considering a prediction "correct"
        return_percent: whether to normalize stats to percentages
    """
    # flatten to [B] if possible
    if len(labels.shape) > 1:
        labels = labels.view(labels.size(0), -1)
        if labels.size(1) == 1:
            labels = labels.flatten()

    if len(pred_combined.shape) > 1:
        pred_combined = pred_combined.view(pred_combined.size(0), -1)
        if pred_combined.size(1) == 1:
            pred_combined = pred_combined.flatten()

    if len(pred_m1.shape) > 1:
        pred_m1 = pred_m1.view(pred_m1.size(0), -1)
        if pred_m1.size(1) == 1:
            pred_m1 = pred_m1.flatten()

    if len(pred_m2.shape) > 1:
        pred_m2 = pred_m2.view(pred_m2.size(0), -1)
        if pred_m2.size(1) == 1:
            pred_m2 = pred_m2.flatten()

    # support scalar regression and vector regression
    if len(labels.shape) == 1:
        mm_err = torch.abs(pred_combined - labels)
        m1_err = torch.abs(pred_m1 - labels)
        m2_err = torch.abs(pred_m2 - labels)
    else:
        mm_err = torch.mean(torch.abs(pred_combined - labels), dim=1)
        m1_err = torch.mean(torch.abs(pred_m1 - labels), dim=1)
        m2_err = torch.mean(torch.abs(pred_m2 - labels), dim=1)

    m1_correct = (m1_err <= tau_err)
    m2_correct = (m2_err <= tau_err)
    mm_correct = (mm_err <= tau_err)

    stats = {
        "mm_wrong": {
            "both_wrong": 0,
            "only_m1_correct": 0,
            "only_m2_correct": 0,
            "both_correct": 0,
        },
        "mm_correct": {
            "both_wrong": 0,
            "only_m1_correct": 0,
            "only_m2_correct": 0,
            "both_correct": 0,
        }
    }

    total = mm_correct.size(0)

    for i in range(total):
        if (not m1_correct[i]) and (not m2_correct[i]):
            col = "both_wrong"
        elif m1_correct[i] and (not m2_correct[i]):
            col = "only_m1_correct"
        elif (not m1_correct[i]) and m2_correct[i]:
            col = "only_m2_correct"
        else:
            col = "both_correct"

        row = "mm_correct" if mm_correct[i] else "mm_wrong"
        stats[row][col] += 1

    if return_percent:
        for row in stats:
            for col in stats[row]:
                stats[row][col] = 100.0 * stats[row][col] / total

    matrix = [
        [
            stats["mm_wrong"]["both_wrong"],
            stats["mm_wrong"]["only_m1_correct"],
            stats["mm_wrong"]["only_m2_correct"],
            stats["mm_wrong"]["both_correct"],
        ],
        [
            stats["mm_correct"]["both_wrong"],
            stats["mm_correct"]["only_m1_correct"],
            stats["mm_correct"]["only_m2_correct"],
            stats["mm_correct"]["both_correct"],
        ],
    ]

    return stats, matrix


class Validator_Tester:
    def __init__(self, agent):
        self.agent = agent
        self.multi_supervised = False

        # NOTE:
        # Your config currently may use task="regression" or task="classification".
        # The previous code was checking get("task", "classification").
        # Keep that behavior but support both strings.
        task_name = self.agent.config.get("task", "classification")

        if task_name == "classification":
            self.valtest_step_func = "valtest_one_step"
        elif task_name == "regression":
            self.valtest_step_func = "valtest_one_step_regression"
        elif task_name == "bias_measure":
            self.valtest_step_func = "valtest_one_step_biasmeasure"
        else:
            # fallback
            self.valtest_step_func = "valtest_one_step"

        self.this_valtest_step_func = getattr(self, self.valtest_step_func)
        self._get_loss_weights()

    def validate(self, test_set=False):

        self.agent.model.eval()

        this_evaluator = self.agent.evaluators.test_evaluator if test_set else self.agent.evaluators.val_evaluator
        this_dataloader = self.agent.data_loader.test_loader if test_set else self.agent.data_loader.valid_loader
        this_evaluator.reset()

        task_name = self.agent.config.get("task", "classification")

        with torch.no_grad():
            all_combined = []
            all_c = []
            all_g = []
            all_labels = []

            pbar = tqdm(
                enumerate(this_dataloader),
                total=len(this_dataloader),
                desc="Validation",
                leave=False,
                disable=True,
                position=1,
            )

            for batch_idx, served_dict in pbar:

                if type(served_dict) == tuple:
                    served_dict = {
                        "data": {
                            "c": served_dict[0][0],
                            "f": served_dict[0][1],
                            "g": served_dict[0][2],
                        },
                        "label": served_dict[3].squeeze(dim=1),
                    }
                    if task_name == "classification" and len(served_dict["label"][served_dict["label"] == -1]) > 0:
                        served_dict["label"][served_dict["label"] == -1] = 0

                step_outcome = self.this_valtest_step_func(served_dict)
                all_outputs = self.agent.accelerator.gather(step_outcome)

                # gather predictions for post-hoc matrix analysis
                if "combined" in all_outputs["pred"] and "c" in all_outputs["pred"] and "g" in all_outputs["pred"]:
                    all_combined.append(all_outputs["pred"]["combined"].detach().cpu())
                    all_c.append(all_outputs["pred"]["c"].detach().cpu())
                    all_g.append(all_outputs["pred"]["g"].detach().cpu())
                    all_labels.append(all_outputs["label"].detach().cpu())

                if all_outputs["process_flag"]:
                    this_evaluator.process(all_outputs)

                del step_outcome, served_dict

                mean_batch_loss, mean_batch_loss_message = this_evaluator.mean_batch_loss()

                pbar_message = "Validation batch {0:d}/{1:d} with {2:}".format(
                    batch_idx,
                    len(this_dataloader) - 1,
                    mean_batch_loss_message,
                )
                pbar.set_description(pbar_message)
                pbar.refresh()

            # --------------------------------------------------
            # Post-hoc multimodal error analysis
            # --------------------------------------------------
            if test_set and len(all_combined) > 0:
                all_combined = torch.cat(all_combined, dim=0)
                all_c = torch.cat(all_c, dim=0)
                all_g = torch.cat(all_g, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                if task_name == "classification":
                    stats, matrix = analyze_multimodal_error(
                        all_combined,
                        all_c,
                        all_g,
                        all_labels,
                        return_percent=True,
                    )
                    analysis_mode = "classification_argmax"
                elif task_name == "regression":
                    # You can change tau_err later if needed
                    tau_err = 1.0
                    stats, matrix = analyze_multimodal_error_regression(
                        all_combined,
                        all_c,
                        all_g,
                        all_labels,
                        tau_err=tau_err,
                        return_percent=True,
                    )
                    analysis_mode = f"regression_abs_error_tau_{tau_err}"
                else:
                    stats, matrix = None, None
                    analysis_mode = "unsupported"

                if matrix is not None:
                    test_acc_from_matrix = sum(matrix[1])

                    save_dir = self.agent.config.model.save_base_dir
                    os.makedirs(save_dir, exist_ok=True)

                    analysis_file = os.path.join(save_dir, "multimodal_error_analysis_test.txt")

                    with open(analysis_file, "a") as f:
                        f.write("=" * 100 + "\n")
                        f.write(f"Epoch: {self.agent.logs['current_epoch']}, Step: {self.agent.logs['current_step']}\n")
                        f.write(f"Mode: {analysis_mode}\n")
                        f.write("Rows: [MM Wrong, MM Correct]\n")
                        f.write("Cols: [Both Wrong, Only M1 Correct, Only M2 Correct, Both Correct]\n")
                        f.write(f"Matrix (%): {matrix}\n")
                        f.write(f"MM Correct Rate from Matrix (%): {test_acc_from_matrix:.4f}\n")
                        f.write(f"Detailed stats: {stats}\n")
                        f.write("=" * 100 + "\n")

    def valtest_one_step(self, served_dict):

        data = {
            view: served_dict["data"][view].cuda()
            for view in served_dict["data"]
            if type(served_dict["data"][view]) is torch.Tensor
        }
        data.update({view: data[view].float() for view in data if type(view) == int})

        label = served_dict["label"].squeeze().type(torch.LongTensor).cuda()

        if len(label.shape) == 0:
            label = label.unsqueeze(dim=0)

        self.agent.model.eval()
        output = self.agent.model(data, label=label)

        total_loss = torch.zeros(1).squeeze().cuda()
        output_losses, ce_loss = {}, {}

        if self.agent.config.model.args.get("validation_loss_w", {"combined": 1}):
            for k, v in output["preds"].items():
                if k in self.agent.config.model.args.multi_loss.multi_supervised_w and self.agent.config.model.args.multi_loss.multi_supervised_w[k] != 0:
                    if len(label) > 1:
                        if len(label.shape) > 1:
                            label = label.flatten()
                        ce_loss[k] = self.agent.loss(v, label)
                        total_loss += self.w_loss[k] * ce_loss[k]
                        ce_loss[k] = ce_loss[k].detach()
                        output_losses.update({"ce_loss_{}".format(k): ce_loss[k]})

        total_loss = total_loss.detach()
        output_losses.update({"total": total_loss})

        for i in output["preds"]:
            output["preds"][i] = output["preds"][i].detach()

        process_flag = False
        if len(label) > 1:
            process_flag = True

        return {
            "loss": output_losses,
            "pred": output["preds"],
            "features": output["features"],
            "label": label,
            "process_flag": process_flag,
        }

    def valtest_one_step_regression(self, served_dict):

        data = {
            view: served_dict["data"][view].cuda()
            for view in served_dict["data"]
            if type(served_dict["data"][view]) is torch.Tensor
        }
        data.update({view: data[view].float() for view in data if type(view) == int})

        label = served_dict["label"].cuda()

        output = self.agent.model(data)

        total_loss = torch.nn.L1Loss()(output["preds"]["combined"].flatten(), label)
        output_losses = {"l1_loss_combined": total_loss}

        total_loss = total_loss.detach()
        output_losses.update({"total": total_loss})

        for i in output["preds"]:
            output["preds"][i] = output["preds"][i].detach()

        process_flag = False
        if len(label) > 1:
            process_flag = True

        return {
            "loss": output_losses,
            "pred": output["preds"],
            "label": label,
            "process_flag": process_flag,
        }

    def _get_loss_weights(self):

        w_loss = defaultdict(int)
        w_loss["total"] = 1

        ws = self.agent.config.model.args.get("validation_loss_w", {"combined": 1})
        for k, v in ws.items():
            w_loss[k] = v

        self.w_loss = w_loss