import os
import json
import copy
import itertools
import subprocess
import re
import time
from pathlib import Path

# =========================================================
# 1. 固定路径
# =========================================================
BASE_CONFIG = "/inspire/ssd/project/sais-scientific-models-evaluation/public/global_user/8331-huao/llj_temp/cxy/MCR/configs/CREMA_D/release/vit/MCR_FSR.json"
DEFAULT_CONFIG = "/inspire/ssd/project/sais-scientific-models-evaluation/public/global_user/8331-huao/llj_temp/cxy/MCR/configs/CREMA_D/default_config_cremad_vit.json"

# 所有新实验统一放这里
EXP_ROOT = "/inspire/ssd/project/sais-scientific-models-evaluation/public/global_user/8331-huao/llj_temp/cxy/exp"

# 给这轮网格搜索起一个专属名字
EXP_NAME = "fsr_syn_dom_grid_v1"

RUN_ROOT = os.path.join(EXP_ROOT, EXP_NAME)
CONFIG_DIR = os.path.join(RUN_ROOT, "configs")
ARTIFACT_DIR = os.path.join(RUN_ROOT, "artifacts")
LOG_DIR = os.path.join(ARTIFACT_DIR, "logs")
SUMMARY_TXT = os.path.join(RUN_ROOT, "summary.txt")
SUMMARY_JSONL = os.path.join(RUN_ROOT, "summary.jsonl")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================================================
# 2. 训练命令模板
# =========================================================
# 注意：
# - cwd 会设成 MCR 项目根目录
# - 每次训练都用生成后的 config
# - 默认配置固定为你给的 default_config_cremad_vit.json
TRAIN_WORKDIR = "/inspire/ssd/project/sais-scientific-models-evaluation/public/global_user/8331-huao/llj_temp/cxy/MCR"

TRAIN_CMD_TEMPLATE = (
    "python train.py "
    "--config {config} "
    "--default_config {default_config} "
    "--fold 0 "
    "--lr 0.00005 "
    "--wd 5e-6 "
    "--l 0.1 "
    "--multil 1 "
    "--num_samples 32 "
    "--batch_size 32 "
    "--contr_coeff 1 "
    "--lib 1"
)

# =========================================================
# 3. 网格搜索范围
#    按目前结果，小范围优先搜 gate 强度和 dom 分支阈值
# =========================================================
GRID = {
    "fsr_gate_floor": [0.2, 0.3],
    "fsr_gate_tau_dom": [0.05, 0.15, 0.20],
    "fsr_gate_kappa_dom": [5.0, 7.0],
}

# =========================================================
# 4. 日志解析
#    解析这种格式：
#    Val Epoch ... Acc_combined: 86.90
#    Test Epoch ... Acc_combined: 83.65
# =========================================================
VAL_ACC_PATTERN = re.compile(
    r"Val Epoch.*?Acc_combined:\s*([0-9]+\.[0-9]+)",
    re.IGNORECASE
)

TEST_ACC_PATTERN = re.compile(
    r"Test Epoch.*?Acc_combined:\s*([0-9]+\.[0-9]+)",
    re.IGNORECASE
)

def extract_best_metric_from_log(log_text, pattern):
    vals = []
    for m in pattern.finditer(log_text):
        try:
            vals.append(float(m.group(1)))
        except Exception:
            pass
    return max(vals) if vals else None

# =========================================================
# 5. 工具函数
# =========================================================
def set_nested(d, keys, value):
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value

def get_bias_infusion_path():
    return ["model", "args", "bias_infusion"]

def format_run_name(params):
    parts = []
    for k, v in params.items():
        parts.append(f"{k}-{v}")
    return "__".join(parts).replace("/", "_")

# =========================================================
# 6. 主流程
# =========================================================
def main():
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    grid_keys = list(GRID.keys())
    grid_values = [GRID[k] for k in grid_keys]
    combinations = list(itertools.product(*grid_values))

    with open(SUMMARY_TXT, "w", encoding="utf-8") as summary_f:
        summary_f.write("FSR Grid Search Summary\n")
        summary_f.write("=" * 120 + "\n")
        summary_f.write(f"EXP_NAME      : {EXP_NAME}\n")
        summary_f.write(f"BASE_CONFIG   : {BASE_CONFIG}\n")
        summary_f.write(f"DEFAULT_CONFIG: {DEFAULT_CONFIG}\n")
        summary_f.write(f"RUN_ROOT      : {RUN_ROOT}\n")
        summary_f.write(f"TOTAL_RUNS    : {len(combinations)}\n\n")

    # 清空 jsonl
    with open(SUMMARY_JSONL, "w", encoding="utf-8") as f:
        pass

    for idx, combo in enumerate(combinations, start=1):
        params = dict(zip(grid_keys, combo))
        run_name = f"run_{idx:03d}__" + format_run_name(params)

        cfg = copy.deepcopy(base_cfg)
        bias_path = get_bias_infusion_path()

        # 强制双门 gate
        set_nested(cfg, bias_path + ["fsr_use_gate"], True)
        set_nested(cfg, bias_path + ["fsr_gate_mode"], "syn_dom")

        # 覆盖本轮搜索参数
        for k, v in params.items():
            set_nested(cfg, bias_path + [k], v)

        # 每个实验单独模型名，避免覆盖
        set_nested(cfg, ["model", "save_base_dir"], os.path.join(ARTIFACT_DIR, "checkpoints"))
        set_nested(cfg, ["model", "save_dir"], f"{run_name}" + "_{}.pth.tar")

        # 保存生成配置
        config_path = os.path.join(CONFIG_DIR, f"{run_name}.json")
        log_path = os.path.join(LOG_DIR, f"{run_name}.log")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        cmd = TRAIN_CMD_TEMPLATE.format(
            config=config_path,
            default_config=DEFAULT_CONFIG,
        )

        start_time = time.time()
        status = "success"
        return_code = 0

        print(f"[{idx}/{len(combinations)}] Running: {run_name}")
        print(f"Command: {cmd}")
        print(f"Config : {config_path}")
        print(f"Log    : {log_path}")

        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(f"RUN_NAME: {run_name}\n")
            log_f.write(f"PARAMS: {json.dumps(params, ensure_ascii=False)}\n")
            log_f.write(f"CONFIG: {config_path}\n")
            log_f.write(f"DEFAULT_CONFIG: {DEFAULT_CONFIG}\n")
            log_f.write(f"COMMAND: {cmd}\n")
            log_f.write("=" * 120 + "\n\n")
            log_f.flush()

            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=TRAIN_WORKDIR,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                return_code = proc.returncode
                if return_code != 0:
                    status = "failed"
            except Exception as e:
                status = "exception"
                return_code = -1
                log_f.write(f"\n[EXCEPTION] {repr(e)}\n")

        elapsed = time.time() - start_time

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_text = f.read()

        best_val = extract_best_metric_from_log(log_text, VAL_ACC_PATTERN)
        best_test = extract_best_metric_from_log(log_text, TEST_ACC_PATTERN)

        result = {
            "run_name": run_name,
            "params": params,
            "config_path": config_path,
            "log_path": log_path,
            "status": status,
            "return_code": return_code,
            "elapsed_sec": round(elapsed, 2),
            "best_val": best_val,
            "best_test": best_test,
        }

        with open(SUMMARY_TXT, "a", encoding="utf-8") as summary_f:
            summary_f.write(f"RUN {idx}/{len(combinations)}\n")
            summary_f.write("-" * 120 + "\n")
            summary_f.write(f"run_name   : {run_name}\n")
            summary_f.write(f"params     : {json.dumps(params, ensure_ascii=False)}\n")
            summary_f.write(f"status     : {status}\n")
            summary_f.write(f"return_code: {return_code}\n")
            summary_f.write(f"elapsed(s) : {elapsed:.2f}\n")
            summary_f.write(f"best_val   : {best_val}\n")
            summary_f.write(f"best_test  : {best_test}\n")
            summary_f.write(f"log_path   : {log_path}\n")
            summary_f.write(f"config_path: {config_path}\n\n")

        with open(SUMMARY_JSONL, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 排序输出
    all_results = []
    with open(SUMMARY_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except Exception:
                    pass

    ranked = [r for r in all_results if r.get("best_test") is not None]
    ranked.sort(key=lambda x: x["best_test"], reverse=True)

    with open(SUMMARY_TXT, "a", encoding="utf-8") as summary_f:
        summary_f.write("\n")
        summary_f.write("=" * 120 + "\n")
        summary_f.write("RANKED BY BEST TEST\n")
        summary_f.write("=" * 120 + "\n")
        for i, r in enumerate(ranked, start=1):
            summary_f.write(
                f"{i:02d}. {r['run_name']} | best_test={r['best_test']} | best_val={r['best_val']} | params={json.dumps(r['params'], ensure_ascii=False)}\n"
            )

    print("All runs finished.")
    print(f"Summary txt  : {SUMMARY_TXT}")
    print(f"Summary jsonl: {SUMMARY_JSONL}")

if __name__ == "__main__":
    main()