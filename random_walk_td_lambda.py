import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# ============================================================
# 1. 实验基本设置
# ============================================================

# Random Walk:
#
# A - B - C - D - E - F - G
#
# A、G 为终止状态
# B~F 对应内部索引 0~4
#
# A = 实际位置 0
# B = 实际位置 1 -> index 0
# C = 实际位置 2 -> index 1
# D = 实际位置 3 -> index 2
# E = 实际位置 4 -> index 3
# F = 实际位置 5 -> index 4
# G = 实际位置 6
#
# 所有序列均从 D 开始。

NUM_STATES = 5

STATE_NAMES = ["B", "C", "D", "E", "F"]

# 理论上的理想预测值
IDEAL_VALUES = np.array([
    1 / 6,
    2 / 6,
    3 / 6,
    4 / 6,
    5 / 6
], dtype=float)

# Figure 4 原图展示的四条代表曲线。
FIGURE4_LAMBDAS = [0.0, 0.3, 0.8, 1.0]

# Figure 5 按原图可见点位采用 0.1 间隔；正文未逐项列出完整网格。
FIGURE5_LAMBDAS = [i / 10 for i in range(11)]

# 只计算两张图需要的参数并集，共用同一批数据，避免重复训练。
LAMBDAS = sorted(set(FIGURE4_LAMBDAS) | set(FIGURE5_LAMBDAS))

# 论文正文没有逐项给出 Figure 4 的完整 alpha 网格。
# 这里采用 0~0.60、步长 0.01 进行复现。
ALPHAS = np.arange(0.00, 0.61, 0.01)

# 论文：100 个 training sets
NUM_TRAINING_SETS = 100

# 每个 training set：10 条 sequences
SEQUENCES_PER_SET = 10

# 固定随机种子，保证每次运行得到相同结果
RANDOM_SEED = 42


# ============================================================
# 2. 生成一条 Random Walk
# ============================================================

def generate_sequence(rng):
    """
    生成一条从 D 开始的随机游走序列。

    返回：
        sequence:
            非终止状态的索引列表
            B=0, C=1, D=2, E=3, F=4

        outcome:
            到 A -> 0
            到 G -> 1

    例如：
        D -> E -> F -> G

    返回：
        sequence = [2, 3, 4]
        outcome = 1.0
    """

    # 实际位置：
    # A=0, B=1, C=2, D=3, E=4, F=5, G=6
    position = 3  # D

    sequence = []

    while position != 0 and position != 6:

        # 转换为 B~F 的 0~4 索引
        state_index = position - 1

        sequence.append(state_index)

        # 0.5 向左，0.5 向右
        if rng.random() < 0.5:
            position -= 1
        else:
            position += 1

    # 到 G 为 1，到 A 为 0
    if position == 6:
        outcome = 1.0
    else:
        outcome = 0.0

    return sequence, outcome


# ============================================================
# 3. 生成全部 training sets
# ============================================================

def generate_training_sets(
        seed=RANDOM_SEED,
        num_sets=NUM_TRAINING_SETS,
        sequences_per_set=SEQUENCES_PER_SET):
    """
    一次性生成并固定所有训练数据。

    所有 lambda、alpha 都使用同一批 training sets，
    保证比较公平。
    """

    rng = np.random.default_rng(seed)

    training_sets = []

    for _ in range(num_sets):

        one_training_set = []

        for _ in range(sequences_per_set):
            sequence = generate_sequence(rng)
            one_training_set.append(sequence)

        training_sets.append(one_training_set)

    return training_sets


# ============================================================
# 4. 单条 sequence 的 TD(lambda) 更新
# ============================================================

def train_one_sequence(w, sequence, outcome, alpha, lam):
    """
    使用论文第三章实验二的方式训练一条 sequence。

    非常重要：
    -------------------
    sequence 内不立即修改 w。

    先累计所有 delta_w，
    sequence 完成后统一：

        w <- w + delta_w

    这对应论文：
        weight updates were performed after each sequence.

    而不是第五章 5.2 的 intra-sequence updating。
    """

    # Eligibility Trace
    eligibility = np.zeros(NUM_STATES)

    # 一条序列累计的总权重变化
    total_delta_w = np.zeros(NUM_STATES)

    # --------------------------------------------------------
    # 由于 w 在整条 sequence 内不更新，
    # 所有预测都使用同一个 w。
    # --------------------------------------------------------

    for t in range(len(sequence)):

        current_state = sequence[t]

        # 当前预测
        p_t = w[current_state]

        # ----------------------------------------------------
        # 下一时刻预测
        # ----------------------------------------------------

        if t == len(sequence) - 1:

            # 最后一个非终止状态之后直接到终点
            # P_{m+1} = z
            p_next = outcome

        else:

            next_state = sequence[t + 1]

            p_next = w[next_state]

        # ----------------------------------------------------
        # TD error
        #
        # delta_t = P_{t+1} - P_t
        # ----------------------------------------------------

        td_error = p_next - p_t

        # ----------------------------------------------------
        # Eligibility Trace
        #
        # e_t = lambda * e_{t-1} + x_t
        #
        # 因为这里 x_t 是 one-hot
        # ----------------------------------------------------

        eligibility *= lam

        eligibility[current_state] += 1.0

        # ----------------------------------------------------
        # 累计本时刻产生的权重变化
        #
        # Delta w_t = alpha * delta_t * e_t
        # ----------------------------------------------------

        total_delta_w += (
            alpha
            * td_error
            * eligibility
        )

    # --------------------------------------------------------
    # 一条 sequence 完成以后统一更新
    # --------------------------------------------------------

    new_w = w + total_delta_w

    return new_w


# ============================================================
# 5. 训练一个 training set
# ============================================================

def train_one_training_set(
        training_set,
        alpha,
        lam):
    """
    一个 training set 包含 10 条 sequence。

    实验二：
    training set 只呈现一次。

    每条 sequence 后更新一次 w。
    """

    # 论文实验二规定：
    # 所有权重初始值设为 0.5
    w = np.full(
        NUM_STATES,
        0.5,
        dtype=float
    )

    for sequence, outcome in training_set:

        w = train_one_sequence(
            w=w,
            sequence=sequence,
            outcome=outcome,
            alpha=alpha,
            lam=lam
        )

    return w


# ============================================================
# 6. 计算 RMS Error
# ============================================================

def calculate_rms_error(w):
    """
    RMS Error：

    sqrt(
        mean(
            (prediction - ideal)^2
        )
    )
    """

    squared_error = (
        w - IDEAL_VALUES
    ) ** 2

    mse = np.mean(squared_error)

    rms = np.sqrt(mse)

    return rms


# ============================================================
# 7. 测试某一个 (lambda, alpha)
# ============================================================

def evaluate_lambda_alpha(
        training_sets,
        lam,
        alpha):
    """
    对 100 个 training sets 分别运行一次。

    每个 training set：
        初始化 w=0.5
        训练 10 条 sequence
        计算 RMS

    最后把 100 个 RMS 求平均。
    """

    errors = []

    for training_set in training_sets:

        w = train_one_training_set(
            training_set=training_set,
            alpha=alpha,
            lam=lam
        )

        error = calculate_rms_error(w)

        errors.append(error)

    average_error = np.mean(errors)

    return average_error


# ============================================================
# 8. 执行完整实验
# ============================================================

def run_experiment(training_sets=None):

    print("=" * 70)
    print("Sutton 1988 TD(lambda) Random Walk Experiment 2")
    print("=" * 70)

    print()
    print("Generating training sets...")

    # --------------------------------------------------------
    # 非常重要：
    #
    # training sets 只生成一次。
    #
    # 所有 lambda、alpha 使用完全相同的数据。
    # --------------------------------------------------------

    if training_sets is None:
        training_sets = generate_training_sets()

    print(
        f"Generated {NUM_TRAINING_SETS} training sets, "
        f"{SEQUENCES_PER_SET} sequences each."
    )

    print()

    # results[lambda] = 各 alpha 对应的 average RMS error
    results = {}

    total_runs = len(LAMBDAS) * len(ALPHAS)

    current_run = 0

    # --------------------------------------------------------
    # lambda × alpha 网格搜索
    # --------------------------------------------------------

    for lam in LAMBDAS:

        lambda_errors = []

        print(f"Running lambda = {lam:.1f}")

        for alpha in ALPHAS:

            error = evaluate_lambda_alpha(
                training_sets=training_sets,
                lam=lam,
                alpha=alpha
            )

            lambda_errors.append(error)

            current_run += 1

        results[lam] = np.array(lambda_errors)

    print()
    print("Experiment finished.")
    print()

    return results


# ============================================================
# 9. 打印每个 lambda 的最佳 alpha
# ============================================================

def print_best_results(results):

    print("=" * 70)
    print("Best alpha for each lambda")
    print("=" * 70)

    print(
        f"{'lambda':>10}"
        f"{'best alpha':>15}"
        f"{'best RMS':>15}"
    )

    print("-" * 40)

    best_results = {}

    for lam in LAMBDAS:

        errors = results[lam]

        best_index = np.argmin(errors)

        best_alpha = ALPHAS[best_index]

        best_error = errors[best_index]

        best_results[lam] = (
            best_alpha,
            best_error
        )

        print(
            f"{lam:>10.1f}"
            f"{best_alpha:>15.2f}"
            f"{best_error:>15.5f}"
        )

    print()

    return best_results


# ============================================================
# 10. 绘制 Figure 4 风格图
# ============================================================

def plot_figure4(results):

    plt.figure(figsize=(9, 6))

    for lam in FIGURE4_LAMBDAS:

        plt.plot(
            ALPHAS,
            results[lam],
            label=f"λ = {lam}"
        )

    plt.xlabel("Learning rate α")

    plt.ylabel("Average RMS Error")

    plt.title(
        "Random Walk: Average Error after One Training Set\n"
        "Figure 4 Style Reproduction"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_file = OUTPUT_DIR / "figure4_alpha_error.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Figure 4 style result saved to: "
        f"{output_file}"
    )




    plt.ylim(0.07, 0.30)
    plt.savefig(OUTPUT_DIR / "figure4_alpha_error_zoom.png", dpi=200,
                bbox_inches="tight")


# ============================================================
# 11. 绘制 Figure 5 风格图
# ============================================================

def plot_figure5(best_results):

    lambda_values = []

    best_errors = []

    for lam in FIGURE5_LAMBDAS:

        lambda_values.append(lam)

        best_errors.append(
            best_results[lam][1]
        )

    plt.figure(figsize=(8, 6))

    plt.plot(
        lambda_values,
        best_errors,
        marker="o"
    )

    plt.xlabel("λ")

    plt.ylabel(
        "Average RMS Error using best α"
    )

    plt.title(
        "Random Walk: Best Error for Each λ\n"
        "Figure 5 Style Reproduction"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.xticks(FIGURE5_LAMBDAS)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "figure5_lambda_best_error.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Figure 5 style result saved to: "
        f"{output_file}"
    )




# ============================================================
# 12. 可选：打印一条 Random Walk 示例
# ============================================================

def print_example_sequence():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    sequence, outcome = generate_sequence(
        rng
    )

    readable_sequence = []

    for state_index in sequence:

        readable_sequence.append(
            STATE_NAMES[state_index]
        )

    if outcome == 1:
        terminal = "G"
    else:
        terminal = "A"

    readable_sequence.append(
        terminal
    )

    print("=" * 70)

    print("Example Random Walk")

    print("=" * 70)

    print(
        " -> ".join(
            readable_sequence
        )
    )

    print(
        f"Outcome z = {outcome}"
    )

    print()


# ============================================================
# 13. 主函数
# ============================================================

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Sutton 1988 random-walk experiment 2")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print_example_sequence()
    training_sets = generate_training_sets()
    results = run_experiment(training_sets)
    best_results = print_best_results(results)
    plot_figure4(results)
    plot_figure5(best_results)
    with (OUTPUT_DIR / "all_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "alpha", "mean_rms"])
        for lam in LAMBDAS:
            writer.writerows((lam, float(a), float(e))
                             for a, e in zip(ALPHAS, results[lam]))
    with (OUTPUT_DIR / "best_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "best_alpha", "best_mean_rms"])
        writer.writerows((lam, float(a), float(e))
                         for lam, (a, e) in best_results.items())
    (OUTPUT_DIR / "training_sets.json").write_text(
        json.dumps(training_sets), encoding="utf-8")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.2f}")
    if not args.no_show:
        plt.show()
    plt.close("all")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()
