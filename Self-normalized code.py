import numpy as np
from numba import njit, prange
from scipy.stats import chi2
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Simulation parameters
# ============================================================

D_GRID = np.array([2, 5], dtype=np.int64)
B = 1_000
N = 50_000_000
BETA_VALUES = np.array([0.505], dtype=np.float64)
ALPHA_BLOCKS = np.array([0.70], dtype=np.float64)
SIGMA = 1.0
GAMMA0 = 0.1

ALPHA_CONF_GRID = np.array([0.20, 0.10, 0.05, 0.01, 0.005], dtype=np.float64)
PLOT_ALPHA_CONF_GRID = np.array([0.20, 0.10, 0.05], dtype=np.float64)

BASE_SEED = 123456789


# ============================================================
# Core numba simulation
# ============================================================

@njit
def _normal_vec(d):
    out = np.empty(d)
    for i in range(d):
        out[i] = np.random.normal()
    return out


@njit
def _outer_add(A, v):
    d = v.shape[0]
    for i in range(d):
        for j in range(d):
            A[i, j] += v[i] * v[j]


@njit
def _quad_form_inv(A, s):
    """
    Compute s^T A^{-1} s.
    Uses np.linalg.solve inside numba.
    """
    z = np.linalg.solve(A, s)
    return np.dot(s, z)


@njit
def _one_replication(d, n, beta, alpha_block, sigma, gamma0, seed):
    """
    One Monte Carlo replication.

    Model:
        a_t ~ N(0, I_d)
        eps_t ~ N(0, sigma^2)
        y_t = a_t^T x_star + eps_t
        x_t = x_{t-1} - gamma_t a_t(a_t^T x_{t-1} - y_t)

    Returns:
        I_even_sq, I_even_emp_sq
    """
    np.random.seed(seed)

    m = int(np.floor(n ** alpha_block))
    k = int(np.floor(n / (2.0 * m)))
    n_used = 2 * k * m

    # x_star = (1, 1/2, ..., 1/d)
    x_star = np.empty(d)
    for i in range(d):
        x_star[i] = 1.0 / (i + 1.0)

    # Initialize SGD at zero.
    x = np.zeros(d)

    # Store only the even block sums Y_{2j}.
    Y_even = np.zeros((k, d))

    block_sum = np.zeros(d)
    block_index = 0
    even_index = 0
    within_block = 0

    for t in range(1, n_used + 1):
        gamma_t = gamma0 * (t ** (-beta))

        a = _normal_vec(d)
        eps = sigma * np.random.normal()

        # y = a^T x_star + eps
        y = np.dot(a, x_star) + eps

        # gradient = a * (a^T x - y)
        resid = np.dot(a, x) - y
        for q in range(d):
            x[q] -= gamma_t * a[q] * resid

        # Add x_t - x_star to current block sum.
        for q in range(d):
            block_sum[q] += x[q] - x_star[q]

        within_block += 1

        if within_block == m:
            block_index += 1

            # Keep even blocks: block_index = 2,4,...,2k.
            if block_index % 2 == 0:
                for q in range(d):
                    Y_even[even_index, q] = block_sum[q]
                even_index += 1

            # Reset block.
            for q in range(d):
                block_sum[q] = 0.0
            within_block = 0

    # S_even = sum_j Y_{2j}
    S_even = np.zeros(d)
    for j in range(k):
        for q in range(d):
            S_even[q] += Y_even[j, q]

    # V_even^2 = sum_j Y_{2j}Y_{2j}^T
    V_even = np.zeros((d, d))
    for j in range(k):
        _outer_add(V_even, Y_even[j])

    I_even_sq = _quad_form_inv(V_even, S_even)

    # Empirical centering:
    # Y_{2j,emp} = Y_{2j} - S_even/k
    S_over_k = np.empty(d)
    for q in range(d):
        S_over_k[q] = S_even[q] / k

    V_even_emp = np.zeros((d, d))
    temp = np.empty(d)

    for j in range(k):
        for q in range(d):
            temp[q] = Y_even[j, q] - S_over_k[q]
        _outer_add(V_even_emp, temp)

    I_even_emp_sq = _quad_form_inv(V_even_emp, S_even)

    return I_even_sq, I_even_emp_sq


def run_replications_for_d(d, B, n, beta, alpha_block, sigma, gamma0, base_seed):
    I_even_sq = np.empty(B)
    I_even_emp_sq = np.empty(B)

    for b in tqdm(range(B), desc=f"Replications for d={d}", leave=False):
        seed = base_seed + 1000003 * d + b
        out1, out2 = _one_replication(d, n, beta, alpha_block, sigma, gamma0, seed)
        I_even_sq[b] = out1
        I_even_emp_sq[b] = out2

    return I_even_sq, I_even_emp_sq


# ============================================================
# Summaries
# ============================================================

def summarize_for_d(d, I_even_sq, I_even_emp_sq, alpha_grid):
    rows = []

    for aconf in alpha_grid:
        q = chi2.ppf(1.0 - aconf, df=d)

        oracle_miss = np.mean(I_even_sq > q)
        emp_miss = np.mean(I_even_emp_sq > q)
        emp_cov = 1.0 - emp_miss

        oracle_rel_error = abs(oracle_miss / aconf - 1.0)
        emp_rel_error = abs(emp_miss / aconf - 1.0)

        # Monte Carlo SE for miss probability estimate.
        oracle_se = np.sqrt(max(oracle_miss * (1.0 - oracle_miss), 0.0) / len(I_even_sq))
        emp_se = np.sqrt(max(emp_miss * (1.0 - emp_miss), 0.0) / len(I_even_emp_sq))

        rows.append({
            "d": d,
            "alpha_conf": aconf,
            "chisq_quantile": q,
            "oracle_miss_rate": oracle_miss,
            "oracle_miss_se": oracle_se,
            "oracle_relative_tail_error": oracle_rel_error,
            "empirical_miss_rate": emp_miss,
            "empirical_miss_se": emp_se,
            "empirical_coverage": emp_cov,
            "nominal_coverage": 1.0 - aconf,
            "empirical_relative_tail_error": emp_rel_error,
        })

    return pd.DataFrame(rows)


def ks_distance_to_chisq(samples, d):
    vals = np.sort(samples)
    B = len(vals)
    empirical_cdf = np.arange(1, B + 1) / B
    theoretical_cdf = chi2.cdf(vals, df=d)
    return np.max(np.abs(empirical_cdf - theoretical_cdf))


def tail_ratio_grid(samples, d, alpha_grid):
    rows = []
    for aconf in alpha_grid:
        q = chi2.ppf(1.0 - aconf, df=d)
        empirical_tail = np.mean(samples > q)
        rows.append({
            "d": d,
            "alpha_conf": aconf,
            "x": np.sqrt(q),
            "q_chisq": q,
            "empirical_tail": empirical_tail,
            "tail_ratio": empirical_tail / aconf,
        })
    return pd.DataFrame(rows)


# ============================================================
# Main run
# ============================================================

def main():
    print("Simulation parameters")
    print(f"B={B}")
    print(f"N requested={N}")
    print(f"alpha_block values={ALPHA_BLOCKS.tolist()}")
    print(f"beta values={BETA_VALUES.tolist()}")
    print(f"gamma0={GAMMA0}")
    print(f"sigma={SIGMA}")
    print()

    all_summary = []
    all_tail_oracle = []
    all_tail_emp = []
    ks_rows = []

    for beta in BETA_VALUES:
        for alpha_block in ALPHA_BLOCKS:
            m = int(np.floor(N ** alpha_block))
            k = int(np.floor(N / (2.0 * m)))
            n_used = 2 * k * m

            print("----------------------------------------")
            print(f"Running beta={beta}, alpha_block={alpha_block}")
            print(f"m=floor(N^alpha)={m}")
            print(f"k=floor(N/(2m))={k}")
            print(f"n_used=2*k*m={n_used}")
            print(f"unused observations={N - n_used}")
            print()

            for d in tqdm(D_GRID, desc=f"beta={beta}, alpha={alpha_block}"):
                start = time.time()

                I_even_sq, I_even_emp_sq = run_replications_for_d(
                    d=d,
                    B=B,
                    n=N,
                    beta=beta,
                    alpha_block=alpha_block,
                    sigma=SIGMA,
                    gamma0=GAMMA0,
                    base_seed=BASE_SEED
                )

                elapsed = time.time() - start
                print(f"d={d} finished in {elapsed:.2f} seconds")

                summary = summarize_for_d(d, I_even_sq, I_even_emp_sq, ALPHA_CONF_GRID)
                summary["alpha_block"] = alpha_block
                summary["beta"] = beta
                all_summary.append(summary)

                tail_oracle = tail_ratio_grid(I_even_sq, d, ALPHA_CONF_GRID)
                tail_oracle["statistic"] = "I_even_sq"
                tail_oracle["alpha_block"] = alpha_block
                tail_oracle["beta"] = beta
                all_tail_oracle.append(tail_oracle)

                tail_emp = tail_ratio_grid(I_even_emp_sq, d, ALPHA_CONF_GRID)
                tail_emp["statistic"] = "I_even_emp_sq"
                tail_emp["alpha_block"] = alpha_block
                tail_emp["beta"] = beta
                all_tail_emp.append(tail_emp)

                ks_rows.append({
                    "d": d,
                    "alpha_block": alpha_block,
                    "beta": beta,
                    "ks_I_even_sq_vs_chisq": ks_distance_to_chisq(I_even_sq, d),
                    "ks_I_even_emp_sq_vs_chisq": ks_distance_to_chisq(I_even_emp_sq, d),
                    "mean_I_even_sq": np.mean(I_even_sq),
                    "mean_I_even_emp_sq": np.mean(I_even_emp_sq),
                    "chisq_mean": d,
                    "var_I_even_sq": np.var(I_even_sq),
                    "var_I_even_emp_sq": np.var(I_even_emp_sq),
                    "chisq_var": 2 * d,
                })

                # Save raw samples per d in case plotting is done later.
                np.savez(
                    f"sgd_selfnorm_beta{beta:.3f}_alpha{alpha_block:.2f}_d{d}_B{B}_N{N}.npz",
                    I_even_sq=I_even_sq,
                    I_even_emp_sq=I_even_emp_sq,
                    d=d,
                    B=B,
                    N=N,
                    beta=beta,
                    alpha_block=alpha_block,
                    sigma=SIGMA,
                    gamma0=GAMMA0,
                    m=m,
                    k=k,
                    n_used=n_used,
                    alpha_conf_grid=ALPHA_CONF_GRID
                )

    summary_df = pd.concat(all_summary, ignore_index=True)
    tail_df = pd.concat(all_tail_oracle + all_tail_emp, ignore_index=True)
    ks_df = pd.DataFrame(ks_rows)

    summary_df.to_csv("coverage_summary.csv", index=False)
    tail_df.to_csv("tail_ratios.csv", index=False)
    ks_df.to_csv("ks_summary.csv", index=False)

    print()
    print("Coverage summary")
    print(summary_df)
    print()
    print("KS summary")
    print(ks_df)


def plot_qq(d, beta, alpha_block):
    filename = f"sgd_selfnorm_beta{beta:.3f}_alpha{alpha_block:.2f}_d{d}_B{B}_N{N}.npz"
    data = np.load(filename)
    samples = np.sort(data["I_even_sq"])
    B_local = len(samples)

    probs = (np.arange(1, B_local + 1) - 0.5) / B_local
    theo = chi2.ppf(probs, df=d)

    plt.figure()
    plt.scatter(theo, samples, s=8)
    max_val = max(np.max(theo), np.max(samples))
    plt.plot([0, max_val], [0, max_val])
    plt.xlabel(r"$\chi^2(d)$ theoretical quantiles")
    plt.ylabel(r"Empirical quantiles of $I_{n,\mathrm{even}}^2$")
    plt.tight_layout()
    plt.savefig(f"qq_I_even_sq_alpha{alpha_block:.2f}_d{d}.png", dpi=200)
    plt.close()


def plot_coverage():
    all_summary = []
    for beta in BETA_VALUES:
        for alpha_block in ALPHA_BLOCKS:
            for d in D_GRID:
                filename = f"sgd_selfnorm_beta{beta:.3f}_alpha{alpha_block:.2f}_d{d}_B{B}_N{N}.npz"
                data = np.load(filename)
                I_even_sq = data["I_even_sq"]
                I_even_emp_sq = data["I_even_emp_sq"]
                summary = summarize_for_d(d, I_even_sq, I_even_emp_sq, PLOT_ALPHA_CONF_GRID)
                summary["alpha_block"] = alpha_block
                summary["beta"] = beta
                all_summary.append(summary)
    df = pd.concat(all_summary, ignore_index=True)
    df["coverage_deviation_ratio"] = (df["empirical_coverage"] - df["nominal_coverage"]) / df["alpha_conf"]
    df["coverage_se_ratio"] = df["empirical_miss_se"] / df["alpha_conf"]

    alpha_blocks = sorted(df["alpha_block"].unique())
    n_alpha = len(alpha_blocks)
    fig, axes = plt.subplots(1, n_alpha, figsize=(5 * n_alpha, 4), sharey=True)
    if n_alpha == 1:
        axes = [axes]

    for ax, alpha_block in zip(axes, alpha_blocks):
        sub = df[df["alpha_block"] == alpha_block]
        for d in sorted(sub["d"].unique()):
            sub_d = sub[sub["d"] == d]
            ax.errorbar(
                sub_d["alpha_conf"],
                sub_d["coverage_deviation_ratio"],
                yerr=1.96 * sub_d["coverage_se_ratio"],
                marker="o",
                capsize=3,
                label=f"d={d}"
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha_{\mathrm{conf}}$")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        if ax is axes[0]:
            ax.set_ylabel("(Empirical - Nominal Coverage) / Nominal Tail")
        ax.legend()

    fig.tight_layout()
    plt.savefig("coverage_deviation_ratio_by_alpha.png", dpi=200)
    plt.close()


def plot_tail_ratios():
    all_tail = []
    for beta in BETA_VALUES:
        for alpha_block in ALPHA_BLOCKS:
            for d in D_GRID:
                filename = f"sgd_selfnorm_beta{beta:.3f}_alpha{alpha_block:.2f}_d{d}_B{B}_N{N}.npz"
                data = np.load(filename)
                I_even_sq = data["I_even_sq"]
                I_even_emp_sq = data["I_even_emp_sq"]
                tail_oracle = tail_ratio_grid(I_even_sq, d, PLOT_ALPHA_CONF_GRID)
                tail_oracle["statistic"] = "I_even_sq"
                tail_oracle["alpha_block"] = alpha_block
                tail_oracle["beta"] = beta
                tail_emp = tail_ratio_grid(I_even_emp_sq, d, PLOT_ALPHA_CONF_GRID)
                tail_emp["statistic"] = "I_even_emp_sq"
                tail_emp["alpha_block"] = alpha_block
                tail_emp["beta"] = beta
                all_tail.extend([tail_oracle, tail_emp])
    df = pd.concat(all_tail, ignore_index=True)

    for d in D_GRID:
        plt.figure()
        for stat in ["I_even_sq", "I_even_emp_sq"]:
            for alpha_block in sorted(df["alpha_block"].unique()):
                sub = df[(df["d"] == d) & (df["statistic"] == stat) & (df["alpha_block"] == alpha_block)].copy()
                plt.plot(sub["alpha_conf"], sub["tail_ratio"], marker="o", label=f"{stat}, alpha={alpha_block:.2f}")

        plt.axhline(1.0, linestyle="--")
        plt.xscale("log")
        plt.xlabel(r"$\alpha_{\mathrm{conf}}$")
        plt.ylabel("Empirical tail / nominal tail")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(f"tail_ratios_d{d}.png", dpi=200)
        plt.close()

        for beta in sorted(df["beta"].unique()):
            for alpha_block in sorted(df["alpha_block"].unique()):
                filename = f"sgd_selfnorm_beta{beta:.3f}_alpha{alpha_block:.2f}_d{d}_B{B}_N{N}.npz"
                data = np.load(filename)
                samples = data["I_even_sq"]
                xmax = np.quantile(samples, 0.995)
                xgrid = np.linspace(0, xmax, 500)

                plt.figure()
                plt.hist(samples, bins=60, density=True, alpha=0.35, label=r"Empirical $I_{n,\mathrm{even}}^2$")
                plt.plot(xgrid, chi2.pdf(xgrid, df=d), linewidth=2, label=rf"$\chi^2({d})$ density")
                plt.xlabel(r"$x$")
                plt.ylabel("Density")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"density_I_even_sq_beta{beta:.3f}_alpha{alpha_block:.2f}_d{d}.png", dpi=200)
                plt.close()

if __name__ == "__main__":
    main()

    for beta in BETA_VALUES:
        for alpha_block in ALPHA_BLOCKS:
            for d in D_GRID:
                plot_qq(d, beta, alpha_block)

    plot_coverage()
    plot_tail_ratios()