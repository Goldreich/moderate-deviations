import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import jit
import time

# --- JIT-Compiled Worker (Standard Regime Only) ---
@jit(nopython=True)
def fast_sgd_standard(seed, n, alpha, beta, gamma_0, x_star_arr, sigma_noise):
    np.random.seed(seed)
    
    # 1. Block Parameters
    m = int(np.floor(n**alpha))      
    k = int(np.floor(n / (2 * m)))   
    
    true_intercept = x_star_arr[0]
    true_slope = x_star_arr[1]
    
    # SGD State
    x_t_0 = 0.0
    x_t_1 = 0.0
    
    # Statistics Storage
    # We sum squares and raw values on the fly to save memory
    total_sum_Y = 0.0
    sum_sq_Y = 0.0
    
    global_step = 1
    
    # --- CHUNKING LOOP ---
    for block_idx in range(2 * k):
        # Generate small chunk of data (fits in CPU cache)
        feature_x = np.random.randn(m)
        epsilon = np.random.randn(m) * sigma_noise
        
        current_block_sum = 0.0
        
        # Inner SGD Loop
        for i in range(m):
            gamma = gamma_0 * (global_step ** -beta)
            
            # Prediction
            pred = x_t_0 + x_t_1 * feature_x[i]
            y_true = true_intercept + true_slope * feature_x[i] + epsilon[i]
            error = y_true - pred
            
            # Updates
            x_t_0 += gamma * error
            x_t_1 += gamma * error * feature_x[i]
            
            # Accumulate Slope Error for this block
            current_block_sum += (x_t_1 - true_slope)
            global_step += 1
            
        # Accumulate Block Stats immediately
        total_sum_Y += current_block_sum
        sum_sq_Y += current_block_sum**2

    # --- Compute Final Statistic ---
    V = np.sqrt(sum_sq_Y)
    
    if V == 0:
        return 0.0
        
    return total_sum_Y / V

def simulation_final_scale():
    # --- Parameters ---
    n = 1e8
    n_sims = 1000
    
    x_star = np.array([2.0, -3.0])
    sigma_noise = 2.0
    
    beta = 0.6
    alpha = 0.8
    gamma_0 = 0.1
    
    print(f"--- Final Scale Simulation ---")
    print(f"Steps (n): {n:.0e}")
    print(f"Alpha: {alpha}")
    
    start_time = time.time()
    
    seeds = np.random.randint(0, 1e9, size=n_sims)
    
    results = Parallel(n_jobs=-3, verbose=5)(
        delayed(fast_sgd_standard)(
            seed, n, alpha, beta, gamma_0, x_star, sigma_noise
        ) for seed in seeds
    )
    
    I_n_samples = np.array(results)
    
    duration = time.time() - start_time
    print(f"Completed in {duration:.1f} seconds")

    # --- PLOT 1: Histogram ---
    plt.figure(figsize=(6, 5))
    plt.hist(I_n_samples, bins=50, density=True, alpha=0.6, color='navy', label='Slope $I_n$')
    
    # Overlay Normal Curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='Standard Normal')
    
    plt.title(f'Distribution ($n=10^8$, $\\alpha=0.8$)')
    plt.xlabel('$I_n$')
    plt.legend()
    plt.tight_layout()
    
    hist_filename = "histogram_n100m.png"
    plt.savefig(hist_filename, dpi=300)
    print(f"Saved histogram to {hist_filename}")
    plt.close() # Close to free memory

    # --- PLOT 2: QQ Plot ---
    plt.figure(figsize=(6, 5))
    stats.probplot(I_n_samples, dist="norm", plot=plt)
    
    plt.title('QQ Plot: Asymptotic Check')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    qq_filename = "qq_plot_n100m.png"
    plt.savefig(qq_filename, dpi=300)
    print(f"Saved QQ plot to {qq_filename}")
    plt.close()

if __name__ == "__main__":
    # Ensure fast_sgd_standard is defined before running
    simulation_final_scale()