### Overview
This backtest implements a message‐by‐message Level‑1 feed (`l1_day.csv`), consolidates the best ask quote per venue at each timestamp, and simulates a limit‐only execution of 5 000 shares. That is, given a target order size, venue liquidity, and penalty parameters, it exhaustively searches feasible splits in fixed‐size chunks and selects the allocation that minimizes a cost function. A simple cost model penalizes under‐ and over‐execution to guide allocation across venues.

### Approach
1. **Data Preparation**: Deduplicate on `(ts_event, publisher_id)` to get one snapshot per venue per event. Resample into 10 ms intervals to align books.
2. **Allocation**: At each snapshot, solve a grid‐search allocation across venues that minimizes cost.
3. **Execution Loop**: Fill up to the displayed ask size at each venue, roll any unfilled shares forward, until all 5 000 shares are filled or data ends.
4. **Grid Search**: Sweep (λ_under, λ_over, θ_queue) triples to find the best parameters by savings vs. a best‐ask baseline.

### Parameter Ranges
| Parameter           | Values             |
| ------------------- | ------------------ |
| λ_under (under‐fill penalty) | [0.1, 0.8, 1.8]  |
| λ_over  (over‐fill penalty)  | [0.2, 1.0, 1.9]  |
| θ_queue (queue penalty)      | [0.01, 0.1, 0.3] |

The grid search explores 3×3×3 = 27 combinations.

### Improving Fill Realism
Fill realism models the stochastic, partial, and probabilistic nature of limit order executions. Rather than assuming deterministic full fills at posted prices, it captures uncertainties from queue positions, market order flow variability, slippage, and predictive signal adjustments—yielding more realistic backtests. 
- **Slippage & Queue Dynamics**: Incorporate a probabilistic slippage model based on historical fill rates and simulated queue position. For each limit order, estimate expected fill as a function of order arrival rank and cancellation probability, rather than assuming full displayed size availability. This would capture partial fills and dynamic queue depletion more realistically.

* **Mean Order Outflow (ξ)**: Model expected venue-level order depletion as $\xi \sim \text{Poisson}(600)$. If the allocated size exceeds ξ, only ξ shares are filled. This introduces stochastic partial fills based on historical market activity, improving realism.

To model slippage and queue dynamics more accurately, incorporate a fill probability that decays with queue depth. For each venue:

  1. **Queue Depth (`Q_k`)**: Estimate current orders ahead from Level-2 book.
  2. **Fill Probability:** `P_fill_k(x) = P(xi_k ≥ Q_k + x)` via empirical order-flow CDF.
  3. **Expected Fill:** `E[e_k] ≈ ∑_{i=1}^x P_fill_k(i)`.
  4. **Slippage:** Adjust price `p_adj = p + γ·(E[e_k]/Q_k)·h` before cost computation.

Integration: Replace deterministic `executed_k = alloc_k` with `E[e_k]` and use `p_adj` in `compute_cost` for realistic execution risk. Adding this into the cost function yields allocations that account for realistic execution risks and dynamic market impact.

### RESULTS
#### At each snapshot attempts to execute as many shares as the allocator tells to
![alt text][r1a]
![alt text][r1b]

[r1a]: https://github.com/Ojaswy/HFT_Equity_Markets/blob/main/Strategy/result1.png
[r1b]: https://github.com/Ojaswy/HFT_Equity_Markets/blob/main/Strategy/res1_json.png

#### At each snapshot attempts to execute a fixed no. of shares which are passed into the allocator
![alt text][r2a]
![alt text][r2b]
[r2a]: https://github.com/Ojaswy/HFT_Equity_Markets/blob/main/Strategy/result2.png
[r2b]: https://github.com/Ojaswy/HFT_Equity_Markets/blob/main/Strategy/res2_json.png
