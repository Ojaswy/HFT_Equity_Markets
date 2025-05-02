### Overview
This project implements a static allocator for limit order placement across multiple venues. Given a target order size, venue liquidity, and penalty parameters, it exhaustively searches feasible splits in fixed‐size chunks and selects the allocation that minimizes a cost function. 

### Code Structure
- **File:** `back_testing.py`
  - **`simulate(df, λ_under, λ_over, θ)`**: Iterates over timestamps, calls `allocate()`, tracks cumulative executed shares and cash.
  - **`allocate(order_size, venues, λ_under, λ_over, θ)`**: Generates all feasible splits in fixed‐size chunks, appends a market‐order venue, and returns the split minimizing `compute_cost()`.
  - **`compute_cost(split, venues, order_size, λ_over, λ_under, θ)`**: Calculates execution cost + rebates + penalty terms (under/overfills, queue weight).


### Allocation Approach
- **Chunked Search**: Orders are allocated in 100‐share increments to control combinatorial complexity.
- **Exhaustive Splits**: All possible distributions of the total size across venues, respecting each venue’s available ask size, are generated. In the pseudo-code, allocate is only distributing across venues (limit orders), hence implicitly setting M = 0. I made a modification in the loop to include Market as well as Limit Orders.
- **Cost Evaluation**: For each valid split, a cost function incorporating penalties for under‐ and over‐execution (λ_under, λ_over) and a queue position factor (θ_queue) is computed; the split with the lowest total cost is chosen.
- **Split Generation:** Nested loops per venue, respecting each venue’s `ask_size`, plus a synthetic market‐order leg at `min_ask + SPREAD`.

### Parameter Ranges
- **λ_over** (overfill penalty): 0.5 – 5.0
- **λ_under** (underfill penalty): 0.5 – 5.0
- **θ_queue** (queue position weight): 0.1 – 1.0
- **Chunk Size**: Adjustable (default 100 shares)

### Improving Fill Realism
Fill realism models the stochastic, partial, and probabilistic nature of limit order executions. Rather than assuming deterministic full fills at posted prices, it captures uncertainties from queue positions, market order flow variability, slippage, and predictive signal adjustments—yielding more realistic backtests. To model slippage and queue dynamics more accurately, incorporate a fill probability that decays with queue depth. For each venue:

  1. **Queue Depth (`Q_k`)**: Estimate current orders ahead from Level-2 book.
  2. **Fill Probability:** `P_fill_k(x) = P(xi_k ≥ Q_k + x)` via empirical order-flow CDF.
  3. **Expected Fill:** `E[e_k] ≈ ∑_{i=1}^x P_fill_k(i)`.
  4. **Slippage:** Adjust price `p_adj = p + γ·(E[e_k]/Q_k)·h` before cost computation.

Integration: Replace deterministic `executed_k = alloc_k` with `E[e_k]` and use `p_adj` in `compute_cost` for realistic execution risk. Adding this into the cost function yields allocations that account for realistic execution risks and dynamic market impact.

