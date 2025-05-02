### Overview
This project implements a static allocator for limit order placement across multiple venues. Given a target order size, venue liquidity, and penalty parameters, it exhaustively searches feasible splits in fixed‐size chunks and selects the allocation that minimizes a cost function. 

### Allocation Approach
- **Chunked Search**: Orders are allocated in 100‐share increments to control combinatorial complexity.
- **Exhaustive Splits**: All possible distributions of the total size across venues, respecting each venue’s available ask size, are generated. In the pseudo-code, allocate is only distributing across venues (limit orders), hence implicitly setting M = 0. I made a modification in the loop to include Market as well as Limit Orders.
- **Cost Evaluation**: For each valid split, a cost function incorporating penalties for under‐ and over‐execution (λ_under, λ_over) and a queue position factor (θ_queue) is computed; the split with the lowest total cost is chosen.


### Parameter Ranges
- **λ_over** (overfill penalty): 0.5 – 5.0
- **λ_under** (underfill penalty): 0.5 – 5.0
- **θ_queue** (queue position weight): 0.1 – 1.0
- **Chunk Size**: Adjustable (default 100 shares)

### Improving Fill Realism
Fill realism models the stochastic, partial, and probabilistic nature of limit order executions. Rather than assuming deterministic full fills at posted prices, it captures uncertainties from queue positions, market order flow variability, slippage, and predictive signal adjustments—yielding more realistic backtests. To model slippage and queue dynamics more accurately, incorporate a fill probability that decays with queue depth. For each venue:

1. **Estimate Queue Size**: Retrieve or simulate the number of orders ahead.
2. **Probability Function**: Define \(P_{	ext{fill}}(q) = \exp(-\alpha \, q)\).
3. **Expected Fill**: Multiply chunk size by \(P_{	ext{fill}}\) and add a slippage adjustment to the price.

Integrating this into the cost function yields allocations that account for realistic execution risks and dynamic market impact.

