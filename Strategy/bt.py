import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import product

DATA_FILE     = 'l1_day.csv'
TOTAL_SHARES  = 5000
STEP_SIZE     = 100

DEFAULT_FEE   = 0.003
DEFAULT_REBATE= 0.00
SPREAD        = 0.01

LAMBDA_UNDER_VAL = [0.1, 0.8, 1.8]
LAMBDA_OVER_VAL  = [0.2, 1.0, 1.9]
THETA_VAL        = [0.01, 0.1, 0.3]

def compute_cost(split, venues, order_size, lam_over, lam_under, theta):
    executed = cash = 0.0
    for alloc, v in zip(split, venues):
        exe    = min(alloc, v['ask_size'])
        executed += exe
        cash     += exe * (v['ask'] + v['fee'])
        missed   = max(alloc - exe, 0)
        cash    -= missed * v['rebate']
    under = max(order_size - executed, 0)
    over  = max(executed - order_size, 0)
    return cash + lam_under * under + lam_over * over + theta * (under + over)

def allocate(order_size, venues, lam_over, lam_under, theta):
    step = STEP_SIZE
    splits = [[]]
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v]['ask_size'])
            for q in range(0, int(max_v)+1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float('inf')
    best_split = []
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lam_over, lam_under, theta)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    return best_split, best_cost

def simulate(df, lam_under, lam_over, theta):
    rem  = TOTAL_SHARES
    cash = 0.0
    rows = []
    for ts, grp in df.groupby('ts_event'):
        if rem <= 0:
            break
        venues = [
            {'ask': r['ask_px_00'], 'ask_size': r['ask_sz_00'],
             'fee': DEFAULT_FEE, 'rebate': DEFAULT_REBATE}
            for _, r in grp.iterrows()
        ]
        # Uncomment these two lines for only executing STEP_SIZE no. of shares at each timestamp
        #order_size = min(STEP_SIZE, rem)
        #split, _ = allocate(order_size, venues, lam_over, lam_under, theta)

        split, _ = allocate(rem, venues, lam_over, lam_under, theta)
        
        limit_exe  = sum(min(alloc, v['ask_size']) for alloc, v in zip(split, venues))
        limit_cost = sum(
            min(alloc, v['ask_size'])*(v['ask']+v['fee']) 
            - max(alloc - v['ask_size'], 0)*v['rebate']
            for alloc, v in zip(split, venues)
        )
        rem  -= limit_exe
        cash += limit_cost
        rows.append((ts, limit_exe, cash))

    return rows

def compute_baselines(df):
    rem = TOTAL_SHARES
    cost = 0.0
    asks = []
    vols = []
    for _, grp in df.groupby('ts_event'):
        row = grp.loc[grp['ask_px_00'].idxmin()]
        p   = row['ask_px_00']
        sz  = row['ask_sz_00']
        px  = p + DEFAULT_FEE + SPREAD
        asks.append(px)
        vols.append(sz)
        buy = min(rem, sz)
        cost += buy * px
        rem -= buy
    if rem > 0 and asks:
        cost += rem * asks[-1]
    best_p = cost / TOTAL_SHARES if TOTAL_SHARES else np.nan
    twap   = np.mean(asks) if asks else np.nan
    vwap   = np.dot(asks, vols)/sum(vols) if sum(vols) else np.nan
    return best_p, twap, vwap

def plot_limit_exec_cost(rows, title="Limit-Only Execution Profile"):
    df = pd.DataFrame(rows, columns=['ts_event','limit_executed','cum_cost'])
    df['ts_event'] = pd.to_datetime(df['ts_event'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna().sort_values('ts_event')

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.step(df['ts_event'], df['limit_executed'], where='post',
             label='Limit Order Shares', linewidth=2)
    ax1.set_xlabel('Event Time')
    ax1.set_ylabel('Shares Executed (Limit Only)')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    ax2 = ax1.twinx()
    ax2.plot(df['ts_event'], df['cum_cost'],
             label='Cumulative Cost', color='tab:green',
             marker='o', linewidth=2)
    ax2.set_ylabel('Cumulative Cost ($)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values(['ts_event', 'publisher_id']) \
           .drop_duplicates(['ts_event', 'publisher_id'])
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    # Sets granularity of the order book data (set relative to STEP_SIZE)
    df = df.set_index('ts_event').resample('10ms').last().dropna().reset_index()

    best_p, twap_p, vwap_p = compute_baselines(df)
    best_res = None

    for lu, lo, th in product(LAMBDA_UNDER_VAL, LAMBDA_OVER_VAL, THETA_VAL):
        rows = simulate(df, lu, lo, th)
        df_exec = pd.DataFrame(rows, columns=['ts_event','limit_executed','cum_cost'])
        total_cash = df_exec['cum_cost'].iloc[-1]
        total_exe  = df_exec['limit_executed'].sum()
        avg_price  = total_cash / total_exe if total_exe else np.nan
        bps        = (best_p - avg_price) / best_p * 1e4 if best_p else 0

        if best_res is None or bps > best_res['bps']:
            best_res = {
                'lu': lu, 'lo': lo, 'th': th,
                'bps': bps,
                'total_cash': total_cash,
                'avg_price': avg_price,
                'rows': rows
            }

    summary = {
        'best_parameters': {
            'lambda_under': best_res['lu'],
            'lambda_over':  best_res['lo'],
            'theta_queue':  best_res['th']
        },
        'our_strategy': {
            'total_cash_spent': best_res['total_cash'],
            'avg_share_price' : best_res['avg_price']
        },
        'baselines': {
            'best_ask': {'avg_price': best_p},
            'TWAP':     {'avg_price': twap_p},
            'VWAP':     {'avg_price': vwap_p}
        },
        'savings_bps': {
            'vs_best_ask': (best_p - best_res['avg_price'])/best_p*1e4 if best_p else 0,
            'vs_TWAP':     (twap_p - best_res['avg_price'])/twap_p*1e4 if twap_p else 0,
            'vs_VWAP':     (vwap_p - best_res['avg_price'])/vwap_p*1e4 if vwap_p else 0
        }
    }
    print(json.dumps(summary, indent=2))

    plot_limit_exec_cost(best_res['rows'],
        title=f"Limit-Only Exec Profile (λ_u={best_res['lu']}, λ_o={best_res['lo']}, θ={best_res['th']})")

if __name__ == '__main__':
    main()
