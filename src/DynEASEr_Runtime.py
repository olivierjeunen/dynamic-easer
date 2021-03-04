import argparse
import numpy as np
import pandas as pd
import time

from datetime import timedelta
from tqdm import tqdm

from util import compute_k_core, EASEr, encode_integer_id, generate_csr, incremental_updates, print_summary

if __name__=='__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_path', type = str, help = 'Number of days to incrementally update')
    parser.add_argument('-days', '--num_days', type = int, help = 'Number of days to incrementally update', default = 6 * 31)
    parser.add_argument('-mins', '--update_minutes', type = int, help = 'Number of minutes between incremental updates', default = 60 * 48) 
    parser.add_argument('-u_k', '--user_k', type = int, help = 'Number of minimal occurrences to keep user', default = 3)
    parser.add_argument('-i_k', '--item_k', type = int, help = 'Number of minimal occurrences to keep item', default = 3)
    args = parser.parse_args()    

    # Parse clicks and deduplicate
    clicks = pd.read_csv(
        args.data_path,
        dtype = {
            'userId': int,
            'movieId': int,
            'rating': float,
            'timestamp': int,
        }
    )
    clicks.columns = ['session','item','rating','timestamp']
    clicks = clicks.loc[clicks.rating > 3.0]
    clicks.drop_duplicates(['session','item'], inplace = True)

    print_summary(clicks, ['session', 'item'])

    # Drop end-of-tail users and items
    print('Computing k-core...')
    clicks = compute_k_core(clicks, user_col='session', item_col='item', user_k=args.user_k, item_k=args.item_k, i=1)
    print_summary(clicks, ['session', 'item'])

    # Ensure integer IDs
    for col in ['session','item']:
        clicks[col] = encode_integer_id(clicks[col])

    # Parse timestamps, sort
    clicks['timestamp'] = pd.to_datetime(clicks['timestamp'] * 1000000000)
    clicks = clicks.sort_values('timestamp').reset_index(drop=True)
    print(f"Dataset spans {clicks['timestamp'].min()} - {clicks['timestamp'].max()} ({clicks['timestamp'].max() - clicks['timestamp'].min()})...")

    # Collect data for all but N days
    init_ts = clicks.timestamp.max() - timedelta(days=args.num_days)
    init_df = clicks.loc[clicks.timestamp < init_ts]
    print(f'Training EASEr on {len(init_df)} initial points...')

    # Turn into csr_matrix
    n_users, n_items = clicks['session'].max() + 1, clicks['item'].max() + 1
    X_init = generate_csr(init_df, (n_users, n_items))

    # Train initial EASEr model
    print('Training initial EASEr model...')
    s = time.perf_counter()
    G, S = EASEr(X_init)
    e = time.perf_counter()
    full_time = e - s
    print(f'Trained EASEr in {full_time} seconds!')

    #############
    # Dyn-EASEr #
    #############
    timestamps, timings, ranks = incremental_updates(clicks,
                                                     X_init,
                                                     G,
                                                     S,
                                                     init_ts,
                                                     args.num_days,
                                                     args.update_minutes,
                                                     rank='user_bound'
                                                     )

    # Write out results
    out = pd.DataFrame({'ts': [init_ts] + timestamps, 'rt': [full_time] + timings, 'rk': [-1] + ranks})
    out.to_csv(f'Results_{args.user_k}_U_{args.item_k}_I_core_last_{args.num_days}_days_{args.update_minutes}_min_updates.csv',index=False)
