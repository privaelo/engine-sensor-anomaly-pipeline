import numpy as np

def generate_windows(df, features, window_size=30, stride=1):
    windows, rul_targets = [], []
    unit_ids, end_cycles = [], []

    for unit, group in df.groupby('unit_number'):
        group = group.sort_values('time_in_cycles')
        vals = group[features].values
        ruls = group['RUL'].values
        cycles = group['time_in_cycles'].values

        for start in range(0, len(vals) - window_size + 1, stride):
            end = start + window_size
            windows.append(vals[start:end].flatten())
            rul_targets.append(ruls[end - 1])
            unit_ids.append(unit)
            end_cycles.append(cycles[end - 1])

    return np.array(windows), np.array(rul_targets), np.array(unit_ids), np.array(end_cycles)
