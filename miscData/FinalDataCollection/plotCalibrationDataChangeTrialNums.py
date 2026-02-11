import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, FormatStrFormatter
import numpy as np
import os

# Define trial groups with their measurements per iteration
# Each group: (list of filenames, measurements_per_iteration, group_label)
trial_groups = [
    {
        'filenames': [
            'P4Q12Trial1.csv',
            'P4Q12Trial2.csv',
            'P4Q12Trial3.csv',
            'P4Q12Trial4.csv',
        ],
        'measurements_per_iter': 4,
        'label': '4 measurements/iter',
        'color': '#e41a1c',  # red
    },
    {
        'filenames': [
            'successfulCalibration9.csv',
            'successfulCalibration10.csv',
            'successfulCalibration11.csv',
            'successfulCalibration12.csv',
            'successfulCalibration13.csv',
            'successfulCalibration14.csv',
            'successfulCalibration15.csv',
            'successfulCalibration16.csv',
        ],
        'measurements_per_iter': 8,
        'label': '8 measurements/iter',
        'color': '#377eb8',  # blue
    },
    {
        'filenames': [
            'P12Q4Trial1.csv',
            'P12Q4Trial2.csv',
            'P12Q4Trial3.csv',
            'P12Q4Trial4.csv',
        ],
        'measurements_per_iter': 12,
        'label': '12 measurements/iter',
        'color': '#4daf4a',  # green
    },
]

# Load all CSV files into their groups
loaded_groups = []
for group in trial_groups:
    group_data = []
    for filename in group['filenames']:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"Loaded {filename}")
            group_data.append((filename, df))
        else:
            print(f"Warning: {filename} not found, skipping...")
    if group_data:
        loaded_groups.append({
            'data': group_data,
            'measurements_per_iter': group['measurements_per_iter'],
            'label': group['label'],
            'color': group['color'],
        })

if not loaded_groups:
    print("No data files found!")
    exit()

# Prepare output image filename base
base_name = "combined_calibration_by_measurements"

# Compute x offsets within each group so trials don't overlap
# Also compute a global offset per group so groups don't overlap
group_offset_range = 0.3  # total spread around each measurement count tick
group_offsets = np.linspace(-group_offset_range / 2, group_offset_range / 2, len(loaded_groups))

# ============================================================
# Plot position error values (logarithmic y-axis)
# ============================================================
plt.figure()
min_vals = []
legend_handles = []
legend_added = {}

for g_idx, group in enumerate(loaded_groups):
    meas_per_iter = group['measurements_per_iter']
    color = group['color']
    n_trials = len(group['data'])
    if n_trials > 1:
        trial_offsets = np.linspace(-0.1, 0.1, n_trials)
    else:
        trial_offsets = [0.0]

    for t_idx, (filename, df) in enumerate(group['data']):
        # X axis = cumulative total measurements = (iteration + 1) * measurements_per_iter
        x_vals = (df.index + 1) * meas_per_iter + trial_offsets[t_idx]
        mean_err = df['Position Error'] * 1000
        std_err = df['Position Error Std'] * 1000 if 'Position Error Std' in df.columns else None

        # Only add label for the first trial in each group
        label = group['label'] if group['label'] not in legend_added else None
        if label:
            legend_added[group['label']] = True

        if std_err is not None:
            line = plt.errorbar(x_vals, mean_err, yerr=std_err, label=label, color=color,
                         linewidth=1.5, capsize=3, capthick=1.2, elinewidth=1.0, fmt='-o', markersize=4, alpha=0.7)
        else:
            line = plt.plot(x_vals, mean_err, label=label, color=color, linewidth=1.5, marker='o', markersize=4, alpha=0.7)

        pos_errors = df['Position Error'] * 1000
        min_val = pos_errors[pos_errors > 0].min()
        if min_val is not None and min_val > 0:
            min_vals.append(min_val)

plt.yscale('log')
plt.title('Position Error vs Total Measurements', fontsize=16)
plt.xlabel('Total Measurements', fontsize=16)
plt.ylabel('Position Error [mm]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, which='both', axis='y')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

if min_vals:
    overall_min = min(min_vals)
    lower_lim = 10**(np.log10(overall_min) - 0.5)
    plt.ylim(bottom=lower_lim)

ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
ax.tick_params(axis='y', which='minor', labelsize=8)
plt.tight_layout()
plt.savefig(f"{base_name}_position_error_log.png", dpi=300)

# ============================================================
# Plot position error values (linear y-axis)
# ============================================================
plt.figure()
legend_added = {}
for g_idx, group in enumerate(loaded_groups):
    meas_per_iter = group['measurements_per_iter']
    color = group['color']
    n_trials = len(group['data'])
    trial_offsets = np.linspace(-0.1, 0.1, n_trials) if n_trials > 1 else [0.0]

    for t_idx, (filename, df) in enumerate(group['data']):
        x_vals = (df.index + 1) * meas_per_iter + trial_offsets[t_idx]
        mean_err = df['Position Error'] * 1000
        std_err = df['Position Error Std'] * 1000 if 'Position Error Std' in df.columns else None
        label = group['label'] if group['label'] not in legend_added else None
        if label:
            legend_added[group['label']] = True

        if std_err is not None:
            plt.errorbar(x_vals, mean_err, yerr=std_err, label=label, color=color,
                         linewidth=1.5, capsize=3, capthick=1.2, elinewidth=1.0, fmt='-o', markersize=4, alpha=0.7)
        else:
            plt.plot(x_vals, mean_err, label=label, color=color, linewidth=1.5, marker='o', markersize=4, alpha=0.7)

plt.title('Position Error vs Total Measurements', fontsize=16)
plt.xlabel('Total Measurements', fontsize=16)
plt.ylabel('Position Error [mm]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"{base_name}_position_error_linear.png", dpi=300)

# ============================================================
# Plot orientation error values (logarithmic y-axis)
# ============================================================
plt.figure()
min_vals = []
legend_added = {}
for g_idx, group in enumerate(loaded_groups):
    meas_per_iter = group['measurements_per_iter']
    color = group['color']
    n_trials = len(group['data'])
    trial_offsets = np.linspace(-0.1, 0.1, n_trials) if n_trials > 1 else [0.0]

    for t_idx, (filename, df) in enumerate(group['data']):
        x_vals = (df.index + 1) * meas_per_iter + trial_offsets[t_idx]
        mean_err = df['Orientation Error'] * 180 / np.pi
        std_err = df['Orientation Error Std'] * 180 / np.pi if 'Orientation Error Std' in df.columns else None
        label = group['label'] if group['label'] not in legend_added else None
        if label:
            legend_added[group['label']] = True

        if std_err is not None:
            plt.errorbar(x_vals, mean_err, yerr=std_err, label=label, color=color,
                         linewidth=1.5, capsize=3, capthick=1.2, elinewidth=1.0, fmt='-o', markersize=4, alpha=0.7)
        else:
            plt.plot(x_vals, mean_err, label=label, color=color, linewidth=1.5, marker='o', markersize=4, alpha=0.7)

        orient_errors = df['Orientation Error'] * 180 / np.pi
        min_val = orient_errors[orient_errors > 0].min()
        if min_val is not None and min_val > 0:
            min_vals.append(min_val)

plt.yscale('log')
plt.title('Orientation Error vs Total Measurements', fontsize=16)
plt.xlabel('Total Measurements', fontsize=16)
plt.ylabel('Orientation Error [degrees]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, which='both', axis='y')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

if min_vals:
    overall_min = min(min_vals)
    lower_lim = 10**(np.log10(overall_min) - 0.5)
    plt.ylim(bottom=lower_lim)

ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
ax.tick_params(axis='y', which='minor', labelsize=8)
plt.tight_layout()
plt.savefig(f"{base_name}_orientation_error_log.png", dpi=300)

# ============================================================
# Plot orientation error values (linear y-axis)
# ============================================================
plt.figure()
legend_added = {}
for g_idx, group in enumerate(loaded_groups):
    meas_per_iter = group['measurements_per_iter']
    color = group['color']
    n_trials = len(group['data'])
    trial_offsets = np.linspace(-0.1, 0.1, n_trials) if n_trials > 1 else [0.0]

    for t_idx, (filename, df) in enumerate(group['data']):
        x_vals = (df.index + 1) * meas_per_iter + trial_offsets[t_idx]
        mean_err = df['Orientation Error'] * 180 / np.pi
        std_err = df['Orientation Error Std'] * 180 / np.pi if 'Orientation Error Std' in df.columns else None
        label = group['label'] if group['label'] not in legend_added else None
        if label:
            legend_added[group['label']] = True

        if std_err is not None:
            plt.errorbar(x_vals, mean_err, yerr=std_err, label=label, color=color,
                         linewidth=1.5, capsize=3, capthick=1.2, elinewidth=1.0, fmt='-o', markersize=4, alpha=0.7)
        else:
            plt.plot(x_vals, mean_err, label=label, color=color, linewidth=1.5, marker='o', markersize=4, alpha=0.7)

plt.title('Orientation Error vs Total Measurements', fontsize=16)
plt.xlabel('Total Measurements', fontsize=16)
plt.ylabel('Orientation Error [degrees]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"{base_name}_orientation_error_linear.png", dpi=300)

plt.show()


