
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, FormatStrFormatter
import numpy as np
import os

# Global figure size (width, height) in inches
FIGURE_SIZE = (4.5, 3)

# Global font size for titles, labels, and legends
FONT_SIZE = 10

# Font size for axis tick labels
TICK_FONT_SIZE = 10

 # Define trial groups with their measurements per iteration
# Option to show faint individual lines or not
SHOW_INDIVIDUAL_LINES = False  # Set to False to hide individual trend lines

# Blue monochrome: dark, medium, light blue
MONO_COLORS = ['#08306b', '#2171b3', '#6baed6']  # dark, medium, light blue
MARKERS = ['o', 's', 'D']  # circle, square, diamond
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
        'color': MONO_COLORS[0],
        'marker': MARKERS[0],
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
        'color': MONO_COLORS[1],
        'marker': MARKERS[1],
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
        'color': MONO_COLORS[2],
        'marker': MARKERS[2],
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
            'marker': group['marker'],
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
plt.figure(figsize=FIGURE_SIZE)
min_vals = []
legend_handles = []
legend_added = {}


# --- Plot faint individual lines and group mean with error band ---
for g_idx, group in enumerate(loaded_groups):
    meas_per_iter = group['measurements_per_iter']
    color = group['color']
    n_trials = len(group['data'])
    marker = group.get('marker', 'o')
    if n_trials > 1:
        trial_offsets = np.linspace(-0.1, 0.1, n_trials)
    else:
        trial_offsets = [0.0]

    # Collect all trial data for group mean
    all_x = []
    all_y = []
    for t_idx, (filename, df) in enumerate(group['data']):
        x_vals = (df.index + 1) * meas_per_iter + trial_offsets[t_idx]
        mean_err = df['Position Error'] * 1000
        std_err = df['Position Error Std'] * 1000 if 'Position Error Std' in df.columns else None
        # Only add label for the first trial in each group
        label = group['label'] if group['label'] not in legend_added else None
        if label:
            legend_added[group['label']] = True
        fmt = '-' + marker
        # Faint individual lines (optional)
        if SHOW_INDIVIDUAL_LINES:
            if std_err is not None:
                plt.errorbar(x_vals, mean_err, yerr=std_err, fmt=fmt, color=color,
                             linewidth=0.8, capsize=2, capthick=1.0, elinewidth=0.8, markersize=3, alpha=0.3)
            else:
                plt.plot(x_vals, mean_err, color=color, linewidth=0.8, marker=marker, markersize=3, alpha=0.3)
        all_x.append(x_vals)
        all_y.append(mean_err.values)
        pos_errors = mean_err
        min_val = pos_errors[pos_errors > 0].min()
        if min_val is not None and min_val > 0:
            min_vals.append(min_val)
    # Compute group mean and std (interpolate to common x if needed)
    if all_x:
        # Use the x values from the first trial as reference
        ref_x = all_x[0]
        y_stack = np.stack([np.interp(ref_x, x, y) for x, y in zip(all_x, all_y)])
        mean_y = np.mean(y_stack, axis=0)
        std_y = np.std(y_stack, axis=0)
        # Plot mean line
        plt.plot(ref_x, mean_y, color=color, marker=marker, markersize=5, linewidth=2, label=group['label'])
        # Plot error band
        plt.fill_between(ref_x, mean_y-std_y, mean_y+std_y, color=color, alpha=0.15)

plt.yscale('log')
plt.xlabel('Total Measurements', fontsize=FONT_SIZE)
plt.ylabel('Position Error [mm]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.grid(True, which='both', axis='y')
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)

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
plt.tight_layout(pad=0.3)
plt.savefig(f"{base_name}_position_error_log.png", dpi=300)


# ============================================================
# Plot position error values (linear y-axis) with group mean and error band
# ============================================================
plt.figure(figsize=FIGURE_SIZE)
legend_added = {}
for g_idx, group in enumerate(loaded_groups):
    meas_per_iter = group['measurements_per_iter']
    color = group['color']
    n_trials = len(group['data'])
    marker = group.get('marker', 'o')
    trial_offsets = np.linspace(-0.1, 0.1, n_trials) if n_trials > 1 else [0.0]
    all_x = []
    all_y = []
    for t_idx, (filename, df) in enumerate(group['data']):
        x_vals = (df.index + 1) * meas_per_iter + trial_offsets[t_idx]
        mean_err = df['Position Error'] * 1000
        std_err = df['Position Error Std'] * 1000 if 'Position Error Std' in df.columns else None
        label = group['label'] if group['label'] not in legend_added else None
        if label:
            legend_added[group['label']] = True
        fmt = '-' + marker
        if SHOW_INDIVIDUAL_LINES:
            if std_err is not None:
                plt.errorbar(x_vals, mean_err, yerr=std_err, fmt=fmt, color=color,
                             linewidth=0.8, capsize=2, capthick=1.0, elinewidth=0.8, markersize=3, alpha=0.3)
            else:
                plt.plot(x_vals, mean_err, color=color, linewidth=0.8, marker=marker, markersize=3, alpha=0.3)
        all_x.append(x_vals)
        all_y.append(mean_err.values)
    if all_x:
        ref_x = all_x[0]
        y_stack = np.stack([np.interp(ref_x, x, y) for x, y in zip(all_x, all_y)])
        mean_y = np.mean(y_stack, axis=0)
        std_y = np.std(y_stack, axis=0)
        plt.plot(ref_x, mean_y, color=color, marker=marker, markersize=5, linewidth=2, label=group['label'])
        plt.fill_between(ref_x, mean_y-std_y, mean_y+std_y, color=color, alpha=0.15)

plt.xlabel('Total Measurements', fontsize=FONT_SIZE)
plt.ylabel('Position Error [mm]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.grid(False)
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.tight_layout(pad=0.3)
plt.savefig(f"{base_name}_position_error_linear.png", dpi=300)


# ============================================================
# Plot orientation error values (logarithmic y-axis) with group mean and error band
# ============================================================
plt.figure(figsize=FIGURE_SIZE)
min_vals = []
legend_added = {}
for g_idx, group in enumerate(loaded_groups):
    meas_per_iter = group['measurements_per_iter']
    color = group['color']
    n_trials = len(group['data'])
    marker = group.get('marker', 'o')
    trial_offsets = np.linspace(-0.1, 0.1, n_trials) if n_trials > 1 else [0.0]
    all_x = []
    all_y = []
    for t_idx, (filename, df) in enumerate(group['data']):
        x_vals = (df.index + 1) * meas_per_iter + trial_offsets[t_idx]
        mean_err = df['Orientation Error'] * 180 / np.pi
        std_err = df['Orientation Error Std'] * 180 / np.pi if 'Orientation Error Std' in df.columns else None
        label = group['label'] if group['label'] not in legend_added else None
        if label:
            legend_added[group['label']] = True
        fmt = '-' + marker
        if SHOW_INDIVIDUAL_LINES:
            if std_err is not None:
                plt.errorbar(x_vals, mean_err, yerr=std_err, fmt=fmt, color=color,
                             linewidth=0.8, capsize=2, capthick=1.0, elinewidth=0.8, markersize=3, alpha=0.3)
            else:
                plt.plot(x_vals, mean_err, color=color, linewidth=0.8, marker=marker, markersize=3, alpha=0.3)
        all_x.append(x_vals)
        all_y.append(mean_err.values)
        orient_errors = mean_err
        min_val = orient_errors[orient_errors > 0].min()
        if min_val is not None and min_val > 0:
            min_vals.append(min_val)
    if all_x:
        ref_x = all_x[0]
        y_stack = np.stack([np.interp(ref_x, x, y) for x, y in zip(all_x, all_y)])
        mean_y = np.mean(y_stack, axis=0)
        std_y = np.std(y_stack, axis=0)
        plt.plot(ref_x, mean_y, color=color, marker=marker, markersize=5, linewidth=2, label=group['label'])
        plt.fill_between(ref_x, mean_y-std_y, mean_y+std_y, color=color, alpha=0.15)

plt.yscale('log')
plt.xlabel('Total Measurements', fontsize=FONT_SIZE)
plt.ylabel('Orientation Error [degrees]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.grid(True, which='both', axis='y')
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)

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
plt.tight_layout(pad=0.3)
plt.savefig(f"{base_name}_orientation_error_log.png", dpi=300)


# ============================================================
# Plot orientation error values (linear y-axis) with group mean and error band
# ============================================================
plt.figure(figsize=FIGURE_SIZE)
legend_added = {}
for g_idx, group in enumerate(loaded_groups):
    meas_per_iter = group['measurements_per_iter']
    color = group['color']
    n_trials = len(group['data'])
    marker = group.get('marker', 'o')
    trial_offsets = np.linspace(-0.1, 0.1, n_trials) if n_trials > 1 else [0.0]
    all_x = []
    all_y = []
    for t_idx, (filename, df) in enumerate(group['data']):
        x_vals = (df.index + 1) * meas_per_iter + trial_offsets[t_idx]
        mean_err = df['Orientation Error'] * 180 / np.pi
        std_err = df['Orientation Error Std'] * 180 / np.pi if 'Orientation Error Std' in df.columns else None
        label = group['label'] if group['label'] not in legend_added else None
        if label:
            legend_added[group['label']] = True
        fmt = '-' + marker
        if SHOW_INDIVIDUAL_LINES:
            if std_err is not None:
                plt.errorbar(x_vals, mean_err, yerr=std_err, fmt=fmt, color=color,
                             linewidth=0.8, capsize=2, capthick=1.0, elinewidth=0.8, markersize=3, alpha=0.3)
            else:
                plt.plot(x_vals, mean_err, color=color, linewidth=0.8, marker=marker, markersize=3, alpha=0.3)
        all_x.append(x_vals)
        all_y.append(mean_err.values)
    if all_x:
        ref_x = all_x[0]
        y_stack = np.stack([np.interp(ref_x, x, y) for x, y in zip(all_x, all_y)])
        mean_y = np.mean(y_stack, axis=0)
        std_y = np.std(y_stack, axis=0)
        plt.plot(ref_x, mean_y, color=color, marker=marker, markersize=5, linewidth=2, label=group['label'])
        plt.fill_between(ref_x, mean_y-std_y, mean_y+std_y, color=color, alpha=0.15)

plt.xlabel('Total Measurements', fontsize=FONT_SIZE)
plt.ylabel('Orientation Error [degrees]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.grid(False)
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.tight_layout(pad=0.3)
plt.savefig(f"{base_name}_orientation_error_linear.png", dpi=300)

plt.show()


