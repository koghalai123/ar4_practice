import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, FormatStrFormatter
import numpy as np
import os

# List of CSV files to load and plot
csv_filenames = [
    'linkLengthModelEnabled3.csv'
]

# Load dial indicator validation data
def load_dial_indicator_data(filename):
    """Load and process dial indicator validation data"""
    try:
        dial_data = {'backRight': [], 'backLeft': [], 'frontLeft': [], 'frontRight': []}
        current_position = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('//') and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]) / 1000.0, float(parts[1]) / 1000.0, float(parts[2]) / 1000.0
                            if len(parts) > 3 and parts[3].strip():
                                current_position = parts[3].strip()
                            if current_position and current_position in dial_data:
                                dial_data[current_position].append([x, y, z])
                        except ValueError:
                            continue
        position_errors = {}
        all_errors = []
        for pos, data in dial_data.items():
            if data:
                data_array = np.array(data)
                errors = np.sqrt(np.sum(data_array**2, axis=1))
                position_errors[pos] = np.mean(errors)
                all_errors.extend(errors)
        if all_errors:
            overall_avg_position_error = np.mean(all_errors)
            avg_orientation_error = np.std(all_errors) * 0.1
            return overall_avg_position_error, avg_orientation_error, position_errors
        else:
            return None, None, {}
    except FileNotFoundError:
        return None, None, {}

# Load initial and final dial indicator data
initial_pos_error, initial_orient_error, initial_position_errors = load_dial_indicator_data('dialIndicatorInitialData.txt')
final_pos_error, final_orient_error, final_position_errors = load_dial_indicator_data('dialIndicatorFinalData.txt')

# Load all CSV files
dataframes = {}
for filename in csv_filenames:
    if os.path.exists(filename):
        dataframes[filename] = pd.read_csv(filename)
        print(f"Loaded {filename}")
    else:
        print(f"Warning: {filename} not found, skipping...")

if not dataframes:
    print("No data files found!")
    exit()

# Prepare output image filename base
base_name = "combined_calibration"

# Get consistent colors for each file using a light blue to dark blue gradient
n_files = len(dataframes)
if n_files > 1:
    blues = plt.cm.Blues(np.linspace(0.25, 1.0, n_files))
else:
    blues = ['#0077BB']

file_colors = {filename: blues[i] for i, filename in enumerate(dataframes.keys())}

# Compute x offsets so that trials don't overlap
# Spread trials evenly around each integer x tick
x_offset_range = 0.4  # total spread around each integer
x_offsets = np.linspace(-x_offset_range / 2, x_offset_range / 2, n_files)
file_x_offsets = {filename: x_offsets[i] for i, filename in enumerate(dataframes.keys())}

# Plot combined position and orientation error (logarithmic y-axes)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

pos_min_vals = []
orient_min_vals = []
counter = 1
for filename, df in dataframes.items():
    color = file_colors[filename]
    x_offset = file_x_offsets[filename]
    x_vals = df.index + x_offset
    
    # Position error on left axis
    pos_mean_err = df['Position Error'] * 1000
    pos_std_err = df['Position Error Std'] * 1000 if 'Position Error Std' in df.columns else None
    if pos_std_err is not None:
        ax1.errorbar(x_vals, pos_mean_err, yerr=pos_std_err, label=f'Position', color=color,
                     linewidth=2, capsize=3, capthick=1.5, elinewidth=1.2, fmt='-o', markersize=4)
    else:
        ax1.plot(x_vals, pos_mean_err, label=f'Position', color=color, linewidth=2, marker='o', markersize=4)
    
    # Orientation error on right axis (dashed lines)
    orient_mean_err = df['Orientation Error'] * 180/np.pi
    orient_std_err = df['Orientation Error Std'] * 180/np.pi if 'Orientation Error Std' in df.columns else None
    if orient_std_err is not None:
        ax2.errorbar(x_vals, orient_mean_err, yerr=orient_std_err, label=f'Orientation', color=color,
                     linewidth=2, capsize=3, capthick=1.5, elinewidth=1.2, fmt='--s', markersize=4)
    else:
        ax2.plot(x_vals, orient_mean_err, label=f'Orientation', color=color, linewidth=2, linestyle='--', marker='s', markersize=4)
    
    pos_errors = df['Position Error'] * 1000
    pos_min_val = pos_errors[pos_errors > 0].min()
    if pos_min_val is not None and pos_min_val > 0:
        pos_min_vals.append(pos_min_val)
    
    orient_errors = df['Orientation Error'] * 180/np.pi
    orient_min_val = orient_errors[orient_errors > 0].min()
    if orient_min_val is not None and orient_min_val > 0:
        orient_min_vals.append(orient_min_val)
    counter += 1

ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_title('Position and Orientation Error Values', fontsize=16)
ax1.set_xlabel('Iteration', fontsize=16)
ax1.set_ylabel('Position Error [mm]', fontsize=16)
ax2.set_ylabel('Orientation Error [degrees]', fontsize=16)

# Set y-axis limits based on minimum values
if pos_min_vals:
    pos_overall_min = min(pos_min_vals)
    pos_lower_lim = 10**(np.log10(pos_overall_min) - 0.5)
    ax1.set_ylim(bottom=pos_lower_lim)

if orient_min_vals:
    orient_overall_min = min(orient_min_vals)
    orient_lower_lim = 10**(np.log10(orient_overall_min) - 0.5)
    ax2.set_ylim(bottom=orient_lower_lim)

ax1.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
ax1.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
ax1.tick_params(axis='y', which='minor', labelsize=8)
ax1.tick_params(axis='both', which='major', labelsize=16)

ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
ax2.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
ax2.tick_params(axis='y', which='minor', labelsize=8)
ax2.tick_params(axis='y', which='major', labelsize=16)

ax1.grid(True, which='both', axis='y')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig(f"{base_name}_combined_error_log.png", dpi=300)

# Plot combined position error and dL J6 calibration parameter (linear y-axes)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

counter = 1
for filename, df in dataframes.items():
    color = file_colors[filename]
    x_offset = file_x_offsets[filename]
    x_vals = df.index + x_offset
    
    # Position error on left axis
    pos_mean_err = df['Position Error'] * 1000
    pos_std_err = df['Position Error Std'] * 1000 if 'Position Error Std' in df.columns else None
    if pos_std_err is not None:
        ax1.errorbar(x_vals, pos_mean_err, yerr=pos_std_err, label=f'Position Error', color=color,
                     linewidth=2, capsize=3, capthick=1.5, elinewidth=1.2, fmt='-o', markersize=4)
    else:
        ax1.plot(x_vals, pos_mean_err, label=f'Position Error', color=color, linewidth=2, marker='o', markersize=4)
    
    # dL J6 calibration parameter on right axis (dashed lines)
    if 'dL Estimated_6' in df.columns:
        dl_j6 = df['dL Estimated_6'].abs()
        ax2.plot(x_vals, dl_j6, label=f'dL J6', color=color, linewidth=2, linestyle='--', marker='s', markersize=4)
    counter += 1

ax1.set_title('Position Error and dL J6 Calibration Parameter', fontsize=16)
ax1.set_xlabel('Iteration', fontsize=16)
ax1.set_ylabel('Position Error [mm]', fontsize=16)
ax2.set_ylabel('dL J6 [m]', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='y', which='major', labelsize=16)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig(f"{base_name}_position_and_dL_linear.png", dpi=300)

# Plot estimated target pose values (difference from final value)
plt.figure()
pose_cols = ['Estimated Target X', 'Estimated Target Y', 'Estimated Target Z',
             'Estimated Target Roll', 'Estimated Target Pitch', 'Estimated Target Yaw']

for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    for col in pose_cols:
        if col in df.columns:
            diff = df[col] - df[col].iloc[-1]
            # Convert angular values to degrees
            if 'Roll' in col or 'Pitch' in col or 'Yaw' in col:
                diff = diff * 180/np.pi
            plt.plot(diff, label=f'{label_base} - {col}', linewidth=2)

plt.title('Estimated Target Pose (Difference from Final Value) - Combined', fontsize=16)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Value - Final Value [mm/degrees]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"{base_name}_est_target_pose.png", dpi=300)

# Plot calibration parameters (absolute value)
plt.figure()
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    calib_cols = [col for col in df.columns if col == 'dL Estimated_6']
    for col in calib_cols:
        plt.plot(df[col].abs(), label=col, linewidth=2)

plt.title('Calibration Parameter dL J6', fontsize=16)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Absolute Value', fontsize=16)
plt.legend(loc='upper right', fontsize=14)
plt.grid(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"{base_name}_calib_params.png", dpi=300)

plt.figure(figsize=(10, 6))

# Collect the three datasets
box_data = []
box_labels = []

# 1. Initial position errors from calibration files
initial_errors = []
for filename, df in dataframes.items():
    if 'Position Error' in df.columns and len(df) > 0:
        initial_errors.append(df['Position Error'].iloc[0] * 1000)

print("Camera Initial Errors:", np.mean(initial_errors))  # Print initial errors

if initial_errors:
    box_data.append(initial_errors)
    box_labels.append('Camera Initial\nError')

# 2. Final position errors from calibration files  
final_errors = []
for filename, df in dataframes.items():
    if 'Position Error' in df.columns and len(df) > 0:
        final_errors.append(df['Position Error'].iloc[-1] * 1000)

print("Camera Final Errors:", np.mean(final_errors))  # Print final errors

if final_errors:
    box_data.append(final_errors)
    box_labels.append('Camera Final\nError')

# 3. Initial dial indicator data
if initial_pos_error is not None:
    dial_initial_errors = [error * 1000 for error in initial_position_errors.values()]
    print("Dial Indicator Initial Errors:", np.mean(dial_initial_errors))  # Print dial initial errors
    if dial_initial_errors:
        box_data.append(dial_initial_errors)
        box_labels.append('Dial Indicator\nInitial Error')

# 4. Final dial indicator data
if final_pos_error is not None:
    dial_final_errors = [error * 1000 for error in final_position_errors.values()]
    print("Dial Indicator Final Errors:", np.mean(dial_final_errors))  # Print dial final errors
    if dial_final_errors:
        box_data.append(dial_final_errors)
        box_labels.append('Dial Indicator\nFinal Error')

# Create the box plot
if len(box_data) >= 4:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    box = ax.boxplot(
        box_data,
        labels=box_labels,
        widths=0.7,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5),
        patch_artist=True
    )
    # Shade the boxes with colorblind-friendly colors
    # Using a subset of the 'Paired' colormap
    box_colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_yscale('log')
    ax.set_title('Position Error Comparison', fontsize=22)
    ax.set_ylabel('Position Error [mm]', fontsize=18)
    ax.grid(True, which='both', axis='y', alpha=0.3)
    ax.set_xticklabels(box_labels, fontsize=14, rotation=10)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 4, 6, 8], numticks=100))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.0e'))
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=14)
    plt.savefig(f"{base_name}_simple_boxplot.png", dpi=300)

plt.show()


