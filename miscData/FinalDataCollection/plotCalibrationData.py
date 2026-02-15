import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, FormatStrFormatter, MultipleLocator
import numpy as np
import os

# Global figure size (width, height) in inches
FIGURE_SIZE = (4.5, 3)

# Global font size for titles, labels, and legends
FONT_SIZE = 10

# Font size for axis tick labels
TICK_FONT_SIZE = 10


# Layout padding (smaller = less whitespace)
LAYOUT_PAD = 0.3

# Legend label spacing (vertical space between legend entries)
LEGEND_LABELSPACING = 0.2

# List of CSV files to load and plot
csv_filenames = [
    'successfulCalibration9.csv',
    'successfulCalibration10.csv',
    'successfulCalibration11.csv',
    'successfulCalibration12.csv',
    'successfulCalibration13.csv',
    'successfulCalibration14.csv',
    'successfulCalibration15.csv',
    'successfulCalibration16.csv',
    # Add more filenames as needed
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

# Plot position error values (logarithmic y-axis)
plt.figure(figsize=FIGURE_SIZE)
min_vals = []
counter = 1
for filename, df in dataframes.items():
    color = file_colors[filename]
    x_offset = file_x_offsets[filename]
    x_vals = df.index + x_offset
    mean_err = df['Position Error'] * 1000
    std_err = df['Position Error Std'] * 1000 if 'Position Error Std' in df.columns else None
    if std_err is not None:
        plt.errorbar(x_vals, mean_err, yerr=std_err, label=f'Trial {counter}', color=color,
                     linewidth=2, capsize=3, capthick=1.5, elinewidth=1.2, fmt='-o', markersize=4)
    else:
        plt.plot(x_vals, mean_err, label=f'Trial {counter}', color=color, linewidth=2, marker='o', markersize=4)
    
    pos_errors = df['Position Error'] * 1000
    min_val = pos_errors[pos_errors > 0].min()
    if min_val is not None and min_val > 0:
        min_vals.append(min_val)
    counter += 1

plt.yscale('log')
plt.title('Position Error Values', fontsize=FONT_SIZE)
plt.xlabel('Iteration', fontsize=FONT_SIZE)
plt.ylabel('Position Error [mm]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE, labelspacing=LEGEND_LABELSPACING)
plt.grid(True, which='both', axis='y')
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)

# Set y-axis limits based on minimum values from all datasets
if min_vals:
    overall_min = min(min_vals)
    lower_lim = 10**(np.log10(overall_min) - 0.5)
    plt.ylim(bottom=lower_lim)

ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
ax.tick_params(axis='y', which='minor', labelsize=TICK_FONT_SIZE-6)
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.tight_layout(pad=LAYOUT_PAD)
plt.savefig(f"{base_name}_position_error_log.png", dpi=300, bbox_inches='tight')

# Plot position error values (linear y-axis)
plt.figure(figsize=FIGURE_SIZE)
counter = 1
for filename, df in dataframes.items():
    color = file_colors[filename]
    x_offset = file_x_offsets[filename]
    x_vals = df.index + x_offset
    mean_err = df['Position Error'] * 1000
    std_err = df['Position Error Std'] * 1000 if 'Position Error Std' in df.columns else None
    if std_err is not None:
        plt.errorbar(x_vals, mean_err, yerr=std_err, label=f'Trial {counter}', color=color,
                     linewidth=2, capsize=3, capthick=1.5, elinewidth=1.2, fmt='-o', markersize=4)
    else:
        plt.plot(x_vals, mean_err, label=f'Trial {counter}', color=color, linewidth=2, marker='o', markersize=4)
    counter += 1

plt.title('Position Error Values', fontsize=FONT_SIZE)
plt.xlabel('Iteration', fontsize=FONT_SIZE)
plt.ylabel('Position Error [mm]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE, labelspacing=LEGEND_LABELSPACING)
plt.grid(False)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.tight_layout(pad=LAYOUT_PAD)
plt.savefig(f"{base_name}_position_error_linear.png", dpi=300, bbox_inches='tight')

# Plot orientation error values (logarithmic y-axis)
plt.figure(figsize=FIGURE_SIZE)
min_vals = []
counter = 1
for filename, df in dataframes.items():
    color = file_colors[filename]
    x_offset = file_x_offsets[filename]
    x_vals = df.index + x_offset
    mean_err = df['Orientation Error'] * 180/np.pi
    std_err = df['Orientation Error Std'] * 180/np.pi if 'Orientation Error Std' in df.columns else None
    if std_err is not None:
        plt.errorbar(x_vals, mean_err, yerr=std_err, label=f'Trial {counter}', color=color,
                     linewidth=2, capsize=3, capthick=1.5, elinewidth=1.2, fmt='-o', markersize=4)
    else:
        plt.plot(x_vals, mean_err, label=f'Trial {counter}', color=color, linewidth=2, marker='o', markersize=4)
    
    orient_errors = df['Orientation Error'] * 180/np.pi
    min_val = orient_errors[orient_errors > 0].min()
    if min_val is not None and min_val > 0:
        min_vals.append(min_val)
    counter += 1

plt.yscale('log')
plt.title('Orientation Error Values', fontsize=FONT_SIZE)
plt.xlabel('Iteration', fontsize=FONT_SIZE)
plt.ylabel('Orientation Error [degrees]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE, labelspacing=LEGEND_LABELSPACING)
plt.grid(True, which='both', axis='y')
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)

# Set y-axis limits based on minimum values from all datasets
if min_vals:
    overall_min = min(min_vals)
    lower_lim = 10**(np.log10(overall_min) - 0.5)
    plt.ylim(bottom=lower_lim)

ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
ax.tick_params(axis='y', which='minor', labelsize=TICK_FONT_SIZE-6)
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.tight_layout(pad=LAYOUT_PAD)
plt.savefig(f"{base_name}_orientation_error_log.png", dpi=300, bbox_inches='tight')

# Plot orientation error values (linear y-axis)
plt.figure(figsize=FIGURE_SIZE)
counter = 1
for filename, df in dataframes.items():
    color = file_colors[filename]
    x_offset = file_x_offsets[filename]
    x_vals = df.index + x_offset
    mean_err = df['Orientation Error'] * 180/np.pi
    std_err = df['Orientation Error Std'] * 180/np.pi if 'Orientation Error Std' in df.columns else None
    if std_err is not None:
        plt.errorbar(x_vals, mean_err, yerr=std_err, label=f'Trial {counter}', color=color,
                     linewidth=2, capsize=3, capthick=1.5, elinewidth=1.2, fmt='-o', markersize=4)
    else:
        plt.plot(x_vals, mean_err, label=f'Trial {counter}', color=color, linewidth=2, marker='o', markersize=4)
    counter += 1

plt.title('Orientation Error Values', fontsize=FONT_SIZE)
plt.xlabel('Iteration', fontsize=FONT_SIZE)
plt.ylabel('Orientation Error [degrees]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE, labelspacing=LEGEND_LABELSPACING)
plt.grid(False)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.tight_layout(pad=LAYOUT_PAD)
plt.savefig(f"{base_name}_orientation_error_linear.png", dpi=300, bbox_inches='tight')

# Plot estimated target pose values (difference from final value)
plt.figure(figsize=FIGURE_SIZE)
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

plt.title('Estimated Target Pose (Difference from Final Value) - Combined', fontsize=FONT_SIZE)
plt.xlabel('Iteration', fontsize=FONT_SIZE)
plt.ylabel('Value - Final Value [mm/degrees]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE, labelspacing=LEGEND_LABELSPACING)
plt.grid(False)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.tight_layout(pad=LAYOUT_PAD)
plt.savefig(f"{base_name}_est_target_pose.png", dpi=300, bbox_inches='tight')

# Plot calibration parameters (difference from final value)
plt.figure(figsize=FIGURE_SIZE)
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    calib_cols = [col for col in df.columns if 'Estimated' in col and col not in pose_cols]
    for col in calib_cols:
        diff = df[col] - df[col].iloc[-1]
        plt.plot(diff, label=f'{label_base} - {col}', linewidth=2)

plt.title('Calibration Parameters (Difference from Final Value) - Combined', fontsize=FONT_SIZE)
plt.xlabel('Iteration', fontsize=FONT_SIZE)
plt.ylabel('Value - Final Value', fontsize=FONT_SIZE)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZE, labelspacing=LEGEND_LABELSPACING)
plt.grid(False)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.tight_layout(pad=LAYOUT_PAD)
plt.savefig(f"{base_name}_calib_params.png", dpi=300, bbox_inches='tight')

# Collect the three datasets
box_data = []
box_labels = []


# 1. Initial position and orientation errors from calibration files
initial_errors = []
initial_orient_errors = []
for filename, df in dataframes.items():
    if 'Position Error' in df.columns and len(df) > 0:
        initial_errors.append(df['Position Error'].iloc[0] * 1000)
    if 'Orientation Error' in df.columns and len(df) > 0:
        initial_orient_errors.append(df['Orientation Error'].iloc[0] * 180/np.pi)

print("Camera Initial Errors:", np.mean(initial_errors))  # Print initial position errors
if initial_orient_errors:
    print("Camera Initial Orientation Errors:", np.mean(initial_orient_errors))  # Print initial orientation errors

if initial_errors:
    box_data.append(initial_errors)
    box_labels.append('Camera Initial\nError')

# 2. Final position and orientation errors from calibration files  
final_errors = []
final_orient_errors = []
for filename, df in dataframes.items():
    if 'Position Error' in df.columns and len(df) > 0:
        final_errors.append(df['Position Error'].iloc[-1] * 1000)
    if 'Orientation Error' in df.columns and len(df) > 0:
        final_orient_errors.append(df['Orientation Error'].iloc[-1] * 180/np.pi)

print("Camera Final Errors:", np.mean(final_errors))  # Print final position errors
if final_orient_errors:
    print("Camera Final Orientation Errors:", np.mean(final_orient_errors))  # Print final orientation errors

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
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
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
    ax.set_title('Position Error Comparison', fontsize=FONT_SIZE)
    ax.set_ylabel('Position Error [mm]', fontsize=FONT_SIZE)
    ax.grid(True, which='both', axis='y', alpha=0.3)
    ax.set_xticklabels(box_labels, fontsize=TICK_FONT_SIZE, rotation=10)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 4, 6, 8], numticks=100))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.0e'))
    ax.tick_params(axis='y', which='major', labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=TICK_FONT_SIZE)
    plt.savefig(f"{base_name}_simple_boxplot.png", dpi=300, bbox_inches='tight')

plt.show()


