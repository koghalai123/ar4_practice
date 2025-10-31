import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, FormatStrFormatter
import numpy as np
import os

# List of CSV files to load and plot
csv_filenames = [
    'successfulCalibration1.csv',
    'successfulCalibration2.csv',
    'successfulCalibration3.csv',
    'successfulCalibration4.csv',
    'successfulCalibration5.csv',
    'successfulCalibration6.csv',
    'successfulCalibration7.csv',
    'successfulCalibration8.csv',

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

# Get consistent colors for each file
colors = plt.cm.tab10(np.linspace(0, 1, len(dataframes)))
file_colors = {filename: colors[i] for i, filename in enumerate(dataframes.keys())}

# --- For all plots, increase legend font size ---
# Example for one plot, apply to all:
# plt.legend(fontsize=14)  # Change from default to larger font

# Plot position error values (logarithmic y-axis)
plt.figure()
min_vals = []
counter = 1
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    color = file_colors[filename]
    plt.plot(df['Position Error'], label=f'Trial {counter}', color=color, linewidth=2)
    
    # Collect min values for y-axis scaling
    pos_errors = df['Position Error']
    min_val = pos_errors[pos_errors > 0].min()
    if min_val is not None and min_val > 0:
        min_vals.append(min_val)
    counter += 1

'''# Add validation lines if available
if validation_pos_error is not None:
    plt.axhline(y=validation_pos_error, color='red', linestyle='--', linewidth=2, 
                label=f'Dial Indicator Overall: {validation_pos_error:.6f}m')

# Add position-specific validation lines
position_colors = {'backRight': 'orange', 'backLeft': 'purple', 'frontLeft': 'green', 'frontRight': 'brown'}
for pos, error in position_specific_errors.items():
    plt.axhline(y=error, color=position_colors.get(pos, 'gray'), linestyle=':', linewidth=2,
                label=f'Dial {pos}: {error:.6f}m')'''

plt.yscale('log')
plt.title('Position Error Values', fontsize=16)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Position Error [meters]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, which='both', axis='y')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

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
ax.tick_params(axis='y', which='minor', labelsize=8)
plt.tight_layout()
plt.savefig(f"{base_name}_position_error_log.png", dpi=300)

# Plot position error values (linear y-axis)
plt.figure()
counter = 1
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    color = file_colors[filename]
    plt.plot(df['Position Error'], label=f'Trial {counter}', color=color, linewidth=2)
    counter += 1

plt.title('Position Error Values', fontsize=16)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Position Error [meters]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"{base_name}_position_error_linear.png", dpi=300)

# Plot orientation error values (logarithmic y-axis)
plt.figure()
min_vals = []
counter = 1
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    color = file_colors[filename]
    plt.plot(df['Orientation Error'], label=f'Trial {counter}', color=color, linewidth=2)
    
    # Collect min values for y-axis scaling
    orient_errors = df['Orientation Error']
    min_val = orient_errors[orient_errors > 0].min()
    if min_val is not None and min_val > 0:
        min_vals.append(min_val)
    counter += 1

'''# Add validation line if available
if validation_orient_error is not None:
    plt.axhline(y=validation_orient_error, color='red', linestyle='--', linewidth=2, 
                label=f'Estimated Validation: {validation_orient_error:.6f}rad')'''

plt.yscale('log')
plt.title('Orientation Error Values', fontsize=16)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Orientation Error [radians]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, which='both', axis='y')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

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
ax.tick_params(axis='y', which='minor', labelsize=8)
plt.tight_layout()
plt.savefig(f"{base_name}_orientation_error_log.png", dpi=300)

# Plot orientation error values (linear y-axis)
plt.figure()
counter = 1
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    color = file_colors[filename]
    plt.plot(df['Orientation Error'], label=f'Trial {counter}', color=color, linewidth=2)
    counter += 1

plt.title('Orientation Error Values', fontsize=16)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Orientation Error [radians]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"{base_name}_orientation_error_linear.png", dpi=300)

# Plot estimated target pose values (difference from final value)
plt.figure()
pose_cols = ['Estimated Target X', 'Estimated Target Y', 'Estimated Target Z',
             'Estimated Target Roll', 'Estimated Target Pitch', 'Estimated Target Yaw']

for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    for col in pose_cols:
        if col in df.columns:
            diff = df[col] - df[col].iloc[-1]
            plt.plot(diff, label=f'{label_base} - {col}', linewidth=2)

plt.title('Estimated Target Pose (Difference from Final Value) - Combined', fontsize=16)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Value - Final Value [meters/radians]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"{base_name}_est_target_pose.png", dpi=300)

# Plot calibration parameters (difference from final value)
plt.figure()
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    calib_cols = [col for col in df.columns if 'Estimated' in col and col not in pose_cols]
    for col in calib_cols:
        diff = df[col] - df[col].iloc[-1]
        plt.plot(diff, label=f'{label_base} - {col}', linewidth=2)

plt.title('Calibration Parameters (Difference from Final Value) - Combined', fontsize=16)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Value - Final Value', fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
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
        initial_errors.append(df['Position Error'].iloc[0])

print("Camera Initial Errors:", np.mean(initial_errors))  # Print initial errors

if initial_errors:
    box_data.append(initial_errors)
    box_labels.append('Camera Initial\nError')

# 2. Final position errors from calibration files  
final_errors = []
for filename, df in dataframes.items():
    if 'Position Error' in df.columns and len(df) > 0:
        final_errors.append(df['Position Error'].iloc[-1])

print("Camera Final Errors:", np.mean(final_errors))  # Print final errors

if final_errors:
    box_data.append(final_errors)
    box_labels.append('Camera Final\nError')

# 3. Initial dial indicator data
if initial_pos_error is not None:
    dial_initial_errors = list(initial_position_errors.values())
    print("Dial Indicator Initial Errors:", np.mean(dial_initial_errors))  # Print dial initial errors
    if dial_initial_errors:
        box_data.append(dial_initial_errors)
        box_labels.append('Dial Indicator\nInitial Error')

# 4. Final dial indicator data
if final_pos_error is not None:
    dial_final_errors = list(final_position_errors.values())
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
    # Shade the boxes
    colors = ["#6EB5F8"]  # Example pastel colors
    for patch in box['boxes']:
        patch.set_facecolor(colors[0])
        patch.set_alpha(0.8)

    ax.set_yscale('log')
    ax.set_title('Position Error Comparison', fontsize=22)
    ax.set_ylabel('Position Error [meters]', fontsize=18)
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


