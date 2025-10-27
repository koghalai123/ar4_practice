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
def load_dial_indicator_data(filename='dialIndicatorData.txt'):
    """Load and process dial indicator validation data"""
    try:
        # Read the dial indicator data
        dial_data = {'backRight': [], 'backLeft': [], 'frontLeft': [], 'frontRight': []}
        current_position = None
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('//') and ',' in line:
                    # Split by comma and take values
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]) / 1000.0, float(parts[1]) / 1000.0, float(parts[2]) / 1000.0  # Convert mm to m
                            
                            # Check if there's a position label
                            if len(parts) > 3 and parts[3].strip():
                                current_position = parts[3].strip()
                            
                            # Add to appropriate position group
                            if current_position and current_position in dial_data:
                                dial_data[current_position].append([x, y, z])
                        except ValueError:
                            continue
        
        # Calculate errors for each position
        position_errors = {}
        all_errors = []
        
        for pos, data in dial_data.items():
            if data:
                data_array = np.array(data)
                errors = np.sqrt(np.sum(data_array**2, axis=1))
                position_errors[pos] = np.mean(errors)
                all_errors.extend(errors)
                print(f"Dial indicator validation - {pos} Avg Position Error: {position_errors[pos]:.6f} m")
        
        if all_errors:
            overall_avg_position_error = np.mean(all_errors)
            # For orientation error, we'll use a simple approximation
            avg_orientation_error = np.std(all_errors) * 0.1  # Rough approximation
            
            print(f"Dial indicator validation - Overall Avg Position Error: {overall_avg_position_error:.6f} m")
            print(f"Dial indicator validation - Est Orientation Error: {avg_orientation_error:.6f} rad")
            
            return overall_avg_position_error, avg_orientation_error, position_errors
        else:
            print("Warning: No valid dial indicator data found")
            return None, None, {}
            
    except FileNotFoundError:
        print(f"Warning: {filename} not found for validation data")
        return None, None, {}

# Load validation data
validation_pos_error, validation_orient_error, position_specific_errors = load_dial_indicator_data()

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

# Plot position error values (logarithmic y-axis)
plt.figure()
min_vals = []

counter = 1
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    color = file_colors[filename]
    plt.plot(df['Position Error'], label=f'Trial {counter}', color=color)
    
    # Collect min values for y-axis scaling
    pos_errors = df['Position Error']
    min_val = pos_errors[pos_errors > 0].min()
    if min_val is not None and min_val > 0:
        min_vals.append(min_val)
    counter += 1

# Add validation lines if available
if validation_pos_error is not None:
    plt.axhline(y=validation_pos_error, color='red', linestyle='--', linewidth=2, 
                label=f'Dial Indicator Overall: {validation_pos_error:.6f}m')

# Add position-specific validation lines
position_colors = {'backRight': 'orange', 'backLeft': 'purple', 'frontLeft': 'green', 'frontRight': 'brown'}
for pos, error in position_specific_errors.items():
    plt.axhline(y=error, color=position_colors.get(pos, 'gray'), linestyle=':', linewidth=1.5,
                label=f'Dial {pos}: {error:.6f}m')

plt.yscale('log')
plt.title('Position Error Values')
plt.xlabel('Iteration')
plt.ylabel('Position Error [meters]')
plt.legend()
plt.grid(True, which='both', axis='y')

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
    plt.plot(df['Position Error'], label=f'Trial {counter}', color=color)
    counter += 1

# Add validation lines if available
if validation_pos_error is not None:
    plt.axhline(y=validation_pos_error, color='red', linestyle='--', linewidth=2, 
                label=f'Dial Indicator Overall: {validation_pos_error:.6f}m')

# Add position-specific validation lines
for pos, error in position_specific_errors.items():
    plt.axhline(y=error, color=position_colors.get(pos, 'gray'), linestyle=':', linewidth=1.5,
                label=f'Dial {pos}: {error:.6f}m')

plt.title('Position Error Values')
plt.xlabel('Iteration')
plt.ylabel('Position Error [meters]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_name}_position_error_linear.png", dpi=300)

# Plot orientation error values (logarithmic y-axis)
plt.figure()
min_vals = []

counter = 1
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    color = file_colors[filename]
    plt.plot(df['Orientation Error'], label=f'Trial {counter}', color=color)
    
    # Collect min values for y-axis scaling
    orient_errors = df['Orientation Error']
    min_val = orient_errors[orient_errors > 0].min()
    if min_val is not None and min_val > 0:
        min_vals.append(min_val)
    counter += 1

# Add validation line if available
if validation_orient_error is not None:
    plt.axhline(y=validation_orient_error, color='red', linestyle='--', linewidth=2, 
                label=f'Estimated Validation: {validation_orient_error:.6f}rad')

plt.yscale('log')
plt.title('Orientation Error Values')
plt.xlabel('Iteration')
plt.ylabel('Orientation Error [radians]')
plt.legend()
plt.grid(True, which='both', axis='y')

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
    plt.plot(df['Orientation Error'], label=f'Trial {counter}', color=color)
    counter += 1

# Add validation line if available
if validation_orient_error is not None:
    plt.axhline(y=validation_orient_error, color='red', linestyle='--', linewidth=2, 
                label=f'Estimated Validation: {validation_orient_error:.6f}rad')

plt.title('Orientation Error Values')
plt.xlabel('Iteration')
plt.ylabel('Orientation Error [radians]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_name}_orientation_error_linear.png", dpi=300)

# ...existing code...

# Plot estimated target pose values (difference from final value)
plt.figure()
pose_cols = ['Estimated Target X', 'Estimated Target Y', 'Estimated Target Z',
             'Estimated Target Roll', 'Estimated Target Pitch', 'Estimated Target Yaw']

for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    for col in pose_cols:
        if col in df.columns:
            diff = df[col] - df[col].iloc[-1]
            plt.plot(diff, label=f'{label_base} - {col}')

plt.title('Estimated Target Pose (Difference from Final Value) - Combined')
plt.xlabel('Iteration')
plt.ylabel('Value - Final Value [meters/radians]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_name}_est_target_pose.png", dpi=300)

# Plot calibration parameters (difference from final value)
plt.figure()
for filename, df in dataframes.items():
    label_base = os.path.splitext(filename)[0]
    calib_cols = [col for col in df.columns if 'Estimated' in col and col not in pose_cols]
    for col in calib_cols:
        diff = df[col] - df[col].iloc[-1]
        plt.plot(diff, label=f'{label_base} - {col}')

plt.title('Calibration Parameters (Difference from Final Value) - Combined')
plt.xlabel('Iteration')
plt.ylabel('Value - Final Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_name}_calib_params.png", dpi=300)

plt.show()