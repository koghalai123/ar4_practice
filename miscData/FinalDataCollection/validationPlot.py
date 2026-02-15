import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Centralized Plot Config ===
FONT_SIZE = 10
TICK_FONT_SIZE = 10
FIG_SIZE = (4.5, 3)
BOXPLOT_WIDTH = 0.7
BOXPLOT_BORDER_WIDTH = 1.5

# File paths
DIAL_INITIAL = 'dialIndicatorInitialData.txt'
DIAL_FINAL = 'dialIndicatorFinalData.txt'
CAMERA_FILES = [
    'successfulCalibration9.csv',
    'successfulCalibration10.csv',
    'successfulCalibration11.csv',
    'successfulCalibration12.csv',
    'successfulCalibration13.csv',
    'successfulCalibration14.csv',
    'successfulCalibration15.csv',
    'successfulCalibration16.csv',
]

def load_dial_indicator_data(filename):
    """Load and process dial indicator validation data"""
    
    dial_data = {'backRight': [], 'backLeft': [], 'frontLeft': [], 'frontRight': []}
    current_position = None
    loaded = False
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('//') and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            if len(parts) > 3 and parts[3].strip():
                                current_position = parts[3].strip()
                            if current_position and current_position in dial_data:
                                dial_data[current_position].append([x, y, z])
                                loaded = True
                        except ValueError:
                            continue
        if loaded:
            print(f"Loaded dial indicator file: {filename}")
        else:
            print(f"No valid data found in dial indicator file: {filename}")
    except Exception as e:
        print(f"Failed to load dial indicator file: {filename} ({e})")
    # Compute mean error for each position
    position_errors = {}
    all_errors = []
    for pos, data in dial_data.items():
        if data:
            data_array = np.array(data)
            errors = np.sqrt(np.sum(data_array**2, axis=1))
            position_errors[pos] = np.mean(errors)
            all_errors.extend(errors)
    return position_errors, all_errors

def load_camera_errors(filenames, which='initial'):
    """Load initial or final camera calibration errors from CSVs (in mm)"""
    errors = []
    for fname in filenames:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname)
                if 'Position Error' in df.columns and len(df) > 0:
                    if which == 'initial':
                        errors.append(df['Position Error'].iloc[0] * 1000)
                    else:
                        errors.append(df['Position Error'].iloc[-1] * 1000)
                    print(f"Loaded camera calibration file: {fname}")
                else:
                    print(f"File {fname} missing 'Position Error' column or is empty.")
            except Exception as e:
                print(f"Failed to load camera calibration file: {fname} ({e})")
        else:
            print(f"Camera calibration file not found: {fname}")
    return errors

def main():
    # Load dial indicator data
    dial_init_pos, dial_init_all = load_dial_indicator_data(DIAL_INITIAL)
    dial_final_pos, dial_final_all = load_dial_indicator_data(DIAL_FINAL)
    dial_init_errors = [e for e in dial_init_pos.values()]
    dial_final_errors = [e for e in dial_final_pos.values()]

    # Load camera calibration errors
    cam_init_errors = load_camera_errors(CAMERA_FILES, which='initial')
    cam_final_errors = load_camera_errors(CAMERA_FILES, which='final')

    # Print average initial and final values for both methods
    def avg_or_nan(lst):
        return np.mean(lst) if lst else float('nan')

    print("\n--- Average Initial and Final Values ---")
    print(f"Camera Initial Mean: {avg_or_nan(cam_init_errors):.2f} mm")
    print(f"Camera Final Mean: {avg_or_nan(cam_final_errors):.2f} mm")
    print(f"Dial Indicator Initial Mean: {avg_or_nan(dial_init_errors):.2f} mm")
    print(f"Dial Indicator Final Mean: {avg_or_nan(dial_final_errors):.2f} mm")

    # Prepare boxplot data
    box_data = []
    box_labels = []
    if cam_init_errors:
        box_data.append(cam_init_errors)
        box_labels.append('Camera Initial')
    if cam_final_errors:
        box_data.append(cam_final_errors)
        box_labels.append('Camera Final')
    if dial_init_errors:
        box_data.append(dial_init_errors)
        box_labels.append('Dial Indicator Initial')
    if dial_final_errors:
        box_data.append(dial_final_errors)
        box_labels.append('Dial Indicator Final')

    # Plot
    if len(box_data) >= 2:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        box = ax.boxplot(
            box_data,
            labels=box_labels,
            widths=BOXPLOT_WIDTH,
            boxprops=dict(linewidth=BOXPLOT_BORDER_WIDTH),
            whiskerprops=dict(linewidth=BOXPLOT_BORDER_WIDTH),
            capprops=dict(linewidth=BOXPLOT_BORDER_WIDTH),
            medianprops=dict(linewidth=BOXPLOT_BORDER_WIDTH),
            patch_artist=True,
            whis=[0, 100],   # Whiskers extend to min/max
            showfliers=False # No outlier markers
        )
        # Monochrome blue: camera = lighter blue, dial = darker blue
        camera_blue = '#4FC3F7'
        dial_blue = '#1976D2'
        n_camera = sum('Camera' in label for label in box_labels)
        n_dial = sum('Dial' in label for label in box_labels)
        box_colors = [camera_blue]*n_camera + [dial_blue]*n_dial
        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        # Linear y-axis with more frequent ticks
        ax.set_title('Validation Comparison', fontsize=FONT_SIZE)
        ax.set_ylabel('Position Error [mm]', fontsize=FONT_SIZE)
        # No grid lines
        ax.set_xticklabels(box_labels, fontsize=TICK_FONT_SIZE, rotation=10)
        ax.tick_params(axis='y', which='major', labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis='y', which='minor', length=4, labelsize=TICK_FONT_SIZE)

        # Add more frequent y-axis ticks (linear scale)
        from matplotlib.ticker import MultipleLocator, AutoMinorLocator
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        # Set major ticks to 5 or 10 depending on range
        if y_range > 50:
            ax.yaxis.set_major_locator(MultipleLocator(10))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

        plt.tight_layout()
        plt.savefig('validation_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print('Not enough data to plot boxplot.')

if __name__ == '__main__':
    main()
