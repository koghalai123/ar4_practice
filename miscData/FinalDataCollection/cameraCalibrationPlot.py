import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: figures are saved, never displayed
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Save outputs next to this script regardless of the working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Single control for every font size in the figures (pt)
FONT_SIZE = 9

def load_and_process_data(filename):
    """Load CSV file and process the camera calibration data"""
    # Read the CSV file, skipping the first 2 header rows
    df = pd.read_csv(filename, skiprows=2)
    
    # Remove any unnamed columns and clean up
    df = df.dropna(subset=[df.columns[0]])  # Remove rows where first column is NaN
    
    # Measurements were recorded in cm; convert to mm (x10) so the mm-labeled axes
    # are correct. Both tape and camera are scaled so the y=x comparison stays valid.
    CM_TO_MM = 10

    # Get tape measure distances (column 1)
    tape_distances = df.iloc[:, 0].values * CM_TO_MM

    # Get camera measurements for each condition
    # 0 deg: columns 1-4, 30 deg: columns 5-8, 60 deg: columns 9-12
    camera_0deg = CM_TO_MM * 15/14.6*df.iloc[:, 1:5].values
    camera_30deg = CM_TO_MM * 15/14.6*df.iloc[:, 5:9].values
    camera_60deg = CM_TO_MM * 15/14.6*df.iloc[:, 9:13].values

    return tape_distances, camera_0deg, camera_30deg, camera_60deg

def create_scatter_plot(tape_distances, camera_data):
    """Create scatter plot of tape measure distance vs camera distance with trendline"""
    conditions = ['0°', '30°', '60°']
    colors = ['#9ecae1', '#4292c6', '#08306b'] # Shades of blue: light, medium, dark
    markers = ['o', 's', '^']
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    # Collect all data points for overall trendline
    all_tape_distances = []
    all_camera_distances = []
    
    # X-axis offsets for each condition to separate overlapping points
    x_offsets = [-9, 0, 9]  # Left, center, right offsets (mm scale)
    
    for i, (condition_data, condition_name, color, marker, x_offset) in enumerate(zip(camera_data, conditions, colors, markers, x_offsets)):
        condition_tape_distances = []
        condition_camera_distances = []
        condition_x_positions = []
        
        for j, tape_dist in enumerate(tape_distances):
            camera_measurements = condition_data[j, :]
            camera_measurements = camera_measurements[~np.isnan(camera_measurements)]
            
            # Add offset to x-positions for this condition
            x_positions = [tape_dist + x_offset] * len(camera_measurements)
            
            condition_tape_distances.extend([tape_dist] * len(camera_measurements))  # Original positions for trendline
            condition_camera_distances.extend(camera_measurements)
            condition_x_positions.extend(x_positions)  # Offset positions for plotting
        
        # Add to overall data for trendline (using original x-positions)
        all_tape_distances.extend(condition_tape_distances)
        all_camera_distances.extend(condition_camera_distances)
        
        # Scatter plot for this condition with offset x-positions
        ax.scatter(condition_x_positions, condition_camera_distances, 
                  color=color, marker=marker, alpha=0.7, s=100, 
                  label=f'{condition_name}', edgecolors='black', linewidth=0.5)
    
    # Add perfect correlation line (y = x)
    min_dist, max_dist = min(tape_distances), max(tape_distances)
    ax.plot([min_dist, max_dist], [min_dist, max_dist], 'k--', linewidth=2, 
            label='Perfect Correlation', alpha=0.7)
    
    # Fit and plot overall trendline using original (non-offset) positions
    slope, intercept, _r_value, _p_value, _std_err = stats.linregress(all_tape_distances, all_camera_distances)
    trendline_y = [slope * x + intercept for x in [min_dist, max_dist]]
    
    ax.plot([min_dist, max_dist], trendline_y, 'purple', linewidth=3, 
            label=f'Overall Trendline\n(y={slope:.1f}x+{intercept:.1f})')
    
    # Formatting
    ax.set_xlabel('Tape Measure Distance [mm]', fontsize=FONT_SIZE)
    ax.set_ylabel('Camera Distance [mm]', fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    ax.set_xlim(min_dist - 20, max_dist + 20)
    ax.set_ylim(min_dist - 20, max_dist + 20)
    
    # Set x-axis ticks to the original tape distances (without offset, every other one)
    xtick_positions = tape_distances[::2]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f'{int(d)}' for d in xtick_positions])
    
    # Increase font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    ax.legend(loc='lower right', fontsize=FONT_SIZE, labelspacing=0.2)
    
    # Add statistics text box (using original positions for residuals)
    residuals = np.array(all_camera_distances) - np.array(all_tape_distances)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    stats_text = f'RMSE: {rmse:.1f} mm\n'
    stats_text += f'MAE: {mae:.1f} mm\n'
    stats_text += f'Mean Residual: {np.mean(residuals):.1f} mm\n'
    stats_text += f'Std Residual: {np.std(residuals):.1f} mm'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=FONT_SIZE,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(pad=0.2)
    return fig, ax

def create_residual_boxplot(tape_distances, camera_data):
    """Create boxplot of residuals as a function of distance"""
    conditions = ['0°', '30°', '60°']
    colors = ['#deebf7', '#4292c6', '#08306b'] # Shades of blue: light, medium, dark (high contrast)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    # Calculate residuals for each condition and distance
    for i, (condition_data, condition_name, color) in enumerate(zip(camera_data, conditions, colors)):
        # Create boxplots for residuals at each distance
        positions = tape_distances + (i - 1) * 12  # Increased offset for better separation (mm scale)
        boxplot_data = []
        
        for j, tape_dist in enumerate(tape_distances):
            camera_measurements = condition_data[j, :]
            camera_measurements = camera_measurements[~np.isnan(camera_measurements)]
            
            # Calculate residuals (camera - tape)
            residuals = camera_measurements - tape_dist
            boxplot_data.append(residuals)
        
        # Create boxplot with increased width
        bp = ax.boxplot(boxplot_data, positions=positions, widths=13,
                       patch_artist=True, manage_ticks=False,
                       whis=[0, 100],  # Extend whiskers to min/max (0th and 100th percentiles)
                       boxprops=dict(facecolor=color, alpha=0.7, linewidth=0.6),
                       whiskerprops=dict(linewidth=0.6),
                       capprops=dict(linewidth=0.6),
                       medianprops=dict(linewidth=1.0, color='red'))
    
    # Add zero line for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero residual')
    
    # Formatting
    ax.set_xlabel('Tape Measure Distance [mm]', fontsize=FONT_SIZE)
    ax.set_ylabel('Residual (Camera - Tape) [mm]', fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks to tape distances (every other one to reduce clutter)
    xtick_positions = tape_distances[::2]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f'{int(d)}' for d in xtick_positions])
    
    # Increase font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    # Create custom legend
    legend_elements = []
    for condition, color in zip(conditions, colors):
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=condition))
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', label='Zero residual'))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0, 0.38), fontsize=FONT_SIZE, labelspacing=0.2)
    
    # Calculate and display overall statistics
    all_residuals = []
    condition_stats = []
    
    for i, (condition_data, condition_name) in enumerate(zip(camera_data, conditions)):
        condition_residuals = []
        for j, tape_dist in enumerate(tape_distances):
            camera_measurements = condition_data[j, :]
            camera_measurements = camera_measurements[~np.isnan(camera_measurements)]
            residuals = camera_measurements - tape_dist
            condition_residuals.extend(residuals)
        
        all_residuals.extend(condition_residuals)
        mean_res = np.mean(condition_residuals)
        std_res = np.std(condition_residuals)
        condition_stats.append(f'{condition_name}: μ={mean_res:.1f}, σ={std_res:.1f} mm')
    
    # Add statistics text box
    overall_mean = np.mean(all_residuals)
    overall_std = np.std(all_residuals)
    
    stats_text = 'Residual Statistics:\n'
    for stat in condition_stats:
        stats_text += stat + '\n'
    stats_text += f'Overall: μ={overall_mean:.1f}, σ={overall_std:.1f} mm'
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=FONT_SIZE,
            verticalalignment='bottom', linespacing=0.8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(pad=0.2)
    return fig, ax

def main():
    """Main function to run the analysis"""
    filename = os.path.join(SCRIPT_DIR, 'CameraCalibrationData.csv')
    
    # Load and process data
    tape_distances, camera_0deg, camera_30deg, camera_60deg = load_and_process_data(filename)
    
    # Plot the raw measured values (no normalization)
    camera_data = [camera_0deg, camera_30deg, camera_60deg]

    # Create scatter plot
    fig1, ax1 = create_scatter_plot(tape_distances, camera_data)
    fig1.savefig(os.path.join(SCRIPT_DIR, 'camera_calibration_scatter_plot.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)

    # Create residual boxplot
    fig2, ax2 = create_residual_boxplot(tape_distances, camera_data)
    fig2.savefig(os.path.join(SCRIPT_DIR, 'camera_calibration_residual_boxplot.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    
    # Print summary statistics
    print("Camera Calibration Analysis Results:")
    print("=" * 50)
    
    conditions = ['0°', '30°', '60°']
    for i, (condition_data, condition_name) in enumerate(zip(camera_data, conditions)):
        all_residuals = []
        for j, tape_dist in enumerate(tape_distances):
            camera_measurements = condition_data[j, :]
            camera_measurements = camera_measurements[~np.isnan(camera_measurements)]
            residuals = camera_measurements - tape_dist
            all_residuals.extend(residuals)
        
        mean_res = np.mean(all_residuals)
        std_res = np.std(all_residuals)
        print(f"{condition_name} condition: Mean residual = {mean_res:.3f} mm, Std = {std_res:.3f} mm")
    
    # Overall statistics
    all_residuals_combined = []
    for condition_data in camera_data:
        for j, tape_dist in enumerate(tape_distances):
            camera_measurements = condition_data[j, :]
            camera_measurements = camera_measurements[~np.isnan(camera_measurements)]
            residuals = camera_measurements - tape_dist
            all_residuals_combined.extend(residuals)
    
    overall_mean = np.mean(all_residuals_combined)
    overall_std = np.std(all_residuals_combined)
    overall_rmse = np.sqrt(np.mean(np.array(all_residuals_combined)**2))
    
    print(f"\nOverall Statistics:")
    print(f"Mean residual: {overall_mean:.3f} mm")
    print(f"Std residual: {overall_std:.3f} mm")
    print(f"RMSE: {overall_rmse:.3f} mm")

if __name__ == "__main__":
    main()