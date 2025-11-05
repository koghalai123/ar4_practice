import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

def load_and_process_data(filename):
    """Load CSV file and process the camera calibration data"""
    # Read the CSV file, skipping the first 2 header rows
    df = pd.read_csv(filename, skiprows=2)
    
    # Remove any unnamed columns and clean up
    df = df.dropna(subset=[df.columns[0]])  # Remove rows where first column is NaN
    
    # Get tape measure distances (column 1)
    tape_distances = df.iloc[:, 0].values
    
    # Get camera measurements for each condition
    # 0 deg: columns 1-4, 30 deg: columns 5-8, 60 deg: columns 9-12
    camera_0deg = 15/14.6*df.iloc[:, 1:5].values
    camera_30deg = 15/14.6*df.iloc[:, 5:9].values
    camera_60deg = 15/14.6*df.iloc[:, 9:13].values
    
    return tape_distances, camera_0deg, camera_30deg, camera_60deg

def normalize_to_20mm(data, tape_distances):
    """Normalize data so that the measurement at 20mm becomes exactly 20mm"""
    normalized_data = []
    
    for condition_data in data:
        # Find the row corresponding to 20mm measurement
        idx_20mm = np.where(tape_distances == 20)[0][0]
        
        # Calculate the mean measurement at 20mm for this condition
        mean_at_20mm = np.nanmean(condition_data[idx_20mm, :])
        
        # Calculate offset needed to make it exactly 20mm
        offset = 20 - mean_at_20mm
        
        # Apply offset to all measurements in this condition
        normalized_condition = condition_data + offset
        normalized_data.append(normalized_condition)
    
    return normalized_data

def create_scatter_plot(tape_distances, normalized_data):
    """Create scatter plot of tape measure distance vs camera distance with trendline"""
    conditions = ['0°', '30°', '60°']
    colors = ['#E69F00', '#56B4E9', '#009E73'] # Colorblind-friendly: Orange, Sky Blue, Bluish Green
    markers = ['o', 's', '^']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect all data points for overall trendline
    all_tape_distances = []
    all_camera_distances = []
    
    # X-axis offsets for each condition to separate overlapping points
    x_offsets = [-0.9, 0, 0.9]  # Left, center, right offsets
    
    for i, (condition_data, condition_name, color, marker, x_offset) in enumerate(zip(normalized_data, conditions, colors, markers, x_offsets)):
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
                  label=f'{condition_name} measurements', edgecolors='black', linewidth=0.5)
    
    # Add perfect correlation line (y = x)
    min_dist, max_dist = min(tape_distances), max(tape_distances)
    ax.plot([min_dist, max_dist], [min_dist, max_dist], 'k--', linewidth=2, 
            label='Perfect Correlation (y=x)', alpha=0.7)
    
    # Fit and plot overall trendline using original (non-offset) positions
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_tape_distances, all_camera_distances)
    trendline_y = [slope * x + intercept for x in [min_dist, max_dist]]
    
    ax.plot([min_dist, max_dist], trendline_y, 'purple', linewidth=3, 
            label=f'Overall Trendline (y={slope:.3f}x+{intercept:.2f})\nR²={r_value**2:.3f}')
    
    # Formatting
    ax.set_xlabel('Tape Measure Distance [mm]', fontsize=18)
    ax.set_ylabel('Camera Distance [mm]', fontsize=18)
    ax.set_title('Camera Distance vs Tape Measure Distance\n(Normalized to 20mm)', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    ax.set_xlim(min_dist - 2, max_dist + 2)
    ax.set_ylim(min_dist - 2, max_dist + 2)
    
    # Set x-axis ticks to the original tape distances (without offset)
    ax.set_xticks(tape_distances)
    ax.set_xticklabels([f'{int(d)}' for d in tape_distances])
    
    # Increase font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.legend(loc='lower right', fontsize=18)
    
    # Add statistics text box (using original positions for residuals)
    residuals = np.array(all_camera_distances) - np.array(all_tape_distances)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    stats_text = f'RMSE: {rmse:.2f} mm\n'
    stats_text += f'MAE: {mae:.2f} mm\n'
    stats_text += f'Mean Residual: {np.mean(residuals):.2f} mm\n'
    stats_text += f'Std Residual: {np.std(residuals):.2f} mm'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def create_residual_boxplot(tape_distances, normalized_data):
    """Create boxplot of residuals as a function of distance"""
    conditions = ['0°', '30°', '60°']
    colors = ['#E69F00', '#56B4E9', '#009E73'] # Colorblind-friendly: Orange, Sky Blue, Bluish Green
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate residuals for each condition and distance
    for i, (condition_data, condition_name, color) in enumerate(zip(normalized_data, conditions, colors)):
        # Create boxplots for residuals at each distance
        positions = tape_distances + (i - 1) * 1.2  # Increased offset for better separation
        boxplot_data = []
        
        for j, tape_dist in enumerate(tape_distances):
            camera_measurements = condition_data[j, :]
            camera_measurements = camera_measurements[~np.isnan(camera_measurements)]
            
            # Calculate residuals (camera - tape)
            residuals = camera_measurements - tape_dist
            boxplot_data.append(residuals)
        
        # Create boxplot with increased width
        bp = ax.boxplot(boxplot_data, positions=positions, widths=1,
                       patch_artist=True, manage_ticks=False,
                       whis=[0, 100],  # Extend whiskers to min/max (0th and 100th percentiles)
                       boxprops=dict(facecolor=color, alpha=0.7, linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=1.5, color='red'))
    
    # Add zero line for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero residual')
    
    # Formatting
    ax.set_xlabel('Tape Measure Distance [mm]', fontsize=18)
    ax.set_ylabel('Residual (Camera - Tape) [mm]', fontsize=18)
    ax.set_title('Measurement Residuals vs Distance', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks to tape distances
    ax.set_xticks(tape_distances)
    ax.set_xticklabels([f'{int(d)}' for d in tape_distances])
    
    # Increase font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Create custom legend
    legend_elements = []
    for condition, color in zip(conditions, colors):
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=condition))
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', label='Zero residual'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18)
    
    # Calculate and display overall statistics
    all_residuals = []
    condition_stats = []
    
    for i, (condition_data, condition_name) in enumerate(zip(normalized_data, conditions)):
        condition_residuals = []
        for j, tape_dist in enumerate(tape_distances):
            camera_measurements = condition_data[j, :]
            camera_measurements = camera_measurements[~np.isnan(camera_measurements)]
            residuals = camera_measurements - tape_dist
            condition_residuals.extend(residuals)
        
        all_residuals.extend(condition_residuals)
        mean_res = np.mean(condition_residuals)
        std_res = np.std(condition_residuals)
        condition_stats.append(f'{condition_name}: μ={mean_res:.2f}, σ={std_res:.2f} mm')
    
    # Add statistics text box
    overall_mean = np.mean(all_residuals)
    overall_std = np.std(all_residuals)
    
    stats_text = 'Residual Statistics:\n'
    for stat in condition_stats:
        stats_text += stat + '\n'
    stats_text += f'Overall: μ={overall_mean:.2f}, σ={overall_std:.2f} mm'
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def main():
    """Main function to run the analysis"""
    filename = 'CameraCalibrationData.csv'
    
    # Load and process data
    tape_distances, camera_0deg, camera_30deg, camera_60deg = load_and_process_data(filename)
    
    # Normalize data to start at 20mm
    camera_data = [camera_0deg, camera_30deg, camera_60deg]
    normalized_data = normalize_to_20mm(camera_data, tape_distances)
    
    # Create scatter plot
    fig1, ax1 = create_scatter_plot(tape_distances, normalized_data)
    plt.savefig('camera_calibration_scatter_plot.png', dpi=300, bbox_inches='tight')
    
    # Create residual boxplot
    fig2, ax2 = create_residual_boxplot(tape_distances, normalized_data)
    plt.savefig('camera_calibration_residual_boxplot.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print("Camera Calibration Analysis Results:")
    print("=" * 50)
    
    conditions = ['0°', '30°', '60°']
    for i, (condition_data, condition_name) in enumerate(zip(normalized_data, conditions)):
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
    for condition_data in normalized_data:
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
    
    plt.show()

if __name__ == "__main__":
    main()