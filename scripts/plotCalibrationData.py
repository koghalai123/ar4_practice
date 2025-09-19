import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, FormatStrFormatter
import numpy as np
import os

# Load the CSV file
csv_filename = 'physical_calibration_data_reasonable.csv'
df = pd.read_csv(csv_filename)

# Prepare output image filename base
base_name = os.path.splitext(csv_filename)[0]

# Plot average error values (logarithmic y-axis, more frequent ticks)
plt.figure()
plt.plot(df['Position Error'], label='Position Error')
plt.plot(df['Orientation Error'], label='Orientation Error')
plt.yscale('log')
plt.title('Average Error Values')
plt.xlabel('Iteration')
plt.ylabel('Error [meters/radians]')
plt.legend()
plt.grid(True, which='both', axis='y')

# Find the minimum value among both error columns (ignoring zeros and negatives)
all_errors = pd.concat([df['Position Error'], df['Orientation Error']])
min_val = all_errors[all_errors > 0].min()
if min_val is not None and min_val > 0:
    lower_lim = 10**(np.log10(min_val) - 0.5)
    plt.ylim(bottom=lower_lim)

ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
# Show more significant figures on the y axis
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
ax.tick_params(axis='y', which='minor', labelsize=8)
plt.tight_layout()
plt.savefig(f"{base_name}_avg_error.png", dpi=300)

# Plot estimated target pose values (difference from final value)
plt.figure()
pose_cols = ['Estimated Target X', 'Estimated Target Y', 'Estimated Target Z',
             'Estimated Target Roll', 'Estimated Target Pitch', 'Estimated Target Yaw']
for col in pose_cols:
    diff = df[col] - df[col].iloc[-1]
    plt.plot(diff, label=col)
plt.title('Estimated Target Pose (Difference from Final Value)')
plt.xlabel('Iteration')
plt.ylabel('Value - Final Value [meters/radians]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_name}_est_target_pose.png", dpi=300)

# Plot calibration parameters (difference from final value)
plt.figure()
calib_cols = [col for col in df.columns if 'Estimated' in col and col not in pose_cols]
for col in calib_cols:
    diff = df[col] - df[col].iloc[-1]
    plt.plot(diff, label=col)
plt.title('Calibration Parameters (Difference from Final Value)')
plt.xlabel('Iteration')
plt.ylabel('Value - Final Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{base_name}_calib_params.png", dpi=300)

plt.show()