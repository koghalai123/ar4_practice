import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('calibrationData.csv')

# Group columns by prefix
dQ = df[[col for col in df.columns if col.startswith('dQ') and not col.startswith('dQMat')]].iloc[0,:].to_numpy()
dL = df[[col for col in df.columns if col.startswith('dL') and not col.startswith('dLMat')]].iloc[0,:].to_numpy()
dX = df[[col for col in df.columns if col.startswith('dX') and not col.startswith('dXMat')]].iloc[0,:].to_numpy()
dQMat = df[[col for col in df.columns if col.startswith('dQMat')]].to_numpy()
dLMat = df[[col for col in df.columns if col.startswith('dLMat')]].to_numpy()
dXMat = df[[col for col in df.columns if col.startswith('dXMat')]].to_numpy()

# Create numpy arrays for the other (scalar) terms
noiseMagnitude = df['noiseMagnitude'].iloc[0]
n = df['n'].iloc[0]
avgAccMat = df['avgAccMat']

plt.figure(figsize=(10, 6))

# Plot dQMat
cmap = plt.get_cmap("Blues")
arr =np.abs( np.cumsum(dQMat,axis=0)-dQ ) # shape: (num_rows, num_columns)
num_cols = arr.shape[1]
x = np.arange(arr.shape[0])  # row indices
for i in range(num_cols):
    color = cmap(0.3 + 0.7 * i / num_cols)  # Vary shade from lighter to darker
    if np.abs(arr[-1, i])<0.001:
        plt.plot(x, arr[:, i], label=f'Joint Angle Offset {i+1}', color=color)

    else:
        plt.plot(x, arr[:, i], label=f'Joint Angle Offset {i+1}', color=color, linestyle='dashed')



# Plot dLMat
cmap = plt.get_cmap("Greens")
arr = np.abs(np.cumsum(dLMat,axis=0)-dL)  # shape: (num_rows, num_columns)
num_cols = arr.shape[1]
x = np.arange(arr.shape[0])  # row indices
for i in range(num_cols):
    color = cmap(0.3 + 0.7 * i / num_cols)  # Vary shade from lighter to darker
    if np.abs(arr[-1, i])<0.001:
        plt.plot(x, arr[:, i], label=f'Joint Length Change {i+1}', color=color)

    else:
        plt.plot(x, arr[:, i], label=f'Joint Length Change {i+1}', color=color, linestyle='dashed')


# Plot dXMat
cmap = plt.get_cmap("Purples")
arr = np.abs(np.cumsum(dXMat,axis=0)-dX  )# shape: (num_rows, num_columns)
num_cols = arr.shape[1]
x = np.arange(arr.shape[0])  # row indices
for i in range(num_cols):
    color = cmap(0.3 + 0.7 * i / num_cols)  # Vary shade from lighter to darker
    if np.abs(arr[-1, i])<0.001:
        plt.plot(x, arr[:, i], label=f'Base Offset Parameters {i+1}', color=color)

    else:
        plt.plot(x, arr[:, i], label=f'Base Offset Parameters {i+1}', color=color, linestyle='dashed')

plt.xlabel('Iterations')
plt.ylabel('Difference from Actual Parameter with n=' + str(n)+'Noise Magnitude: '+ str(noiseMagnitude))
plt.title('Iterative Linearization using Least Squares Error Calibration Results')

box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.yscale('log')

plt.savefig('calibrationResultsParameterError.png', dpi=300, bbox_inches='tight')






plt.figure(figsize=(10, 6))
plt.plot(np.arange(avgAccMat.shape[0]), avgAccMat, label='Avg Accuracy Error', color='black')
plt.xlabel('Iterations')
plt.ylabel('Average Difference from Desired Position with n=' + str(n)+'Noise Magnitude: '+ str(noiseMagnitude))
plt.title('Average Position Error for Iterative Linearization Calibration')
plt.yscale('log')

plt.savefig('calibrationResultsAveragePositionError.png', dpi=300, bbox_inches='tight')












plt.show()

print('done')