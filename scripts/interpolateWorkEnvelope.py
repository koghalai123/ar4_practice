import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
from pathlib import Path

class WorkEnvelopeRefiner:
    def __init__(self, data_files=None):
        """Initialize with one or more data files"""
        self.points = np.empty((0, 3))
        self.reachable = np.empty(0, dtype=bool)
        self.iteration = 0
        self.history = []
        
        if data_files:
            if isinstance(data_files, str):
                data_files = [data_files]
            for f in data_files:
                self.load_data(f)

    def load_data(self, data_file):
        """Load data from CSV file and merge with existing data"""
        try:
            # Load new data
            new_data = np.genfromtxt(data_file, delimiter=',', comments='#')
            if new_data.size == 0:
                return
            
            if new_data.ndim == 1:
                new_data = new_data.reshape(1, -1)
                
            new_points = new_data[:, :3]
            new_reachable = new_data[:, 3].astype(bool) if new_data.shape[1] > 3 else np.zeros(len(new_data), dtype=bool)
            
            # Merge with existing data
            if len(self.points) > 0:
                # Remove duplicates before merging
                dist_matrix = np.linalg.norm(new_points[:, np.newaxis] - self.points, axis=2)
                is_new = np.all(dist_matrix > 1e-6, axis=1)
                new_points = new_points[is_new]
                new_reachable = new_reachable[is_new]
                
                self.points = np.vstack((self.points, new_points))
                self.reachable = np.concatenate((self.reachable, new_reachable))
            else:
                self.points = new_points
                self.reachable = new_reachable
                
            self.history.append(data_file)
            print(f"Loaded {len(new_points)} points from {data_file}")
            
        except Exception as e:
            print(f"Error loading {data_file}: {str(e)}")
            raise

    def find_boundary_samples(self, max_distance=0.5):
        distances = np.linalg.norm(self.points[~self.reachable, np.newaxis] - self.points[self.reachable], axis=2)
        distancesReachable = np.linalg.norm(self.points[self.reachable, np.newaxis] - self.points[self.reachable], axis=2)
        minNeighborDist = 0.08
        minNumNeighbors = 11
        newPoints_screened = np.empty((0, 3))  # Initialize as an empty array with shape (0, 3)
        min_distance = 0.028
        
        closeNeighborsLogical = distancesReachable <minNeighborDist
        closeNeighborsNum = closeNeighborsLogical.sum(axis = 0)
        lonelyPoints = closeNeighborsNum<minNumNeighbors
        
        k = 5
        indices = np.argpartition(distances, k, axis=0)[:k]
        newPoints = (self.points[~self.reachable][indices] - self.points[self.reachable][np.newaxis, :, :])/2 + self.points[self.reachable][np.newaxis, :, :]
        newPoints_flattened = newPoints.reshape(-1, 3)
        
        for i in newPoints_flattened:
            distances2 = np.linalg.norm(self.points - i, axis=1)
            if not np.any(distances2 < min_distance):
                newPoints_screened = np.vstack((newPoints_screened, i.reshape(1, 3)))  # Add i with the correct shape
        
        reachablePoints = self.points[self.reachable][lonelyPoints]
        
        distances = np.linalg.norm(self.points[~self.reachable, np.newaxis] -reachablePoints, axis=2)
        minNeighborDist = 0.08
        
        k = 100
        indices = np.argpartition(distances, k, axis=0)[:k]
        newPoints = (self.points[~self.reachable][indices] - reachablePoints[np.newaxis, :, :])/2 + reachablePoints[np.newaxis, :, :]
        newPoints_flattened = newPoints.reshape(-1, 3)
        
        for i in newPoints_flattened:
            distances2 = np.linalg.norm(self.points - i, axis=1)
            if not np.any(distances2 < min_distance):
                newPoints_screened = np.vstack((newPoints_screened, i.reshape(1, 3))) 
        return newPoints_screened

    def save_samples(self, samples, output_file):
        """Save samples to CSV using numpy"""
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        np.savetxt(
            output_file,
            samples,
            delimiter=',',
            header='x,y,z',
            comments='',
            fmt='%.6f'
        )
        print(f"Saved {len(samples)} boundary samples to {output_file}")

    def visualize(self, new_samples=None):
        """Visualize current state with optional new samples"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all points
        ax.scatter(
            self.points[self.reachable, 0],
            self.points[self.reachable, 1],
            self.points[self.reachable, 2],
            c='green', alpha=0.6, s=20, label='Reachable'
        )
        ax.scatter(
            self.points[~self.reachable, 0],
            self.points[~self.reachable, 1],
            self.points[~self.reachable, 2],
            c='red', alpha=0.2, s=20, label='Unreachable'
        )
        
        # Plot new samples if provided
        if new_samples is not None and len(new_samples) > 0:
            ax.scatter(
                new_samples[:, 0], new_samples[:, 1], new_samples[:, 2],
                c='blue', marker='o', s=40, alpha=0.8, label='New Samples'
            )
        
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title(f'Work Envelope ({len(self.points)} points)')
        ax.legend()
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    refiner = WorkEnvelopeRefiner([
        'results0_2.csv',
        'results3.csv',
        'results4.csv',
        'results5.csv',
        'results6.csv',
        'results7.csv',
        'results8.csv',
        'results9.csv',
        'results10.csv',
        'results11.csv',
        'results12.csv',
    ])
    
    # Find boundary samples
    print("Finding boundary samples...")
    new_samples = refiner.find_boundary_samples(max_distance=0.5)
    print(f"Found {len(new_samples)} boundary samples")
    
    # Save for testing
    refiner.save_samples(new_samples, 'boundary_samples.csv')
    
    # Visualize
    refiner.visualize(new_samples)
    
    # After testing, merge new results:
    # refiner.load_data('test_results.csv')

if __name__ == "__main__":
    main()