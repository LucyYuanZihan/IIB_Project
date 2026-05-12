import os
import numpy as np

def process_seg2tunnel_pointclouds(input_dir, output_dir):
    """
    Reads raw point cloud .txt files from the Seg2Tunnel dataset, 
    applies Unit Sphere coordinate normalization and percentile-based 
    intensity scaling, and saves them to a new directory.
    Assumes columns: [x, y, z, intensity, label(s)...]
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all .txt files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"Processing: {filename}...")
            
            # Load the point cloud data
            # If your text files use commas instead of spaces, add delimiter=','
            data = np.loadtxt(input_path)
            
            # Separate features (Columns 0, 1, 2 are x, y, z; Column 3 is intensity)
            xyz = data[:, :3]
            intensity = data[:, 3]
            
            # Extract labels if they exist (Columns 4 and beyond)
            labels = data[:, 4:] if data.shape[1] > 4 else None
            
            # --- 1. PointNet++ Unit Sphere Coordinate Normalization ---
            # Shift the point cloud to its centroid (center of mass)
            centroid = np.mean(xyz, axis=0)
            xyz_centered = xyz - centroid
            
            # Find the maximum Euclidean distance from the centroid to any point
            max_distance = np.max(np.sqrt(np.sum(xyz_centered**2, axis=1)))
            
            # Scale all points by this maximum distance to fit within [-1, 1]
            if max_distance > 0:
                xyz_norm = xyz_centered / max_distance
            else:
                xyz_norm = xyz_centered
                
            # --- 2. Intensity Normalization (Seg2Tunnel method) ---
            # Calculate 1% and 99% percentiles
            i_min = np.percentile(intensity, 1)
            i_max = np.percentile(intensity, 99)
            
            # Apply piecewise min-max scaling
            if i_max - i_min == 0:
                intensity_norm = np.zeros_like(intensity)
            else:
                intensity_norm = (intensity - i_min) / (i_max - i_min)
                
            # Clip strictly to the [0, 1] interval
            intensity_norm = np.clip(intensity_norm, 0.0, 1.0)
            intensity_norm = intensity_norm.reshape(-1, 1)
            
            # --- 3. Recombine and Save ---
            if labels is not None:
                processed_data = np.hstack((xyz_norm, intensity_norm, labels))
            else:
                processed_data = np.hstack((xyz_norm, intensity_norm))
            
            # Format output: 6 decimal places for coordinates/intensity, integers for labels
            formats = ['%.6f', '%.6f', '%.6f', '%.6f']
            if labels is not None:
                formats += ['%d'] * labels.shape[1]
                
            np.savetxt(output_path, processed_data, fmt=formats)
            
    print("All point clouds processed and saved successfully! The aspect ratio is now preserved.")

# === Configuration ===
# Replace these strings with your actual folder paths
INPUT_FOLDER = "/mnt/c/Users/zy349/Documents/Points2NeRF/Seg2Tunnel/T2"
OUTPUT_FOLDER = "/mnt/c/Users/zy349/Documents/Points2NeRF/Seg2Tunnel/Normalised/T2"

if __name__ == "__main__":
    process_seg2tunnel_pointclouds(INPUT_FOLDER, OUTPUT_FOLDER)

