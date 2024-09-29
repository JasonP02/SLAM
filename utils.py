import numpy as np

def filter_outliers(points_3d, threshold=3):
    """
    Filters out outliers from 3D points using the Z-score method.
    
    Parameters:
    points_3d (np.ndarray): Array of 3D points (N, 3).
    threshold (float): Z-score threshold to identify outliers.
    
    Returns:
    np.ndarray: Filtered 3D points.
    """
    if points_3d.size == 0:
        return points_3d

    # Calculate the Z-scores for each dimension
    z_scores = np.abs((points_3d - points_3d.mean(axis=0)) / points_3d.std(axis=0))
    
    # Filter points where all dimensions have Z-scores below the threshold
    filtered_points = points_3d[(z_scores < threshold).all(axis=1)]
    
    return filtered_points