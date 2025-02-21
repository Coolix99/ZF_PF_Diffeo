import numpy as np
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)

def project_image_to_surface(surface, image,scale,name):
    """
    Projects a 3D image onto a surface mesh.

    Args:
        surface (pv.PolyData): The surface mesh.
        image (np.ndarray): The 3D image.

    Returns:
        pv.PolyData: Updated surface mesh with projected image data.
    """
    # Extract surface vertex coordinates
    vertices = np.array(surface.points)

    # Get nonzero voxel indices and values
    nonzero_indices = np.argwhere(image > 0)  # (N, 3) indices
    nonzero_values = image[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

    if nonzero_values.size == 0:
        logger.warning("No nonzero pixels in the image. Skipping projection.")
        return surface

    # Build KDTree for nearest neighbor search
    tree = cKDTree(vertices)

    # Find nearest surface vertex for each nonzero voxel
    nearest_vertex_ids = tree.query(nonzero_indices*scale)[1]

    # Compute max intensity per vertex
    max_intensity = np.zeros(len(vertices))
    np.maximum.at(max_intensity, nearest_vertex_ids, nonzero_values)

    # Count pixels per vertex
    pixel_count = np.zeros(len(vertices), dtype=int)
    np.add.at(pixel_count, nearest_vertex_ids, 1)

    # Add data to the surface mesh
    surface[name+"_max_intensity"] = max_intensity
    surface[name+"_pixel_count"] = pixel_count

    return surface

def project_df_surface(surface, df, pos_columns, feature_columns):
    """
    Projects a DataFrame onto a surface mesh.

    Args:
        surface (pv.PolyData): The surface mesh.
        df (pd.DataFrame): The DataFrame containing point positions and features.
        pos_columns (list of str): Column names for X, Y, Z coordinates (already scaled).
        feature_columns (list of str): Column names for features to be projected.

    Returns:
        pv.PolyData: Updated surface mesh with projected feature data.
    """
    # Extract surface vertex coordinates
    vertices = np.array(surface.points)

    # Extract positions from df (Z, Y, X format)
    positions = df[pos_columns].values  # (N, 3)

    if positions.shape[0] == 0:
        logger.warning("No valid positions found in the DataFrame. Skipping projection.")
        return surface

    # Build KDTree for nearest neighbor search
    tree = cKDTree(vertices)

    # Find nearest surface vertex for each point in df
    nearest_vertex_ids = tree.query(positions)[1]

    # Initialize storage for projected features
    for feature in feature_columns:
        feature_values = df[feature].values

        # Compute sum and count per vertex for averaging
        sum_feature = np.zeros(len(vertices))
        np.add.at(sum_feature, nearest_vertex_ids, feature_values)

        pixel_count = np.zeros(len(vertices), dtype=int)
        np.add.at(pixel_count, nearest_vertex_ids, 1)

        # Compute average value per vertex
        avg_feature = np.zeros(len(vertices))
        valid = pixel_count > 0  # Avoid division by zero
        avg_feature[valid] = sum_feature[valid] / pixel_count[valid]

        # Add averaged feature to the surface
        surface[feature + "_avg"] = avg_feature
        surface[feature + "_count"] = pixel_count

    return surface
