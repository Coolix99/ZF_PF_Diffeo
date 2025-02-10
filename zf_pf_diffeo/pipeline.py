import os
import numpy as np
import pandas as pd
import pyvista as pv
import logging
from tqdm import tqdm
from collections import defaultdict
import pickle

from zf_pf_geometry.utils import make_path
from zf_pf_geometry.metadata_manager import should_process, write_JSON, get_JSON, calculate_dict_checksum
from zf_pf_diffeo.reference_geometries import create_refGeometry, create_TempRefGeometry




logger = logging.getLogger(__name__)

def setup_folders(base_dirs, data_name):
    """Setup and ensure output directories exist."""
    paths = {key: os.path.join(base_dir, data_name) for key, base_dir in base_dirs.items()}
    make_path(paths["output"])
    return paths

def group_by_keys(surface_folders, surface_dir, category_keys):
    """Group datasets based on metadata keys."""
    groups = defaultdict(list)
    metadata_map = {}
    
    for data_name in surface_folders:
        metadata = get_JSON(os.path.join(surface_dir, data_name))
        if metadata and "Thickness_MetaData" in metadata:
            key_tuple = tuple(metadata["Thickness_MetaData"].get(k, None) for k in category_keys)
            groups[key_tuple].append(data_name)
            metadata_map[data_name] = metadata
    
    return groups, metadata_map

def should_process_group(group, metadata_map, output_dir, output_key):
    """Determine if a group needs reprocessing based on contained datasets and input files."""
    input_data = {name: metadata_map[name]["Thickness_MetaData"] for name in group}
    input_files = {name: metadata_map[name]["Thickness_MetaData"]["Surface file"] for name in group}
    input_data_checksum = calculate_dict_checksum(input_data)
    output_data = get_JSON(output_dir)

    if output_data is None or output_key not in output_data:
        return input_data, input_files, input_data_checksum
    
    previous_contained = set(output_data[output_key].get("contained_datanames", []))
    previous_files = set(output_data[output_key].get("input_files", []))
    
    if not set(group).issubset(previous_contained) or not set(input_files.values()).issubset(previous_files):
        return input_data, input_files, input_data_checksum
    
    if "input_data_checksum" in output_data[output_key] and input_data_checksum == output_data[output_key]["input_data_checksum"]:
        logger.info("Skipping processing as input data has not changed.")
        return False
    
    return input_data, input_files, input_data_checksum

def save_2d_mesh(nodes, triangles, file_path):
    """Saves a 2D mesh consisting of nodes and triangles to a .npz file."""
    np.savez(file_path, nodes=nodes, triangles=triangles)
    logger.info(f"Saved 2D mesh to {file_path}")

def do_referenceGeometries(surface_dir, category_keys, output_dir):
    """
    Computes representative surfaces for given category groups.
    
    Args:
        surface_dir (str): Directory containing 3D surface data.
        category_keys (list): Keys used to group surfaces for averaging.
        output_dir (str): Path to save reference geometries.
    """
    logger.info("Processing reference geometries.")
    base_dirs = {"surface": surface_dir, "output": output_dir}
    
    surface_folders = [
        item for item in os.listdir(surface_dir) if os.path.isdir(os.path.join(surface_dir, item))
    ]
    
    groups, metadata_map = group_by_keys(surface_folders, surface_dir, category_keys)
    
    for key_tuple, group in tqdm(groups.items(), desc="Processing reference geometries", unit="group"):
        representative_name = "_".join(map(str, key_tuple))
        paths = setup_folders(base_dirs, representative_name)
        
        res = should_process_group(group, metadata_map, paths["output"], "reference_geometry")
        if not res:
            logger.info(f"Skipping group {key_tuple}: No processing needed.")
            continue
        
        input_data, input_files, input_checksum = res
        
        # Process reference geometries
        meshes3d = [pv.read(os.path.join(surface_dir, name, input_files[name])) for name in group]
        avg_coeff, (nodes, triangles) = create_refGeometry(meshes3d, n_harmonics=10, n_coords=40)
        
        ref_mesh_file = os.path.join(paths["output"], f"{representative_name}_ref.npz")
        save_2d_mesh(nodes, triangles, ref_mesh_file)
        
        file_paths = {
            "Reference Geometry": ref_mesh_file, 
            "contained_datanames": group, 
            "input_files": list(input_files.values()),
        }
        category_metadata = dict(zip(category_keys, key_tuple))

        res_MetaData = {
            "input_data_checksum": input_checksum, 
            "reference_coefficients": avg_coeff.tolist(),
            **category_metadata,  # Add dynamically extracted metadata (e.g., time in hpf, genotype)
            **file_paths,  # Includes Reference Geometry, contained_datanames, and input_files
        }

        
        write_JSON(paths["output"], "reference_geometry", res_MetaData)
        
        logger.info(f"Saved reference geometry for group {key_tuple} at {ref_mesh_file}.")
    
    logger.info("Reference geometry processing completed.")


def save_interpolated_data(interpolated_nodes, output_dir):
    """
    Saves interpolated mesh data for all time steps.

    Args:
        interpolated_nodes (dict): Dictionary {time: nodes}.
        output_dir (str): Directory to save data.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "interpolated_nodes.pkl")

    with open(file_path, 'wb') as f:
        pickle.dump(interpolated_nodes, f)

    logger.info(f"Saved interpolated nodes to {file_path}")

def do_temporalreferenceGeometries(surfaces2d_dir, time_key, category_keys, output_dir):
    """
    Computes interpolated reference geometries over time.

    Args:
        surfaces2d_dir (str): Directory containing 2D reference surfaces.
        time_key (str): The key defining time steps.
        category_keys (list): Keys used to group data (excluding time).
        output_dir (str): Directory to save interpolated geometries.
    """
    logger.info("Processing temporal reference geometries.")
    base_dirs = {"surfaces2d": surfaces2d_dir, "output": output_dir}

    surface_folders = [
        item for item in os.listdir(surfaces2d_dir) if os.path.isdir(os.path.join(surfaces2d_dir, item))
    ]

    # Group reference geometries by category_keys (ignoring time)
    groups = defaultdict(list)
    metadata_map = {}

    for folder in surface_folders:
        metadata = get_JSON(os.path.join(surfaces2d_dir, folder))
        if metadata and "reference_geometry" in metadata:
            # Extract values from metadata

            time_value = metadata["reference_geometry"].get(time_key, None)
            key_tuple = tuple(metadata["reference_geometry"].get(k, "MISSING") for k in category_keys)

            # Debugging logs
            logger.debug(f"Folder: {folder}, Extracted keys: {key_tuple}, Time: {time_value}")

            # Ensure keys are correctly extracted
            if None in key_tuple or time_value is None:
                logger.warning(f"Skipping {folder} due to missing keys: {key_tuple} or time: {time_value}")
                continue

            groups[key_tuple].append((time_value, folder))
            metadata_map[folder] = metadata
            
    for key_tuple, time_series in tqdm(groups.items(), desc="Processing temporal geometries", unit="group"):
        logger.info(f"Processing group: {key_tuple}")
        logger.debug(f"Time series: {time_series}")

        paths = {"output": os.path.join(output_dir, "_".join(map(str, key_tuple)))}


        input_data = {}
        input_checksums = []

        for time, subfolder in sorted(time_series):  # Ensure time order
            paths_sub = {"output": os.path.join(surfaces2d_dir, subfolder)}
            res = should_process([paths_sub["output"]], ["reference_geometry"], paths_sub["output"], "temporal_reference_geometry")

            if res:
                data, checksum = res
                input_data[time] = data
                input_checksums.append(checksum)


        # Compute a global checksum for all time steps
        input_checksum = calculate_dict_checksum(input_data)

        # Sort by time and extract Fourier coefficients
        times, folders = zip(*sorted(time_series))
        ref_coeffs = {t: np.array(metadata_map[f]["reference_geometry"]["reference_coefficients"]) for t, f in time_series}

        interpolated_nodes,last_mesh = create_TempRefGeometry(ref_coeffs,times)


        # Save interpolated data
        save_interpolated_data(interpolated_nodes, paths["output"])

        # Update metadata
        res_MetaData = {
            "input_data_checksum": input_checksum,
        }
        write_JSON(paths["output"], "temporal_reference_geometry", res_MetaData)

        logger.info(f"Saved temporal reference geometry for {key_tuple}.")
    
    logger.info("Temporal reference geometry processing completed.")


def do_HistPointData(surface_dir, surfaces2d_dir, category_keys, output_dir):
    """
    Generates histograms from point data projected onto deformed 2D surfaces.
    
    Args:
        surface_dir (str): Directory with original 3D surface data.
        surfaces2d_dir (str): Directory with processed 2D surfaces.
        category_keys (list): Keys used for grouping data.
        output_dir (str): Output directory.
    """
    logger.info("Processing histogram point data.")
    base_dirs = {"surface": surface_dir, "surfaces2d": surfaces2d_dir, "output": output_dir}

    surface_folders = [
        item for item in os.listdir(surface_dir) if os.path.isdir(os.path.join(surface_dir, item))
    ]

    for data_name in tqdm(surface_folders, desc="Processing histograms", unit="dataset"):
        paths = setup_folders(base_dirs, data_name)

        res = should_process([paths["surface"], paths["surfaces2d"]], ["surface", "surfaces2d"], paths["output"], "histogram_data")
        if not res:
            logger.info(f"Skipping {data_name}: No processing needed.")
            continue

        input_data, input_checksum = res

        # TODO: Implement point projection and histogram computation

        res_MetaData = update_metadata(input_data["surface"], {}, input_checksum)
        write_JSON(paths["output"], "histogram_data", res_MetaData)

    logger.info("Histogram processing completed.")

def do_temporalHistInterpolation(output_dir):
    """
    Interpolates histograms over time.
    
    Args:
        output_dir (str): Directory containing histogram data.
    """
    logger.info("Processing temporal histogram interpolation.")

    hist_folders = [
        item for item in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, item))
    ]

    for data_name in tqdm(hist_folders, desc="Interpolating histograms", unit="dataset"):
        hist_path = os.path.join(output_dir, data_name)

        # TODO: Implement temporal interpolation for histograms

        logger.info(f"Interpolated histograms for {data_name}")

    logger.info("Temporal histogram interpolation completed.")

def do_all(surface_dir, surfaces2d_dir, category_keys, output_dir):
    """
    Runs the entire pipeline in sequence.
    
    Args:
        surface_dir (str): Directory containing 3D surface data.
        surfaces2d_dir (str): Directory containing 2D surface data.
        category_keys (list): Keys defining categories for grouping.
        output_dir (str): Output directory.
    """
    do_referenceGeometries(surface_dir, category_keys, output_dir)
    do_temporalreferenceGeometries(surfaces2d_dir, "time", category_keys, output_dir)
    do_HistPointData(surface_dir, surfaces2d_dir, category_keys, output_dir)
    do_temporalHistInterpolation(output_dir)

    logger.info("Full pipeline processing completed.")
