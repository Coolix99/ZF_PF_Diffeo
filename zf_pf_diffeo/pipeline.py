import os
import numpy as np
import pyvista as pv
import logging
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
import os

from zf_pf_geometry.utils import make_path
from zf_pf_geometry.metadata_manager import should_process, write_JSON, get_JSON, calculate_dict_checksum
from zf_pf_diffeo.reference_geometries import create_refGeometry, create_TempRefGeometry,transfer_data_to_reference


logger = logging.getLogger(__name__)

def setup_folders(base_dirs, data_name):
    """Setup and ensure output directories exist."""
    paths = {key: os.path.join(base_dir, data_name) for key, base_dir in base_dirs.items()}
    make_path(paths["output"])
    return paths

def group_by_keys(surface_folders, surface_dir, category_keys,surface_key):
    """Group datasets based on metadata keys."""
    groups = defaultdict(list)
    metadata_map = {}
    
    for data_name in surface_folders:
        metadata = get_JSON(os.path.join(surface_dir, data_name))
        if metadata and surface_key in metadata:
            key_tuple = tuple(metadata[surface_key].get(k, None) for k in category_keys)
            groups[key_tuple].append(data_name)
            metadata_map[data_name] = metadata
    
    return groups, metadata_map

def should_process_group(group, metadata_map, output_dir, output_key,surface_key, surface_file_key):
    """Determine if a group needs reprocessing based on contained datasets and input files."""
    input_data = {name: metadata_map[name][surface_key] for name in group}
    input_files = {name: metadata_map[name][surface_key][surface_file_key] for name in group}
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

def save_2d_mesh(nodes, triangles, boundary_indices, file_path):
    """
    Saves a 2D mesh consisting of nodes, triangles, and boundary indices to a .npz file.

    Args:
        nodes (ndarray): Array of shape (N, 2) containing mesh node coordinates.
        triangles (ndarray): Array of shape (M, 3) defining mesh connectivity.
        boundary_indices (ndarray): Indices of boundary nodes.
        file_path (str): Path to save the .npz file.
    """
    np.savez(file_path, nodes=nodes, triangles=triangles, boundary_indices=boundary_indices)
    logger.info(f"Saved 2D mesh to {file_path}")

def do_referenceGeometries(surface_dir, category_keys, output_dir,surface_key="Thickness_MetaData",surface_file_key="Surface file"):
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
    
    groups, metadata_map = group_by_keys(surface_folders, surface_dir, category_keys,surface_key)
    logger.debug(f"groups: {groups}")
    for key_tuple, group in tqdm(groups.items(), desc="Processing reference geometries", unit="group"):
        representative_name = "_".join(map(str, key_tuple))
        paths = setup_folders(base_dirs, representative_name)
        
        res = should_process_group(group, metadata_map, paths["output"], "reference_geometry",surface_key=surface_key,surface_file_key=surface_file_key)
        if not res:
            logger.info(f"Skipping group {key_tuple}: No processing needed.")
            continue
        
        input_data, input_files, input_checksum = res
        
        # Process reference geometries
        meshes3d = [pv.read(os.path.join(surface_dir, name, input_files[name])) for name in group]
    
        avg_coeff, (nodes, triangles, boundary_indices) = create_refGeometry(meshes3d, n_harmonics=10, n_coords=40)

        ref_mesh_file = os.path.join(paths["output"], f"{representative_name}_ref.npz")
        save_2d_mesh(nodes, triangles, boundary_indices, ref_mesh_file)

        
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

def save_interpolated_data(interpolated_nodes, last_mesh, output_path, last_mesh_file):
    """
    Saves interpolated node positions and last mesh (triangles) for reference geometry.

    Args:
        interpolated_nodes (dict): {time: deformed_mesh_nodes}
        last_mesh (tuple): (nodes, triangles) from the last time step
        output_path (str): Path to save interpolated data
        last_mesh_file (str): Path to save last mesh triangulation
    """
    os.makedirs(output_path, exist_ok=True)

    # Convert time keys to strings to avoid TypeError
    interpolated_nodes_str_keys = {str(time): nodes for time, nodes in interpolated_nodes.items()}

    # Save interpolated nodes
    nodes_file = os.path.join(output_path, "interpolated_nodes.npz")
    np.savez(nodes_file, **interpolated_nodes_str_keys)

    # Save last mesh separately (nodes & triangles)
    nodes, triangles,boundary_indices = last_mesh
    np.savez(last_mesh_file, nodes=nodes, triangles=triangles, boundary_indices=boundary_indices)

    logger.info(f"Saved interpolated nodes to {nodes_file}")
    logger.info(f"Saved last mesh to {last_mesh_file}")

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


        # Save interpolated data along with last_mesh
        last_mesh_file = os.path.join(paths["output"], "last_mesh.npz")
        save_interpolated_data(interpolated_nodes, last_mesh, paths["output"], last_mesh_file)

        # Update metadata
        res_MetaData = {
            "input_data_checksum": input_checksum,
            "last_mesh_file": last_mesh_file,
        }
        write_JSON(paths["output"], "temporal_reference_geometry", res_MetaData)

        logger.info(f"Saved temporal reference geometry for {key_tuple}.")
    
    logger.info("Temporal reference geometry processing completed.")

def get_reference_geometry(reference_dir, category_keys, metadata,surface_key):
    """
    Finds the correct reference geometry file based on metadata.

    Args:
        reference_dir (str): Path where reference geometries are stored.
        category_keys (list): Keys used for grouping.
        metadata (dict): Metadata from the current 3D surface.

    Returns:
        tuple: (reference nodes, reference triangles, boundary indices) if found, else (None, None, None).
    """
    key_tuple = tuple(metadata[surface_key].get(k, "MISSING") for k in category_keys)
    reference_name = "_".join(map(str, key_tuple))
    ref_file = os.path.join(reference_dir, reference_name, f"{reference_name}_ref.npz")

    if not os.path.exists(ref_file):
        logger.warning(f"Reference geometry not found for {reference_name}")
        return None, None, None

    ref_data = np.load(ref_file)
    return ref_data["nodes"], ref_data["triangles"], ref_data["boundary_indices"],reference_name

def do_HistPointData(surface_dir, reference_dir, category_keys, output_dir,surface_key,surface_file_key):
    """
    Projects 3D point data onto deformed 2D reference surfaces.

    Args:
        surface_dir (str): Directory containing 3D surfaces.
        reference_dir (str): Directory containing computed 2D reference geometries.
        category_keys (list): Keys used for grouping data.
        output_dir (str): Path to save processed histograms.
    """
    logger.info("Processing histogram point data.")

    surface_folders = [
        item for item in os.listdir(surface_dir) if os.path.isdir(os.path.join(surface_dir, item))
    ]

    hist_map = {}

    for data_name in tqdm(surface_folders, desc="Processing histograms", unit="dataset"):
        surface_path = os.path.join(surface_dir, data_name)
        metadata = get_JSON(surface_path)

        if not metadata or surface_key not in metadata:
            logger.warning(f"Skipping {data_name}: Missing metadata")
            continue

        ref_nodes, ref_triangles, ref_boundaries, reference_name = get_reference_geometry(reference_dir, category_keys, metadata,surface_key=surface_key)
        if ref_nodes is None:
            continue

        mesh_3d = pv.read(os.path.join(surface_path, metadata[surface_key][surface_file_key]))

        # Get indices mapping 3D points to reference nodes
        indices = transfer_data_to_reference(ref_nodes, ref_triangles, ref_boundaries, mesh_3d)

        # Initialize storage for a new reference group if not exists
        if reference_name not in hist_map:
            hist_map[reference_name] = {
                "ref_nodes": ref_nodes,
                "data": defaultdict(lambda: defaultdict(list))  # {feature: {node_idx: [values]}}
            }

        # Iterate through all point data keys
        for key in mesh_3d.point_data.keys():
            values = np.array(mesh_3d.point_data[key])

            # Assign values to corresponding reference nodes
            for i, node_idx in enumerate(indices):
                hist_map[reference_name]["data"][key][node_idx].append(values[i])

    # Process collected data into histograms and statistics
    for reference_name, ref_data in hist_map.items():
        ref_nodes = ref_data["ref_nodes"]
        processed_data = {}

        for key, node_dict in ref_data["data"].items():
            node_values = []

            for node_idx in range(len(ref_nodes)):  # Ensure ordering
                values = node_dict.get(node_idx, [])

                if values:
                    values = np.array(values)
                else:
                    node_values.append([])


            # Store histograms and statistics
            processed_data[f"{key}"] = node_values

        # Save results
        output_path = os.path.join(output_dir, reference_name)
        make_path(output_path)
        for key in processed_data:
            processed_data[key] = np.array(processed_data[key], dtype=object)
        np.savez(os.path.join(output_path, "histogram_data.npz"), **processed_data)

        # Update metadata
        res_MetaData = {
            #"input_data_checksum": calculate_dict_checksum(processed_data),
            "histogram_data": os.path.join(output_path, "histogram_data.npz"),
        }
        write_JSON(output_path, "histogram_data", res_MetaData)

        logger.info(f"Saved histogram data for reference {reference_name}.")

    logger.info("Histogram processing completed.")



def get_temp_reference_geometry(temp_maps_dir, temp_key,category_keys, metadata,surface_key):
    """
    Finds the correct reference geometry file based on metadata.

    Args:
        temp_maps_dir (str): Path where temp reference geometries are stored.
        temp_key  (str): Key defining time steps.
        category_keys (list): Keys used for grouping.
        metadata (dict): Metadata from the current 3D surface.

    Returns:
        tuple: (reference nodes, reference triangles, boundary indices) if found, else (None, None, None).
    """
    key_tuple = tuple(metadata[surface_key].get(k, "MISSING") for k in category_keys)
    reference_name = "_".join(map(str, key_tuple))
    interpolated_nodes_file = os.path.join(temp_maps_dir, reference_name, f"interpolated_nodes.npz")
    last_mesh_file = os.path.join(temp_maps_dir, reference_name, f"last_mesh.npz")

    if not os.path.exists(interpolated_nodes_file) or not os.path.exists(last_mesh_file):
        logger.warning(f"Reference geometry not found for {reference_name}")
        return

    interpolated_nodes = np.load(interpolated_nodes_file)
    time=metadata[surface_key][temp_key]
    nodes_t = interpolated_nodes[str(time)]
    last_mesh = np.load(last_mesh_file)
    return nodes_t, last_mesh["triangles"], last_mesh["boundary_indices"],reference_name,time


def do_temporalHistInterpolation(surface_dir, temp_maps_dir, temp_key, category_keys, value_functions, surface_key="Thickness_MetaData", surface_file_key="Surface file"):
    """
    Transfers 3D point data onto time-dependent reference geometries and interpolates missing data.

    Args:
        surface_dir (str): Directory containing 3D surfaces.
        temp_maps_dir (str): Directory containing computed 2D reference geometries.
        temp_key (str): Key representing time in metadata.
        category_keys (list): Keys used for grouping data.
        value_functions (dict): Mapping of feature keys to functions that extract scalar values from histogram data.
        surface_key (str, optional): Key for surface metadata.
        surface_file_key (str, optional): Key for surface file in metadata.
    """
    logger.info("Processing temporal histogram interpolation.")

    surface_folders = [
        item for item in os.listdir(surface_dir) if os.path.isdir(os.path.join(surface_dir, item))
    ]

    hist_map = {}

    # Collect data for all reference names and times
    for data_name in tqdm(surface_folders, desc="Processing histograms", unit="dataset"):
        surface_path = os.path.join(surface_dir, data_name)
        metadata = get_JSON(surface_path)
        if not metadata or surface_key not in metadata:
            logger.warning(f"Skipping {data_name}: Missing metadata")
            continue

        mesh_3d = pv.read(os.path.join(surface_path, metadata[surface_key][surface_file_key]))

        ref_nodes, ref_triangles, ref_boundaries, reference_name, time = get_temp_reference_geometry(
            temp_maps_dir, temp_key, category_keys, metadata, surface_key=surface_key
        )

        # Get indices mapping 3D points to reference nodes
        indices = transfer_data_to_reference(ref_nodes, ref_triangles, ref_boundaries, mesh_3d)

        if reference_name not in hist_map:
            hist_map[reference_name] = {
                "ref_nodes": ref_nodes, 
                "data": defaultdict(lambda: defaultdict(list)),  # {time: {feature: {node_idx: [values]}}}
                "times": set()
            }

        hist_map[reference_name]["times"].add(time)

        # Ensure `hist_map[reference_name]["data"][time][key]` is initialized with an empty list for each node
        for key in mesh_3d.point_data.keys():
            if key not in hist_map[reference_name]["data"][time]:
                hist_map[reference_name]["data"][time][key] = [[] for _ in range(len(ref_nodes))]  # Initialize empty lists for each node

        # Iterate through all point data keys
        for key in mesh_3d.point_data.keys():
            values = np.array(mesh_3d.point_data[key])

            # Assign values to corresponding reference nodes
            for i, node_idx in enumerate(indices):
                if 0 <= node_idx < len(ref_nodes):  # Ensure node_idx is in bounds
                    hist_map[reference_name]["data"][time][key][node_idx].append(values[i])
                    
                else:
                    logger.warning(f"Skipping out-of-bounds node_idx {node_idx} for reference {reference_name}, time {time}")
                    raise


    for reference_name, ref_data in hist_map.items():
        ref_nodes = ref_data["ref_nodes"]
        collected_times = sorted(ref_data["times"])

        print(reference_name)
        print(hist_map[reference_name]["data"].keys())
        print(hist_map[reference_name]["data"][60].keys())

        # Ensure all intermediate times exist
        min_time, max_time = min(collected_times), max(collected_times)
        full_time_range = list(range(min_time, max_time + 1))  # Fill gaps in time

        processed_data = {}

        # Step 1: Process raw histogram data (before applying value functions)
        structured_histograms = {}  # Store per time

        for time in collected_times:
            structured_histograms[time] = {}  # Initialize for this time

            for feature_key in ref_data["data"][time]:
                hist_data = ref_data["data"][time][feature_key]  # Raw histogram data at this time
                structured_data = []

                for node_idx in range(len(ref_nodes)):  # Ensure ordering
                    values = hist_data[node_idx] 

                    if values:
                        structured_data.append(np.array(values))
                    else:
                        structured_data.append(np.nan)  # Empty nodes get NaN

                structured_histograms[time][feature_key] = np.array(structured_data, dtype=object)
               
        # Step 2: Apply all value functions
        for feature_key, func in value_functions.items():
            extracted_values = {}

            for time in collected_times:
                print(time)

                if time in structured_histograms:
                    extracted_values[time] = func(structured_histograms[time])
                    #print(extracted_values[time])
                    print(extracted_values[time].shape)

            # Convert to sorted numpy arrays for interpolation
            times_with_data = np.array(sorted(extracted_values.keys()))
            values_with_data = np.array([extracted_values[t] for t in times_with_data])

            # Interpolate missing values
            interpolator = interp1d(times_with_data, values_with_data, axis=0, kind="linear", fill_value="extrapolate")
            interpolated_values = {str(t): interpolator(t) for t in full_time_range}

            # Store processed data
            processed_data[feature_key] = interpolated_values

        # Save results
        output_path = os.path.join(temp_maps_dir, reference_name)
        make_path(output_path)
        np.savez(os.path.join(output_path, "interpolated_histograms.npz"), **processed_data)

        # Update metadata
        res_MetaData = {
            "interpolated_histograms": os.path.join(output_path, "interpolated_histograms.npz"),
        }
        write_JSON(output_path, "interpolated_histograms", res_MetaData)

        logger.info(f"Saved interpolated histograms for reference {reference_name}.")


def do_all(surface_dir, surfaces2d_dir, category_keys, output_dir,surface_file_key):
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
    do_HistPointData(surface_dir, surfaces2d_dir, category_keys, output_dir,surface_file_key)
    do_temporalHistInterpolation(output_dir)

    logger.info("Full pipeline processing completed.")
