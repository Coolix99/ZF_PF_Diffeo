import pytest
import numpy as np
import pyvista as pv
from zf_pf_diffeo.reference_geometries import (
    getPolygon, centroid_Polygon, shift_coeff, create_2dGeometry, generate_2d_mesh
)

@pytest.fixture
def sample_mesh():
    """Creates a simple PyVista triangular mesh for testing."""
    points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Square corners
        [0.5, 0.5, 0], [1.5, 0.5, 0], [0.5, 1.5, 0]  # Additional points
    ])
    
    faces = np.hstack([
        [3, 0, 1, 4],  # Triangle (0,1,4)
        [3, 1, 2, 4],  # Triangle (1,2,4)
        [3, 2, 3, 4],  # Triangle (2,3,4)
        [3, 3, 0, 4],  # Triangle (3,0,4)
        [3, 1, 5, 4],  # Triangle (1,5,4)
        [3, 3, 6, 4]   # Triangle (3,6,4)
    ]).astype(np.int32)  # Ensure correct dtype for PyVista
    
    mesh = pv.PolyData(points, faces)
    
    # Mock coordinate data (2D projection)
    mesh.point_data["coord_1"] = points[:, 0]
    mesh.point_data["coord_2"] = points[:, 1]

    return mesh

def test_centroid_polygon():
    """Test centroid computation for a simple square."""
    x = np.array([0, 1, 1, 0])
    y = np.array([0, 0, 1, 1])
    centroid = centroid_Polygon(x, y)

    assert np.allclose(centroid, [0.5, 0.5]), "Centroid of square should be at (0.5, 0.5)"

# def test_getPolygon(sample_mesh):
#     """Test extraction of 2D boundary from a mesh."""
#     print(sample_mesh)
#     polygon = getPolygon(sample_mesh)

#     assert polygon.shape[0] == 2, "Polygon should have shape (2, N)"
#     assert polygon.shape[1] == 4, "Polygon should have 4 boundary points"
#     assert np.allclose(polygon[:, 0], [0, 0]), "First boundary point should match mesh"

def test_shift_coeff():
    """Test shifting of Fourier coefficients."""
    coeff = np.array([
        [1, 0, 0, 1],
        [0.5, -0.5, 0.5, -0.5]
    ])
    theta = np.pi / 2
    shifted = shift_coeff(coeff, theta)

    assert shifted.shape == coeff.shape, "Shifted coefficients should have the same shape"
    assert not np.allclose(shifted, coeff), "Shifted coefficients should be different"

# def test_create_2dGeometry(sample_mesh):
#     """Test creation of 2D Fourier coefficients."""
#     coeff = create_2dGeometry(sample_mesh, n_harmonics=5)

#     assert coeff.shape[1] == 4, "Each harmonic should have 4 coefficients"
#     assert coeff.shape[0] == 5, "Number of harmonics should match input"

def test_generate_2d_mesh():
    """Test mesh generation from boundary points."""
    boundary_points = np.array(
        [[ 2.11090569e+02, -4.51678859e-02],
        [ 2.08263076e+02,  2.98093612e+01],
        [ 1.98360085e+02,  5.73051047e+01],
        [ 1.82101654e+02,  8.12125871e+01],
        [ 1.60940138e+02,  1.01745164e+02],
        [ 1.36424316e+02,  1.18277409e+02],
        [ 1.09719626e+02,  1.29047683e+02],
        [ 8.14265190e+01,  1.33683737e+02],
        [ 5.19645651e+01,  1.34261127e+02],
        [ 2.20808115e+01,  1.32835684e+02],
        [-7.43816970e+00,  1.29061510e+02],
        [-3.64131204e+01,  1.21510909e+02],
        [-6.51474452e+01,  1.11191521e+02],
        [-9.40559124e+01,  1.02513720e+02],
        [-1.23780401e+02,  9.91825325e+01],
        [-1.54244174e+02,  9.86289151e+01],
        [-1.82224921e+02,  9.24201620e+01],
        [-2.01661589e+02,  7.43606025e+01],
        [-2.09162856e+02,  4.70441019e+01],
        [-2.08303153e+02,  1.78500319e+01],
        [-2.05379365e+02, -1.00020323e+01],
        [-2.01374865e+02, -3.82051912e+01],
        [-1.91476673e+02, -6.59343496e+01],
        [-1.72500130e+02, -8.75245034e+01],
        [-1.47059219e+02,  -9.96157932e+01],
        [-1.19553828e+02, -1.06021994e+02],
        [-9.15521522e+01, -1.12562048e+02],
        [-6.27464788e+01, -1.20395779e+02],
        [-3.35854211e+01,  -1.27227531e+02],
        [-4.82394353e+00, -1.32027364e+02],
        [ 2.38037873e+01, -1.35242032e+02],
        [ 5.27462907e+01, -1.36230091e+02],
        [ 8.15314443e+01, -1.33462818e+02],
        [ 1.09436782e+02, -1.26380449e+02],
        [ 1.36096654e+02, -1.15200187e+02],
        [ 1.60675666e+02, -9.98016371e+01],
        [ 1.81501809e+02, -8.00760790e+01],
        [ 1.97165401e+02, -5.64695336e+01],
        [ 2.07154625e+02, -2.95172826e+01],
        [ 2.11090569e+02, -4.51678859e-02]]
    )
    nodes, triangles = generate_2d_mesh(boundary_points,mesh_size=100)

    assert nodes.shape[1] == 2, "Nodes should have 2D coordinates"
    assert triangles.shape[1] == 3, "Triangles should have 3 indices"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
