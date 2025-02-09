from zf_pf_diffeo.reference_geometries import create_refGeometry


def do_referenceGeometries(surface_dir,category_keys,output_dir):
    """
    Takes all surfaces in 2d and calculates for the groups given by keys the means/representing surfaces
    """

    meshes3d=None #TODO fill with actual surfaces
    
    avg_coeff,avgPolygon = create_refGeometry(meshes3d,n_harmonics=10,n_coords=40)
    #safe resulting 
    return

def do_temporalreferenceGeometries(surfaces2d_dir,time_key,category_keys):
    """
    interpolated temporal and findes moving meshes
    """

def do_HistPointData(surface_dir,surfaces2d_dir,category_keys,output_dir):
    """
    Takes PointData in 3d Space, projetc to surface in 3d, transports further and bins for surface2d
    """




#TODO geometry data

#TODO image data



def do_temporalInterpolation(idk):
    
    return