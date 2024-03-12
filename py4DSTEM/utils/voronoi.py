import numpy as np
from scipy.spatial import Voronoi


def get_voronoi_vertices(voronoi, nx, ny, dist=10):
    """
    From a scipy.spatial.Voronoi instance, return a list of ndarrays, where each array
    is shape (N,2) and contains the (x,y) positions of the vertices of a voronoi region.

    The problem this function solves is that in a Voronoi instance, some vertices outside
    the field of view of the tesselated region are left unspecified; only the existence
    of a point beyond the field is referenced (which may or may not be 'at infinity').
    This function specifies all points, such that the vertices and edges of the
    tesselation may be directly laid over data.

    Args:
        voronoi (scipy.spatial.Voronoi): the voronoi tesselation
        nx (int): the x field-of-view of the tesselated region
        ny (int): the y field-of-view of the tesselated region
        dist (float, optional): place new vertices by extending new voronoi edges outside
            the frame by a distance of this factor times the distance of its known vertex
            from the frame edge

    Returns:
        (list of ndarrays of shape (N,2)): the (x,y) coords of the vertices of each
        voronoi region
    """
    assert isinstance(
        voronoi, Voronoi
    ), "voronoi must be a scipy.spatial.Voronoi instance"

    vertex_list = []

    # Get info about ridges containing an unknown vertex.  Include:
    #   -the index of its known vertex, in voronoi.vertices, and
    #   -the indices of its regions, in voronoi.point_region
    edgeridge_vertices_and_points = []
    for i in range(len(voronoi.ridge_vertices)):
        ridge = voronoi.ridge_vertices[i]
        if -1 in ridge:
            edgeridge_vertices_and_points.append(
                [max(ridge), voronoi.ridge_points[i, 0], voronoi.ridge_points[i, 1]]
            )
    edgeridge_vertices_and_points = np.array(edgeridge_vertices_and_points)

    # Loop over all regions
    for index in range(len(voronoi.regions)):
        # Get the vertex indices
        vertex_indices = voronoi.regions[index]
        vertices = np.array([0, 0])
        # Loop over all vertices
        for i in range(len(vertex_indices)):
            index_current = vertex_indices[i]
            if index_current != -1:
                # For known vertices, just add to a running list
                vertices = np.vstack((vertices, voronoi.vertices[index_current]))
            else:
                # For unknown vertices, get the first vertex it connects to,
                # and the two voronoi points that this ridge divides
                index_prev = vertex_indices[(i - 1) % len(vertex_indices)]
                edgeridge_index = int(
                    np.argwhere(edgeridge_vertices_and_points[:, 0] == index_prev)
                )
                index_vert, region0, region1 = edgeridge_vertices_and_points[
                    edgeridge_index, :
                ]
                x, y = voronoi.vertices[index_vert]
                # Only add new points for unknown vertices if the known index it connects to
                # is inside the frame.  Add points by finding the line segment starting at
                # the known point which is perpendicular to the segment connecting the two
                # voronoi points, and extending that line segment outside the frame.
                if (x > 0) and (x < nx) and (y > 0) and (y < ny):
                    x_r0, y_r0 = voronoi.points[region0]
                    x_r1, y_r1 = voronoi.points[region1]
                    m = -(x_r1 - x_r0) / (y_r1 - y_r0)
                    # Choose the direction to extend the ridge
                    ts = np.array([-x, -y / m, nx - x, (ny - y) / m])
                    x_t = lambda t: x + t
                    y_t = lambda t: y + m * t
                    t = ts[np.argmin(np.hypot(x - x_t(ts), y - y_t(ts)))]
                    x_new, y_new = x_t(dist * t), y_t(dist * t)
                    vertices = np.vstack((vertices, np.array([x_new, y_new])))
                else:
                    # If handling unknown points connecting to points outside the frame is
                    # desired, add here
                    pass

                # Repeat for the second vertec the unknown vertex connects to
                index_next = vertex_indices[(i + 1) % len(vertex_indices)]
                edgeridge_index = int(
                    np.argwhere(edgeridge_vertices_and_points[:, 0] == index_next)
                )
                index_vert, region0, region1 = edgeridge_vertices_and_points[
                    edgeridge_index, :
                ]
                x, y = voronoi.vertices[index_vert]
                if (x > 0) and (x < nx) and (y > 0) and (y < ny):
                    x_r0, y_r0 = voronoi.points[region0]
                    x_r1, y_r1 = voronoi.points[region1]
                    m = -(x_r1 - x_r0) / (y_r1 - y_r0)
                    # Choose the direction to extend the ridge
                    ts = np.array([-x, -y / m, nx - x, (ny - y) / m])
                    x_t = lambda t: x + t
                    y_t = lambda t: y + m * t
                    t = ts[np.argmin(np.hypot(x - x_t(ts), y - y_t(ts)))]
                    x_new, y_new = x_t(dist * t), y_t(dist * t)
                    vertices = np.vstack((vertices, np.array([x_new, y_new])))
                else:
                    pass

        # Remove regions with insufficiently many vertices
        if len(vertices) < 4:
            vertices = np.array([])
        # Remove initial dummy point
        else:
            vertices = vertices[1:, :]
        # Update vertex list with this region's vertices
        vertex_list.append(vertices)

    return vertex_list


