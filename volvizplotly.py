import numpy as np
import plotly.graph_objects as go


# MAIN FUNCTIONS

def volume_slicer(vol, slices, cmin=None, cmax=None, colorscale='Gray',
                fig=None, show=True, title = '', width=600, height=600):
    ''' Visualizes chosen slices from volume.
        
        vol: a 3D numpy array. 
        slices: list of 3 lists containing slice indices in three directions.
           Alternatively, each of the three elements may be one of the strings
           'mid', 'ends', 'first', 'last' or None (None equals empty).
        min, cmax and colorscale: values passed to plotly colormapping.
        fig: plotly figure, if None new figure will be created.
        show: whether to show figure, if not figure will be returned.
        title, width, height: values passed to ploty layout.

    '''
  
    if cmin is None:
        cmin = vol.min()
    if cmax is None:
        cmax = vol.max()
      
    dim = vol.shape

    for i in range(len(slices)):
        sl = slices[i]
        if type(sl) is not list:
            if sl=='mid':
                slices[i] = [dim[i]//2]
            elif sl=='ends':
                slices[i] = [0, dim[i]-1]
            elif sl=='first':
                slices[i] = [0]
            elif sl=='last':
                slices[i] = [dim[i]-1]
            elif type(sl) is int:
                slices[i] = [sl]
            else: # inclusiv None
                slices[i] = []

    surfs = []
    for i in range(3): # three directions
        gi = grids(dim,i)
        for j in range(len(slices[i])):
            g = gi.copy()
            s = slices[i][j]
            g[i] *= s
            surf = dict(x=g[2], y=g[1], z=g[0], surfacecolor=volslice(vol, i, s))
            surfs.append(surf)

    common = dict(colorscale=colorscale, cmin=cmin, cmax=cmax)
    surfaces = [go.Surface({**s, **common}) for s in surfs]

    # Set limits and aspect ratio.
    d = max(dim)
    scene = dict(xaxis = dict(range=[-1, dim[2]], autorange=False),
            yaxis = dict(range=[-1, dim[1]], autorange=False),
            zaxis = dict(range=[-1, dim[0]], autorange=False), 
            aspectratio = dict(x=dim[2]/d, y=dim[1]/d, z=dim[0]/d))
    layout = dict(title=title, width=width, height=height, scene=scene)

    if fig is None:
        fig = go.Figure()
    
    fig.add_traces(surfaces)
    fig.update_layout(layout)  
    
    if show:
        fig.show()
        return
    else:
        return fig


def show_mesh(vertices, faces=None, wireframe=False, surf=True,
            fig=None, show=True, title = '', width=600, height=600):
    ''' Show triangle surface mesh in 3d.
        
        vertices and faces: mesh entities as n x 3 numpy arrays
        wireframe and surf: flags for showing wireframe, surface, or both. If 
            neither wireframe or surf are chosen, pointcloud is shown. 
        fig: plotly figure, if None new figure will be created.
        show: whether to show figure, if not figure will be returned.
        title, width, height: values passed to ploty layout.
    '''

    if fig is None:
        fig = go.Figure()
  
    if surf and faces: 
        fig.add_trace(mesh_surface_plot(vertices, faces))
  
    if wireframe and faces:        
        fig.add_trace(mesh_wireframe_plot(vertices, faces))
  
    if ((not surf) and (not wireframe)) or (not faces):
        fig.add_trace(pointcloud_plot(vertices))
        fig.update_traces(marker_size = 1)
    
    fig.update_layout(title_text=title, height=height, width=width)
    
    if show:
        fig.show()
        return
    else:
        return fig

# Helping functions

def grids(dim, i):
    ''' Returns matrices with coordinates for slicing along axis i. '''
    two = dim[:i] + dim[i+1:]
    out = np.mgrid[0:two[0], 0:two[1]]
    out = np.insert(out, i, np.ones(two), axis=0)
    return out


def volslice(vol, i, s):
    ''' Returns volume slice s along axis i.'''
    s_xyz = (slice(None),) * i + (slice(s,s+1),)
    return vol[s_xyz].squeeze(axis=i)

def pointcloud_plot(points):

    gm = go.Scatter3d(z=points[:,0], y=points[:,1], x=points[:,2], 
                      mode='markers', name='')
    return gm 


def mesh_surface_plot(vertices, faces):

    gm = go.Mesh3d(z=vertices[:,0], y=vertices[:,1], x=vertices[:,2], 
            i=faces[:,0], j=faces[:,1], k=faces[:,2])
    return gm


def mesh_wireframe_plot(vertices, faces):
    
    Xe = np.concatenate((vertices[faces, 0], np.full((faces.shape[0],1), None)),
                        axis=1).ravel()
    Ye = np.concatenate((vertices[faces, 1], np.full((faces.shape[0],1), None)),
                        axis=1).ravel()
    Ze = np.concatenate((vertices[faces, 2], np.full((faces.shape[0],1), None)),
                        axis=1).ravel()
    
    gm = go.Scatter3d(z=Xe, y=Ye, x=Ze, mode='lines', name='',
            line=dict(color= 'rgb(40,40,40)', width=1))  
    
    return gm

   
# EXPERIMENTAL FUCNTIONS

def interactive_volume_slicer(vol, cmin=None, cmax=None, colorscale='Gray',
                  title = '', width=600, height=600):
    '''
    This function is greatly inspired by (basicaly copied from)  
    https://plotly.com/python/visualizing-mri-volume-slices/.
    TODO: input to choose slicing direction
    '''
    if cmin is None:
        cmin = vol.min()
    if cmax is None:
        cmax = vol.max()
    Z, Y, X = vol.shape

    o = np.ones((Y, X))
    surfaces = [go.Surface(
        z = z * o,
        surfacecolor = vol[z],
        colorscale = colorscale,
        cmin=cmin, cmax=cmax)
        for z in range(Z)]

    # you need to name the frame for the animation to behave properly
    frames = [go.Frame(data=surface, name=str(z)) 
        for z, surface in enumerate(surfaces)]

    # Define frames for animation
    fig = go.Figure(frames = frames)

    # Add surface to be displayed before animation starts
    fig.add_trace(surfaces[0])

    frame_args = {
                "frame": {"duration": 0},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": 0, "easing": "linear"},
            }

    # Set slicer placement and settings.
    slider_dict = {
                    "pad": {"t": 100},
                    "len": 0.8,
                    "x": 0.2,
                    "y": 0,
                    "xanchor": "left",
                    "yanchor": "middle",
                    "steps": [
                        {
                            "args": [[f.name], frame_args],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in fig.frames
                    ],
                }

    # Set button placement and settings.
    buttons_dict = {
                    "pad": {"t": 100},
                    "yanchor": "middle",
                    "xanchor": "right",
                    "x": 0.1,
                    "y": 0,
                    "type": "buttons",
                    "direction": "left", # make buttons left-rigth
                    "buttons": [
                        {
                            "args": [None, frame_args],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {   # note [] around None, this makes it a Pause button!!!
                            "args": [[None], frame_args],  
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                }

    d = max(X, Y, Z)
    scene_dict = {
        "xaxis": dict(range=[-1, X], autorange=False),
        "yaxis": dict(range=[-1, Y], autorange=False),    
        "zaxis": dict(range=[-1, Z], autorange=False),
        "aspectratio": dict(x=X/d, y=Y/d, z=Z/d),
    }

    # Layout
    fig.update_layout(
            title=title,
            width=width, height=height,
            scene=scene_dict,
            updatemenus = [buttons_dict],
            sliders = [slider_dict]
    )

    fig.show()
