# Til Anders fra vand@dtu.dk

import numpy as np
import PIL.Image as Image
import plotly.graph_objects as go
import plotly.colors as pc

def orthovis(slices, width=1200, height=1200): 
    ''' Visualizes slices from volume, takes three rgb images.
    
    Args:
        dim: tuple of volume dimensions.
        slices: three-element list of dictionaries, keys are slice positions, 
            values are slices. Slices should have pixel values in range 0 to N.
        colorscale: list of lists, each sublist contains a pixel value and a 
            color. Pixel values should be in range 0 to N.
        Note that the number of colors (N colored labels and one more for 
            background) is inferred from the length of colorscale.
    ''' 
    # Infer size from some slices, assume that all slices have the appropriate size.
    dim = (next(iter(slices[1].values())).shape[0],  
           *next(iter(slices[0].values())).shape)    
    
    # Infer number of labels from slices (not used, assuming N<24)
    # N = max([v.max() for v in slices[0].values() for d in range(3)])

    # Adapting built-in colorscale from https://plotly.com/python/discrete-color/
    colorscale = ([[0, 'rgb(255, 255, 255)']] + 
        [[i + 1, c] for i, c in enumerate(pc.qualitative.Light24)])
    
    # Create surfaces.
    surfs = []
    for i in range(3):
        for pos, slice in slices[i].items():
            grid = np.mgrid[0:slice.shape[0], 0:slice.shape[1]]
            grid = np.insert(grid, i, pos * np.ones(slice.shape[:2]), axis=0)
            surf = {'x': grid[2], 'y': grid[1], 'z': grid[0], 
                    'surfacecolor': slice}
            surfs.append(surf)

    N = len(colorscale)
    colorscale = [[val/(N - 1), color] for val, color in colorscale]
    common = {'colorscale': colorscale, 'cmin': 0, 'cmax': N, 'showscale': False}
    surfaces = [go.Surface({**s, **common}) for s in surfs]

    # Set limits and aspect ratio.
    d = max(dim)
    scene = {'xaxis': {'range': [-1, dim[2]], 'autorange': False},
            'yaxis': {'range': [-1, dim[1]], 'autorange': False},
            'zaxis': {'range': [-1, dim[0]], 'autorange': False}, 
            'aspectratio': {'x': dim[2]/d, 'y': dim[1]/d, 'z': dim[0]/d}}
    layout = {'title': '', 'width': width, 'height': height, 'scene': scene}

    fig = go.Figure()        
    fig.add_traces(surfaces)
    fig.update_layout(layout)  
    fig.show()
    

# Loading data as a three-element list of dicts, keys are positions, values are slices.
slices = [{s: np.array(Image.open(f'orhtovis_data/Mix 10_000_{d}_{s}_seg_index.png'))} 
          for d, s in [('x', 450), ('y', 325), ('z', 325)]]

orthovis(slices)

# %%
