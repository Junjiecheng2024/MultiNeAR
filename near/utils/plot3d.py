"""3D visualization utilities for medical images."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from skimage import measure
from plotly.offline import init_notebook_mode, plot
from plotly.figure_factory import create_trisurf
from skimage.measure import find_contours

init_notebook_mode(connected=True)


def plot_volume_as_panel(volume, aux1=None, aux2=None, aux3=None, title=None, 
                        height=8, width=None, cmap=None, cmap_aux=None, save_path=None):
    """Plot 2D slices of 3D volume in a grid panel.
    
    Args:
        volume: 3D volume array (D, H, W)
        aux1, aux2, aux3: Optional overlay masks for visualization
        title: Panel title
        height: Number of rows in grid
        width: Number of columns (auto-calculated if None)
        cmap: Colormap for main volume
        cmap_aux: Colormap for auxiliary masks
        save_path: Path to save figure (without extension)
    """
    if aux1 is not None:
        assert aux1.shape == volume.shape
    n_slices = volume.shape[0]

    if width is None:
        width = int(np.ceil(n_slices / height))

    fig, axes = plt.subplots(height, width, figsize=(width*3, height*3), dpi=500)
    fig.patch.set_facecolor('white')

    if title is not None:
        fig.suptitle(title, fontsize=30)

    i = 0
    for ax_row in axes:
        for ax in ax_row:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n_slices:
                continue
            ax.imshow(volume[i], cmap=cmap)
            
            if aux1 is not None:
                edges = find_edges(aux1[i])
                if edges is not None:
                    xs, ys = edges
                    ax.plot(xs, ys, color='red', alpha=0.5, linewidth=2)
            if aux2 is not None:
                edges = find_edges(aux2[i])
                if edges is not None:
                    xs, ys = edges
                    ax.plot(xs, ys, color='blue', alpha=0.5, linewidth=2)
            if aux3 is not None:
                edges = find_edges(aux3[i])
                if edges is not None:
                    xs, ys = edges
                    ax.plot(xs, ys, color='yellow', alpha=0.5, linewidth=2)
            i += 1

    if save_path is not None:
        plt.savefig(save_path+'.png')

    plt.clf()
    plt.close()


def plot_voxels(voxels, aux=None):
    """Plot voxel slices sequentially."""
    if aux is not None:
        assert voxels.shape == aux.shape
    n = voxels.shape[0]
    for i in range(n):
        plt.figure(figsize=(4, 4))
        plt.title(f"Slice {i}")
        plt.imshow(voxels[i], cmap=plt.cm.gray)
        if aux is not None:
            plt.imshow(aux[i], alpha=0.3)
        plt.show()


def plot_hist(voxel):
    """Plot histogram of voxel intensities."""
    plt.hist(voxel.flatten(), bins=50, color='c')
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.show()


def plot_voxels_stack(stack, rows=6, cols=6, start=10, interval=5):
    """Plot multiple slices in a grid for quick browsing."""
    fig, ax = plt.subplots(rows, cols, figsize=[18, 18])
    for i in range(rows * cols):
        ind = start + i * interval
        ax[int(i / rows), int(i % rows)].set_title(f'Slice {ind}')
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    plt.show()


def find_edges(mask, level=0.5):
    """Extract contour edges from binary mask."""
    edges = find_contours(mask, level)
    if len(edges) == 0:
        return None
    edges = edges[0]
    ys = edges[:, 0]
    xs = edges[:, 1]
    return xs, ys


def plot_contours(arr, aux, level=0.5, ax=None, **kwargs):
    """Plot image with overlaid contours."""
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(arr, cmap=plt.cm.gray)
    edges = find_edges(aux, level)
    if edges is not None:
        xs, ys = edges
        ax.plot(xs, ys)


def plot_contours_slices(arr, aux, aux2=None, level=0.5, ax=None, cmap=None, 
                        color1='red', color2='blue', linewidth1=3, linewidth2=2.5, 
                        save_path=None, **kwargs):
    """Plot 2D slice with multiple contour overlays."""
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.imshow(arr, cmap=cmap, alpha=0.6)
    edges = find_edges(aux, level)
    if edges is not None:
        xs, ys = edges
        ax.plot(xs, ys, color=color1, alpha=1, linewidth=linewidth1)

    if aux2 is not None:
        edges = find_edges(aux2, level)
        if edges is not None:
            xs, ys = edges
            ax.plot(xs, ys, color=color2, alpha=0.8, linewidth=linewidth2)

    if save_path is not None:
        plt.savefig(save_path+'.png', transparent=True)

    plt.clf()
    plt.close()


def plot_voxel(voxel, title='voxel'):
    """Plot single voxel slice."""
    plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.imshow(voxel, cmap=plt.cm.gray)
    plt.show()


def plot_voxel_slice(voxels, slice_i=0, title="Slice "):
    """Plot specific slice from volume."""
    plot_voxel(voxels[slice_i], title=title + str(slice_i))


def animate_voxels(voxels, aux=None, interval=300):
    """Create animation of volume slices."""
    fig = plt.figure()
    layer1 = plt.imshow(voxels[0], cmap=plt.cm.gray, animated=True)
    if aux is not None:
        assert voxels.shape == aux.shape
        layer2 = plt.imshow(aux[0] * 1., alpha=0.3, animated=True)

    def animate(i):
        plt.title(f"Slice {i}")
        layer1.set_array(voxels[i])
        if aux is not None:
            layer2.set_array(aux[i] * 1.)

    ani = animation.FuncAnimation(fig, animate, range(1, voxels.shape[0]),
                                 interval=interval, blit=True)
    return ani


def make_mesh(image, threshold, step_size):
    """Generate 3D mesh using marching cubes algorithm.
    
    Args:
        image: 3D volume array
        threshold: Isosurface threshold value
        step_size: Sampling step size (larger = faster but coarser)
    
    Returns:
        verts: Vertex coordinates
        faces: Triangle face indices
    """
    p = image.transpose(2, 1, 0)
    verts, faces, norm, val = measure.marching_cubes(p, threshold,
                                                     step_size=step_size,
                                                     allow_degenerate=True)
    return verts, faces


def hidden_axis(ax, r=None):
    """Hide Plotly axis elements."""
    ax.showgrid = False
    ax.zeroline = False
    ax.showline = False
    ax.ticks = ''
    ax.showticklabels = False
    ax.range = r
    ax.title = ""


def plotly_3d_to_html(verts, faces, filename="tmp.html", title="",
                     zyx_range=[[0, 128],[0, 128],[0, 128]],
                     camera=dict(eye=dict(x=0., y=2.5, z=0.)),
                     colormap=['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)'],
                     save_figure=True):
    """Render 3D mesh to interactive HTML using Plotly."""
    x, y, z = zip(*verts)

    fig = create_trisurf(x=x, y=y, z=z, showbackground=False, plot_edges=True,
                        colormap=colormap, simplices=faces, title=title,
                        show_colorbar=False)
    
    if zyx_range is not None:
        hidden_axis(fig.layout.scene.zaxis, zyx_range[0])
        hidden_axis(fig.layout.scene.yaxis, zyx_range[1])
        hidden_axis(fig.layout.scene.xaxis, zyx_range[2])

    if camera is not None:
        fig.update_layout(scene_camera=camera)

    plot(fig, filename=filename)

    if save_figure:
        fig.write_image(filename+".png")
    return fig


def plotly_3d_scan_to_html(scan, filename, threshold=0.5, step_size=1, title="",
                          zyx_range=None, pad_width=1,
                          camera=dict(eye=dict(x=0., y=2.5, z=0.)),
                          colormap=['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']):
    """Convert volume scan directly to 3D HTML visualization."""
    if pad_width is not None:
        scan = np.pad(scan, pad_width=pad_width)
    v, f = make_mesh(scan, threshold=threshold, step_size=step_size)
    return plotly_3d_to_html(v, f, filename, title, zyx_range, camera, colormap)


def plotly_3d(verts, faces, filename='tmp',
             zyx_range=[[0, 128],[0, 128],[0, 128]],
             camera=dict(eye=dict(x=0., y=2.5, z=0.)),
             colormap=['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)'],
             save_figure=True):
    """Render 3D mesh to static image file."""
    x, y, z = zip(*verts)
    fig = create_trisurf(x=x, y=y, z=z, title=None, showbackground=False,
                        plot_edges=True, colormap=colormap, simplices=faces,
                        show_colorbar=False)
    
    fig.layout.scene.zaxis.showticklabels = False
    fig.layout.scene.yaxis.showticklabels = False
    fig.layout.scene.xaxis.showticklabels = False
    fig.layout.scene.xaxis.title = ''
    fig.layout.scene.yaxis.title = ''
    fig.layout.scene.zaxis.title = ''
    
    if camera is not None:
        fig.update_layout(scene_camera=camera)

    if save_figure:
        fig.write_image(filename+".png")


def plotly_3d_scan(scan, filename='tmp', threshold=0.5, step_size=1,
                  zyx_range=None, pad_width=1,
                  camera=dict(eye=dict(x=0., y=2.5, z=0.)),
                  colormap=['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']):
    """Convert volume scan to 3D mesh and save as static image."""
    if pad_width is not None:
        scan = np.pad(scan, pad_width=pad_width)
    v, f = make_mesh(scan, threshold=threshold, step_size=step_size)
    return plotly_3d(v, f, filename, zyx_range, camera, colormap)
