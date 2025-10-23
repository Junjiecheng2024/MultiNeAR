import numpy as np  # 数值计算库，用于矩阵/数组运算
import matplotlib.pyplot as plt  # 可视化库 matplotlib 的绘图接口
from matplotlib import animation  # matplotlib 动画模块
from skimage import measure  # skimage 的图像测量模块（包含 marching_cubes）
from plotly.offline import init_notebook_mode, plot  # plotly 离线绘图接口
from plotly.figure_factory import create_trisurf  # 创建三角网格三维曲面图
from skimage.measure import find_contours  # 从二维掩膜中提取等值线（轮廓）

init_notebook_mode(connected=True)  # 初始化 Plotly 离线模式，确保在 Jupyter 中显示

# ============================================================
# 二维切片绘图函数
# ============================================================

def plot_volume_as_panel(volume, aux1=None, aux2=None, aux3=None, title=None, height=8, width=None, cmap=None, cmap_aux=None, save_path=None):
    # 绘制体积数据的二维切片拼图（如 MRI/CT 体素）
    if aux1 is not None:
        assert aux1.shape == volume.shape  # 如果有辅助掩膜，确保尺寸匹配
    n_slices = volume.shape[0]  # 切片数量（假定第一维为切片数）

    if width is None:
        width = int(np.ceil(n_slices / height))  # 自动计算列数以容纳所有切片

    fig, axes = plt.subplots(height, width, figsize=(width*3, height*3), dpi=500)  # 创建子图网格
    fig.patch.set_facecolor('white')  # 设置背景为白色

    if title is not None:
        fig.suptitle(title, fontsize=30)  # 添加整体标题

    i = 0
    for ax_row in axes:  # 遍历每一行
        for ax in ax_row:  # 遍历每个子图
            ax.get_xaxis().set_visible(False)  # 隐藏坐标轴
            ax.get_yaxis().set_visible(False)
            if i == n_slices:
                continue  # 如果切片用完则跳过
            ax.imshow(volume[i], cmap=cmap)  # 显示灰度图
            # 以下三个辅助掩膜（如分割轮廓）分别叠加在图上
            if aux1 is not None:
                edges = find_edges(aux1[i])  # 提取边缘线
                if edges is not None:
                    xs, ys = edges
                    ax.plot(xs, ys, color='red', alpha=0.5, linewidth=2)  # 红色边缘
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
        plt.savefig(save_path+'.png')  # 保存为图片文件

    plt.clf()
    plt.close()  # 清理内存防止过多打开窗口


def plot_voxels(voxels, aux=None):
    """ 绘制体素堆栈（逐层显示）"""
    if aux is not None:
        assert voxels.shape == aux.shape  # 辅助数据必须同形状
    n = voxels.shape[0]
    for i in range(n):
        plt.figure(figsize=(4, 4))
        plt.title("@%s" % i)
        plt.imshow(voxels[i], cmap=plt.cm.gray)
        if aux is not None:
            plt.imshow(aux[i], alpha=0.3)  # 半透明叠加
        plt.show()


def plot_hist(voxel):
    """ 绘制体素像素值直方图 """
    plt.hist(voxel.flatten(), bins=50, color='c')
    plt.xlabel("pixel value")
    plt.ylabel("frequency")
    plt.show()


def plot_voxels_stack(stack, rows=6, cols=6, start=10, interval=5):
    """ 批量绘制切片序列（适合快速浏览3D扫描）"""
    fig, ax = plt.subplots(rows, cols, figsize=[18, 18])
    for i in range(rows * cols):
        ind = start + i * interval  # 计算当前切片索引
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    plt.show()


def find_edges(mask, level=0.5):
    """ 从二值掩膜中提取边缘轮廓 """
    edges = find_contours(mask, level)
    if len(edges) == 0:
        return None
    edges = edges[0]
    ys = edges[:, 0]
    xs = edges[:, 1]
    return xs, ys


def plot_contours(arr, aux, level=0.5, ax=None, **kwargs):
    """ 在图像上叠加单个轮廓 """
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(arr, cmap=plt.cm.gray)
    edges = find_edges(aux, level)
    if edges is not None:
        xs, ys = edges
        ax.plot(xs, ys)


def plot_contours_slices(arr, aux, aux2=None, level=0.5, ax=None, cmap=None, color1='red', color2='blue', linewidth1=3, linewidth2=2.5, save_path=None, **kwargs):
    """ 在二维切片上绘制多个轮廓叠加（可保存透明背景图）"""
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.imshow(arr, cmap=cmap, alpha=0.6)  # 主图像半透明显示
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
        plt.savefig(save_path+'.png', transparent=True)  # 保存为透明 PNG

    plt.clf()
    plt.close()


def plot_voxel(voxel, title='voxel'):
    """ 绘制单张体素灰度图 """
    plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.imshow(voxel, cmap=plt.cm.gray)
    plt.show()


def plot_voxel_slice(voxels, slice_i=0, title="@"):
    """ 快捷绘制第 slice_i 层切片 """
    plot_voxel(voxels[slice_i], title=title + str(slice_i))


def animate_voxels(voxels, aux=None, interval=300):
    """ 生成体素切片动画（可视化3D体积）"""
    fig = plt.figure()
    layer1 = plt.imshow(voxels[0], cmap=plt.cm.gray, animated=True)
    if aux is not None:
        assert voxels.shape == aux.shape
        layer2 = plt.imshow(aux[0] * 1., alpha=0.3, animated=True)

    def animate(i):
        plt.title("@%s" % i)
        layer1.set_array(voxels[i])
        if aux is not None:
            layer2.set_array(aux[i] * 1.)

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  range(1, voxels.shape[0]),
                                  interval=interval,
                                  blit=True)
    return ani  # 返回动画对象，可在 notebook 中显示

# ============================================================
# 三维 Plotly 网格可视化部分
# ============================================================

def make_mesh(image, threshold, step_size):
    """ 生成三维网格，用于3D显示
    threshold：等值面阈值
    step_size：采样步长（越大越快但越粗）
    """
    p = image.transpose(2, 1, 0)  # 调整坐标轴顺序（z,y,x）

    verts, faces, norm, val = measure.marching_cubes(p,
                                                     threshold,
                                                     step_size=step_size,
                                                     allow_degenerate=True)
    return verts, faces  # 返回顶点坐标和三角面索引


def hidden_axis(ax, r=None):
    """ 隐藏 Plotly 坐标轴 """
    ax.showgrid = False
    ax.zeroline = False
    ax.showline = False
    ax.ticks = ''
    ax.showticklabels = False
    ax.range = r
    ax.title = ""


def plotly_3d_to_html(verts,
                      faces,
                      filename="tmp.html",
                      title="",
                      zyx_range=[[0, 128],[0, 128],[0, 128]],
                      camera=dict(
                          eye=dict(x=0., y=2.5, z=0.)
                      ),
                      colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)'],
                      save_figure=True):
    """ 使用 Plotly 离线绘制 3D 网格并保存为 HTML """
    x, y, z = zip(*verts)  # 拆分顶点坐标

    fig = create_trisurf(
        x=x, y=y, z=z,
        showbackground=False,
        plot_edges=True,
        colormap=colormap,
        simplices=faces,
        title=title,
        show_colorbar=False)
    if zyx_range is not None:
        hidden_axis(fig.layout.scene.zaxis, zyx_range[0])
        hidden_axis(fig.layout.scene.yaxis, zyx_range[1])
        hidden_axis(fig.layout.scene.xaxis, zyx_range[2])

    if camera is not None:
        fig.update_layout(scene_camera=camera)

    plot(fig, filename=filename)  # 生成交互式 HTML 文件

    if save_figure:
        fig.write_image(filename+".png")  # 保存静态图像
    return fig


def plotly_3d_scan_to_html(scan,
                           filename,
                           threshold=0.5,
                           step_size=1,
                           title="",
                           zyx_range=None,
                           pad_width=1,
                           camera=dict(
                               eye=dict(x=0., y=2.5, z=0.)
                           ),
                           colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']):
    """ 将体积数据直接转换为 3D 网格并保存为 HTML """
    if pad_width is not None:
        scan = np.pad(scan, pad_width=pad_width)  # 边缘补零避免边界缺失
    v, f = make_mesh(scan, threshold=threshold, step_size=step_size)
    return plotly_3d_to_html(v, f, filename, title, zyx_range, camera, colormap)


def plotly_3d(verts,
              faces,
              filename='tmp',
              zyx_range=[[0, 128],[0, 128],[0, 128]],
              camera=dict(eye=dict(x=0., y=2.5, z=0.)),
              colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)'],
              save_figure=True):
    """ 绘制 3D 网格并保存静态图片（不生成 HTML）"""
    x, y, z = zip(*verts)
    fig = create_trisurf(
        x=x, y=y, z=z,
        title=None,
        showbackground=False,
        plot_edges=True,
        colormap=colormap,
        simplices=faces,
        show_colorbar=False)
    
    # 隐藏坐标轴与标题
    fig.layout.scene.zaxis.showticklabels=False
    fig.layout.scene.yaxis.showticklabels=False
    fig.layout.scene.xaxis.showticklabels=False
    fig.layout.scene.xaxis.title=''
    fig.layout.scene.yaxis.title=''
    fig.layout.scene.zaxis.title=''
    
    if camera is not None:
        fig.update_layout(scene_camera=camera)

    if save_figure:
        fig.write_image(filename+".png")


def plotly_3d_scan(scan,
                   filename='tmp',
                   threshold=0.5,
                   step_size=1,
                   zyx_range=None,
                   pad_width=1,
                   camera=dict(eye=dict(x=0., y=2.5, z=0.)),
                   colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']):
    """ 将体素扫描转换为 3D 网格并保存静态图片 """
    if pad_width is not None:
        scan = np.pad(scan, pad_width=pad_width)
    v, f = make_mesh(scan, threshold=threshold, step_size=step_size)
    return plotly_3d(v, f, filename, zyx_range, camera, colormap)
