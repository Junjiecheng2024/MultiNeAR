import collections  # 标准库：容器抽象与工具（如 Iterable 接口、计数器等）
from itertools import repeat  # 用于生成重复元素的迭代器
import numpy as np  # 数值计算库
import scipy  # 科学计算库（此处用到 ndimage.interpolation.zoom）


def crop_at_zyx_with_dhw(voxel, zyx, dhw, fill_with):
    '''Crop and pad on the fly.'''  # 说明：以中心坐标 zyx 和尺寸 dhw 进行裁剪；若越界则在外部用常数填充
    shape = voxel.shape  # 读取体素体的整体尺寸 (D, H, W)
    # z, y, x = zyx  # 可用于调试的拆包（这里留作注释）
    # d, h, w = dhw  # 同上
    crop_pos = []  # 存储每个轴上的裁剪 [low, high) 区间
    padding = [[0, 0], [0, 0], [0, 0]]  # 记录每个轴在低端/高端需要补的像素数
    for i, (center, length) in enumerate(zip(zyx, dhw)):  # 逐轴处理 (z,y,x) 与对应的 (d,h,w)
        assert length % 2 == 0  # 要求裁剪长度为偶数，便于对称围绕中心
        # assert center < shape[i] # it's not necessary for "moved center"  # 中心可超出原范围，允许移动后再补边
        low = round(center) - length // 2  # 计算下界：四舍五入中心后向左/下/前扩半长
        high = round(center) + length // 2  # 计算上界：向右/上/后扩半长
        if low < 0:  # 若下界越过起点
            padding[i][0] = int(0 - low)  # 记录需要在低端补多少像素
            low = 0  # 截断到 0
        if high > shape[i]:  # 若上界越过终点
            padding[i][1] = int(high - shape[i])  # 记录高端需要补的像素
            high = shape[i]  # 截断到最大边界
        crop_pos.append([int(low), int(high)])  # 记录该轴最终的裁剪区间
    cropped = voxel[crop_pos[0][0]:crop_pos[0][1], crop_pos[1]  # 根据三轴区间切片裁剪子体素
                    [0]:crop_pos[1][1], crop_pos[2][0]:crop_pos[2][1]]
    if np.sum(padding) > 0:  # 如果任一方向发生越界，需要补边
        cropped = np.lib.pad(cropped, padding, 'constant',  # 使用常数填充值进行外部填充
                             constant_values=fill_with)
    return cropped  # 返回裁剪且（可能）补边后的体素块


def window_clip(v, window_low=-1024, window_high=400, dtype=np.uint8):
    '''Use lung windown to map CT voxel to grey.'''  # 用 CT 的窗宽窗位把强度映射到 [0,255]
    # assert v.min() <= window_low  # 可选检查：最低值应小于等于下窗限
    return np.round(np.clip((v - window_low) / (window_high - window_low) * 255., 0, 255)).astype(dtype)
    # 先线性归一化到 [0,255]，再截断到区间内，最后四舍五入并转为目标 dtype（默认 uint8）


def spatial_normalize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''  # 根据原 spacing 重采样到目标 new_spacing
    resize_factor = []  # 存放每轴缩放因子
    for sp, nsp in zip(spacing, new_spacing):  # 遍历原/新体素间距
        resize_factor.append(float(sp) / nsp)  # 缩放因子 = 原间距 / 新间距
    resized = scipy.ndimage.interpolation.zoom(  # 体素级缩放（注意 interpolation.zoom 已逐步弃用）
        voxel.astype(float), resize_factor, mode='constant')  # 用常数模式处理边界
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):  # 根据实际结果回写精确 new_spacing
        new_spacing[i] = float(sp) * shape / rshape  # 新间距 = 原间距 * 原尺寸 / 新尺寸（避免累计误差）
    return resized, new_spacing  # 返回重采样体素与更新后的精确间距


def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''  # 通过欧拉角(离散 0/90/180/270 度)在三个平面依次旋转
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # 围绕 z 轴所在平面 (0,1) 旋转 angle[0] 次 90°
    Y = np.rot90(X, angle[1], axes=(0, 2))  # 围绕 y 轴所在平面 (0,2) 旋转 angle[1] 次 90°
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # 围绕 x 轴所在平面 (1,2) 旋转 angle[2] 次 90°
    return Z  # 返回旋转后的体素


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''  # 沿指定轴进行镜像翻转，-1 表示不翻转
    if axis != -1:
        ref = np.flip(array, axis)  # 沿 axis 翻转
    else:
        ref = np.copy(array)  # 不翻转则返回副本
    return ref  # 返回结果


def crop(array, zyx, dhw):
    z, y, x = zyx  # 拆解中心坐标 (z, y, x)
    d, h, w = dhw  # 拆解裁剪尺寸 (depth, height, width)
    cropped = array[z - d // 2:z + d // 2,  # 以中心为基准的对称裁剪（注意：不做越界检查）
                    y - h // 2:y + h // 2,
                    x - w // 2:x + w // 2]
    return cropped  # 返回裁剪区域（可能会因越界抛异常）


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)  # 在 [-move, move] 内随机偏移三个轴
    zyx = np.array(shape) // 2 + offset  # 在体素中心附近随机扰动得到新的中心
    return zyx  # 返回随机中心坐标


def get_uniform_assign(length, subset):
    assert subset > 0  # 子集数量必须大于 0
    per_length, remain = divmod(length, subset)  # 均分 length 到 subset 个子集，得到每份大小与剩余
    total_set = np.random.permutation(list(range(subset)) * per_length)  # 基本均匀分配后随机打乱
    remain_set = np.random.permutation(list(range(subset)))[:remain]  # 将余数再随机分配到若干子集
    return list(total_set) + list(remain_set)  # 合并得到长度为 length 的子集索引列表（0..subset-1）


def split_validation(df, subset, by):
    df = df.copy()  # 复制以免修改原 DataFrame
    for sset in df[by].unique():  # 以列 `by` 的唯一取值为组进行划分
        length = (df[by] == sset).sum()  # 该组的样本数
        df.loc[df[by] == sset, 'subset'] = get_uniform_assign(length, subset)  # 组内均匀随机分配子集编号
    df['subset'] = df['subset'].astype(int)  # 确保类型为 int
    return df  # 返回带有 'subset' 列的新 DataFrame


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):  # 若 x 本身是可迭代（如 tuple/list），直接返回
            return x
        return tuple(repeat(x, n))  # 否则将标量 x 重复 n 次，生成 n 元组

    return parse  # 返回一个“把输入扩展为 n 元组”的解析函数


_single = _ntuple(1)   # 生成 1 元组解析器（常用于把标量统一成可迭代）
_pair = _ntuple(2)     # 生成 2 元组解析器（如将 stride=2 扩展为 (2,2)）
_triple = _ntuple(3)   # 生成 3 元组解析器（如将尺寸 s 扩展为 (s,s,s)）
_quadruple = _ntuple(4)  # 生成 4 元组解析器
