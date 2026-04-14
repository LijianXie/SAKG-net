from osgeo import gdal
import numpy as np
import os


def check_input_stats(image_path, label_path):
    """
    查看输入影像和标签的大小、最大值、最小值、NoData 值和 NaN 值。

    参数：
    image_path (str): 多通道遥感影像路径
    label_path (str): 单通道标签数据路径
    """
    # 打开影像和标签文件
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    label_ds = gdal.Open(label_path, gdal.GA_ReadOnly)

    if image_ds is None or label_ds is None:
        raise Exception("无法打开输入文件")

    # 获取影像信息
    img_cols = image_ds.RasterXSize
    img_rows = image_ds.RasterYSize
    img_bands = image_ds.RasterCount
    print(f"\n影像信息: 行={img_rows}, 列={img_cols}, 波段数={img_bands}")

    # 获取标签信息
    label_cols = label_ds.RasterXSize
    label_rows = label_ds.RasterYSize
    label_bands = label_ds.RasterCount
    print(f"标签信息: 行={label_rows}, 列={label_cols}, 波段数={label_bands}")

    # 检查影像波段统计
    for band_idx in range(1, img_bands + 1):
        band = image_ds.GetRasterBand(band_idx)
        nodata = band.GetNoDataValue() or "未定义"
        data = band.ReadAsArray().astype(float)

        total_pixels = data.size
        nodata_count = np.sum(data == nodata) if nodata != "未定义" else 0
        data = np.where(data == nodata, np.nan, data) if nodata != "未定义" else data

        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        nan_count = np.sum(np.isnan(data))

        print(f"影像波段 {band_idx}: min={min_val:.4f}, max={max_val:.4f}, "
              f"NoData值={nodata}, NoData数量={nodata_count}, NaN数量={nan_count}")

    # 检查标签统计
    label_band = label_ds.GetRasterBand(1)
    label_nodata = label_band.GetNoDataValue() or "未定义"
    label_data = label_band.ReadAsArray().astype(float)

    label_total_pixels = label_data.size
    label_nodata_count = np.sum(label_data == label_nodata) if label_nodata != "未定义" else 0
    label_data = np.where(label_data == label_nodata, np.nan, label_data) if label_nodata != "未定义" else label_data

    label_min = np.nanmin(label_data)
    label_max = np.nanmax(label_data)
    label_nan_count = np.sum(np.isnan(label_data))

    print(f"标签: min={label_min:.4f}, max={label_max:.4f}, "
          f"NoData值={label_nodata}, NoData数量={label_nodata_count}, NaN数量={label_nan_count}")

    # 关闭数据集
    image_ds = None
    label_ds = None


def crop_to_training_data(image_path, label_path, output_dir, crop_size=(256, 256), stride=None):
    """
    将多通道遥感影像和单通道标签数据裁剪为训练数据块。

    参数：
    image_path (str): 多通道遥感影像路径
    label_path (str): 单通道标签数据路径
    output_dir (str): 输出训练数据保存目录
    crop_size (tuple): 裁剪块的大小 (height, width)，默认 (256, 256)
    stride (tuple): 滑动窗口步幅 (height, width)，默认 None（无重叠）
    """
    # 检查输入数据统计
    # check_input_stats(image_path, label_path)

    # 设置默认步幅
    if stride is None:
        stride = crop_size

    # 打开影像和标签文件
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    label_ds = gdal.Open(label_path, gdal.GA_ReadOnly)

    if image_ds is None or label_ds is None:
        raise Exception("无法打开输入文件")

    # 获取影像信息
    img_cols = image_ds.RasterXSize
    img_rows = image_ds.RasterYSize
    img_bands = image_ds.RasterCount
    projection = image_ds.GetProjection()
    geotransform = image_ds.GetGeoTransform()

    # 检查标签数据尺寸是否匹配
    label_cols = label_ds.RasterXSize
    label_rows = label_ds.RasterYSize
    if img_cols != label_cols or img_rows != label_rows:
        raise ValueError("影像和标签数据的尺寸不匹配")

    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lab"), exist_ok=True)

    # 计算裁剪块的行列数
    crop_height, crop_width = crop_size
    stride_height, stride_width = stride
    num_rows = (img_rows - crop_height) // stride_height + 1
    num_cols = (img_cols - crop_width) // stride_width + 1

    print(f"\n裁剪信息:")
    print(f"裁剪块大小: {crop_height}x{crop_width}, 步幅: {stride_height}x{stride_width}")
    print(f"将生成 {num_rows * num_cols} 个训练数据块")

    # GDAL 驱动
    driver = gdal.GetDriverByName("GTiff")

    # 裁剪并保存训练数据
    for i in range(num_rows):
        for j in range(num_cols):
            xoff = j * stride_width
            yoff = i * stride_height

            win_xsize = min(crop_width, img_cols - xoff)
            win_ysize = min(crop_height, img_rows - yoff)
            if win_xsize <= 0 or win_ysize <= 0:
                continue

            img_data = np.array([image_ds.GetRasterBand(k + 1).ReadAsArray(
                xoff, yoff, win_xsize, win_ysize) for k in range(img_bands)])
            label_data = label_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, win_xsize, win_ysize)

            if img_data.shape[1:] != crop_size or label_data.shape != crop_size:
                continue

            output_img_path = os.path.join(output_dir, "src", f"{i}_{j}.tif")
            output_label_path = os.path.join(output_dir, "lab", f"{i}_{j}.tif")

            img_out_ds = driver.Create(output_img_path, crop_width, crop_height, img_bands, gdal.GDT_Float32)
            img_out_ds.SetProjection(projection)
            new_geotransform = list(geotransform)
            new_geotransform[0] += xoff * geotransform[1]
            new_geotransform[3] += yoff * geotransform[5]
            img_out_ds.SetGeoTransform(tuple(new_geotransform))

            for band in range(img_bands):
                img_out_ds.GetRasterBand(band + 1).WriteArray(img_data[band])

            label_out_ds = driver.Create(output_label_path, crop_width, crop_height, 1, gdal.GDT_Byte)
            label_out_ds.SetProjection(projection)
            label_out_ds.SetGeoTransform(tuple(new_geotransform))
            label_out_ds.GetRasterBand(1).WriteArray(label_data)

            img_out_ds = None
            label_out_ds = None

    image_ds = None
    label_ds = None

    print(f"裁剪完成，训练数据已保存至: {output_dir}")


# 示例用法
if __name__ == "__main__":
    image_path = r"G:\xx\xx.tif"  # 多通道影像
    label_path = r"E:\xx\xx.tif"  # 单通道标签
    output_dir = r"G:\xx\xx"  # 输出目录

    crop_size = (256, 256)
    stride = (128, 128)
    # 重叠率0.5

    crop_to_training_data(image_path, label_path, output_dir, crop_size, stride)