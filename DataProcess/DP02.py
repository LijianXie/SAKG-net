"""
NDWI 用于提取地表水体，NDMI 用于监测植被水分。
这里以NDWI为例进行示例计算，请结合实际需要进行修改
"""

from osgeo import gdal
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count


# 处理单个时相的函数
def process_phase(input_file):
    start_time = time.time()

    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    if dataset is None:
        raise Exception(f"无法打开输入文件: {input_file}")

    rows = dataset.RasterYSize
    cols = dataset.RasterXSize
    bands = dataset.RasterCount
    nodata_value = dataset.GetRasterBand(1).GetNoDataValue() or 0

    print(f"\n处理文件: {input_file} (行={rows}, 列={cols}, 波段数={bands})")

    # 一次性读取所有波段
    all_data = np.array([dataset.GetRasterBand(i).ReadAsArray().astype(float) for i in range(1, bands + 1)])
    all_data = np.where(all_data == nodata_value, np.nan, all_data)

    # 归一化10个波段
    band_data = []
    mins, maxs = np.nanmin(all_data, axis=(1, 2)), np.nanmax(all_data, axis=(1, 2))
    for i in range(bands):
        if np.isnan(mins[i]) or mins[i] == maxs[i]:
            normalized = np.zeros_like(all_data[i])
        else:
            normalized = (all_data[i] - mins[i]) / (maxs[i] - mins[i])
        band_data.append(np.nan_to_num(normalized, nan=0.0))

    # 计算 NDVI 和 NDWI
    band_green, band_red, band_nir = all_data[1], all_data[2], all_data[6]  # B03, B04, B08
    ndvi = np.where((band_nir + band_red) == 0, 0, (band_nir - band_red) / (band_nir + band_red))
    ndwi = np.where((band_green + band_nir) == 0, 0, (band_green - band_nir) / (band_green + band_nir))

    ndvi_min, ndvi_max = np.nanmin(ndvi), np.nanmax(ndvi)
    ndwi_min, ndwi_max = np.nanmin(ndwi), np.nanmax(ndwi)
    ndvi_normalized = np.where(ndvi_max == ndvi_min, 0, (ndvi - ndvi_min) / (ndvi_max - ndvi_min))
    ndwi_normalized = np.where(ndwi_max == ndwi_min, 0, (ndwi - ndwi_min) / (ndwi_max - ndwi_min))

    phase_data = band_data + [np.nan_to_num(ndvi_normalized, nan=0.0), np.nan_to_num(ndwi_normalized, nan=0.0)]

    print(f"处理时相 {input_file} 耗时: {time.time() - start_time:.2f} 秒")
    return phase_data, dataset.GetProjection(), dataset.GetGeoTransform(), rows, cols


# 精简统计信息函数
def print_image_stats(input_files, all_band_data):
    print("\n=== 影像统计信息 ===")
    for idx, input_file in enumerate(input_files, 1):
        print(f"时相 {idx} ({os.path.basename(input_file)}): 处理后波段数 = 12")
    print(f"\n总波段数: {len(all_band_data)} (预期: {len(input_files) * 12})")


if __name__ == '__main__':
    # 输入文件列表"......这里添加你的数据"
    input_files = [
        r"......................................... ",
        r"......................................... ",
        r"......................................... ",
        r"......................................... ",
        r"......................................... "

    ]

    output_file = r"G:\XXXX\Process_img\XX\XX.tif"

    # 开始计时
    total_start_time = time.time()

    # 检查输入输出路径
    start_time = time.time()
    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
    output_dir = os.path.dirname(output_file) or "."
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"输出目录不可写: {output_dir}")
    print(f"路径检查耗时: {time.time() - start_time:.2f} 秒")

    # 并行处理每个时相
    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_phase, input_files)

    all_band_data = []
    projection, geotransform, rows, cols = None, None, None, None
    for phase_data, proj, geo, r, c in results:
        all_band_data.extend(phase_data)
        if projection is None:
            projection, geotransform, rows, cols = proj, geo, r, c
    print(f"并行处理所有时相耗时: {time.time() - start_time:.2f} 秒")

    # 统计信息
    start_time = time.time()
    print_image_stats(input_files, all_band_data)
    print(f"统计信息耗时: {time.time() - start_time:.2f} 秒")

    # 创建输出影像
    start_time = time.time()
    total_bands = len(all_band_data)
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise Exception("无法加载 GTiff 驱动")

    out_dataset = driver.Create(output_file, cols, rows, total_bands, gdal.GDT_Float32)
    if out_dataset is None:
        raise Exception(f"创建输出文件失败: {output_file}")

    out_dataset.SetProjection(projection)
    out_dataset.SetGeoTransform(geotransform)

    for i, band_array in enumerate(all_band_data, 1):
        phase_idx = (i - 1) // 12 + 1
        band_idx = (i - 1) % 12 + 1
        description = (f"Phase_{phase_idx}_Band_{band_idx}_Normalized" if band_idx <= 10 else
                       f"Phase_{phase_idx}_NDVI_Normalized" if band_idx == 11 else
                       f"Phase_{phase_idx}_NDWI_Normalized")
        out_dataset.GetRasterBand(i).WriteArray(band_array)
        out_dataset.GetRasterBand(i).SetDescription(description)

    out_dataset = None
    print(f"创建和写入输出影像耗时: {time.time() - start_time:.2f} 秒")

    # 总耗时
    total_time = time.time() - total_start_time
    print(f"\n总处理耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
    print("处理完成，新影像已保存为:", output_file)