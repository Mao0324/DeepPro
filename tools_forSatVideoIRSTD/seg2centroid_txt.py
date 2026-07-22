import cv2
import numpy as np
import os
import glob


def calculate_centroids(mask):
    """计算二值图中各目标的质心 ``(id, x, y)``，并分配动态 ID。"""
    num_labels, labels, stats, component_centroids = (
        cv2.connectedComponentsWithStats(mask, connectivity=8)
    )
    results = []

    # 遍历所有目标（忽略背景）
    for label in range(1, num_labels):
        left, top, width, height, _ = stats[label]
        component_labels = labels[top:top + height, left:left + width]
        target_mask = np.uint8(component_labels == label) * 255
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)

            if M["m00"] != 0:
                cx = left + M["m10"] / M["m00"]
                cy = top + M["m01"] / M["m00"]
                # 使用label作为唯一ID（从1开始）
                results.append((label, round(cx, 2), round(cy, 2)))
            else:
                cx, cy = component_centroids[label]
                # 使用label作为唯一ID（从1开始）
                results.append((label, round(cx, 2), round(cy, 2)))

    return results


def format_centroid_line(frame_idx, centroids):
    """Format one frame using the challenge centroid TXT convention."""
    centroids = sorted(centroids, key=lambda x: x[0])
    line = f"{frame_idx:05d}   {len(centroids)}   "
    for _, x_coordinate, y_coordinate in centroids:
        line += f"0     {x_coordinate:.2f}  {y_coordinate:.2f}   "
    return line.strip()


def process_sequence(seq_folder, output_txt, thresh):
    mask_files = sorted(glob.glob(os.path.join(seq_folder, '*.png')))
    if not mask_files:
        print(f"警告：在 {seq_folder} 中未找到PNG文件")
        return

    with open(output_txt, 'w') as f:
        for frame_idx, mask_path in enumerate(mask_files, start=1):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告：无法读取图像 {mask_path}")
                continue

            _, binary_mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)
            centroids = calculate_centroids(binary_mask)
            num_targets = len(centroids)

            f.write(format_centroid_line(frame_idx, centroids) + '\n')
            print(f"已处理：{os.path.basename(mask_path)} - 检测到 {num_targets} 个目标")


# 主处理流程保持不变
if __name__ == '__main__':
    root_dir = r"./log/sem_seg/SatVideoIRSDT__2025-06-19_18-10__SoftLoUloss_DeepPro-Plus_DataL40/visual"
    output_base_dir = r"./log/sem_seg/SatVideoIRSDT__2025-06-19_18-10__SoftLoUloss_DeepPro-Plus_DataL40/out_centroid"
    os.makedirs(output_base_dir, exist_ok=True)
    thresh = 100

    seq_folders = sorted(glob.glob(os.path.join(root_dir, '0*')))
    if not seq_folders:
        print(f"错误：在 {root_dir} 中未找到序列文件夹")
    else:
        print(f"找到 {len(seq_folders)} 个序列文件夹，开始处理...")
        for seq_folder_path in seq_folders:
            seq_name = os.path.basename(seq_folder_path)
            mask_folder_path = os.path.join(seq_folder_path)
            output_txt_path = os.path.join(output_base_dir, f"{seq_name}.txt")

            if os.path.exists(mask_folder_path):
                print(f"\n处理序列: {seq_name}")
                os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
                process_sequence(mask_folder_path, output_txt_path, thresh)
                print(f"处理完成！结果已保存至: {output_txt_path}")
            else:
                print(f"警告：序列 {seq_name} 中未找到mask文件夹")
        print("\n所有序列处理完成！")
