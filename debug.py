import os
import numpy as np
import matplotlib.pyplot as plt
import jittor as jt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from dataset.format import id_to_name, parents, symmetric_joint_pairs
from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix, SamplerFPS

# 设置Jittor
jt.flags.use_cuda = 1

def check_symmetry(joints, pair_indices=None, tolerance=0.05):
    """
    检查关节是否沿x轴对称
    
    Args:
        joints: 关节坐标, shape (J, 3)
        pair_indices: 对称关节对的索引列表，如果为None则使用symmetric_joint_pairs
        tolerance: 对称误差容忍度
    
    Returns:
        is_symmetric: 是否对称
        errors: 每对关节的对称误差
    """
    if pair_indices is None:
        pair_indices = symmetric_joint_pairs
        
    is_symmetric = True
    errors = []
    
    print("检查关节x轴对称性:")
    for pair in pair_indices:
        left_joint = joints[pair[0]]
        right_joint = joints[pair[1]]
        
        # 检查x坐标是否近似相反
        x_error = abs(left_joint[0] + right_joint[0])
        # 检查y坐标是否接近
        y_error = abs(left_joint[1] - right_joint[1])
        # 检查z坐标是否接近
        z_error = abs(left_joint[2] - right_joint[2])
        
        error = (x_error, y_error, z_error)
        errors.append(error)
        
        pair_symmetric = x_error < tolerance #and y_error < tolerance and z_error < tolerance
        if not pair_symmetric:
            is_symmetric = False
            
        print(f"  关节对 {id_to_name[pair[0]]:12} ↔ {id_to_name[pair[1]]:12}: "
              f"{'✓' if pair_symmetric else '✗'} "
              f"(x误差: {x_error:.4f}, y误差: {y_error:.4f}, z误差: {z_error:.4f})")
    
    return is_symmetric, errors

def check_spine_at_origin(joints, spine_indices=None, tolerance=0.05):
    """
    检查spine是否位于原点附近
    
    Args:
        joints: 关节坐标, shape (J, 3)
        spine_indices: spine相关关节的索引，默认为[0, 1, 2, 3]
        tolerance: 误差容忍度
    
    Returns:
        is_at_origin: spine是否在原点附近
        stats: spine位置统计信息
    """
    if spine_indices is None:
        spine_indices = [0, 1, 2, 3, 4, 5]  # hips, spine, chest, upper_chest
        
    spine_joints = joints[spine_indices]
    
    # 计算spine各坐标轴的平均值
    x_avg = np.mean(spine_joints[:, 0])
    y_avg = np.mean(spine_joints[:, 1])
    z_avg = np.mean(spine_joints[:, 2])
    
    # 检查x坐标是否接近0
    is_at_origin_x = abs(x_avg) < tolerance
    
    print(f"\n检查spine是否在原点附近:")
    print(f"  X轴: {'✓' if is_at_origin_x else '✗'} (平均值: {x_avg:.4f})")
    print(f"  Y轴平均值: {y_avg:.4f}")
    print(f"  Z轴平均值: {z_avg:.4f}")
    
    stats = {
        'x_avg': x_avg,
        'y_avg': y_avg,
        'z_avg': z_avg
    }
    
    return is_at_origin_x, stats

def check_spine_symmetry(joints, spine_indices=None, pair_indices=None, tolerance=0.05):
    """
    检查是否相对于spine的y-z平面对称
    
    Args:
        joints: 关节坐标, shape (J, 3)
        spine_indices: spine相关关节的索引，默认为[0, 1, 2, 3]
        pair_indices: 对称关节对的索引列表，如果为None则使用symmetric_joint_pairs
        tolerance: 对称误差容忍度
        
    Returns:
        is_symmetric: 是否关于spine对称
        errors: 每对关节的对称误差
    """
    if spine_indices is None:
        spine_indices = [0, 1, 2, 3, 4, 5]  # hips, spine, chest, upper_chest
    if pair_indices is None:
        pair_indices = symmetric_joint_pairs
    
    # 计算脊柱的y-z平面位置 (使用spine关节的平均坐标)
    spine_x = np.mean(joints[spine_indices, 0])
    spine_y = np.mean(joints[spine_indices, 1])
    spine_z = np.mean(joints[spine_indices, 2])
    
    print(f"\n检查相对于spine的对称性:")
    print(f"  Spine中心位置: ({spine_x:.4f}, {spine_y:.4f}, {spine_z:.4f})")
    
    is_symmetric = True
    errors = []
    
    for pair in pair_indices:
        left_joint = joints[pair[0]]
        right_joint = joints[pair[1]]
        
        # 计算左右关节到spine平面的距离 (应该大小相等，符号相反)
        left_dist_to_spine = left_joint[0] - spine_x
        right_dist_to_spine = right_joint[0] - spine_x
        
        # 检查距离是否对称 (绝对值相等，符号相反)
        dist_error = abs(left_dist_to_spine + right_dist_to_spine)
        
        # y和z坐标相对于spine的偏移应该接近
        y_error = abs((left_joint[1] - spine_y) - (right_joint[1] - spine_y))
        z_error = abs((left_joint[2] - spine_z) - (right_joint[2] - spine_z))
        
        pair_symmetric = dist_error < tolerance #and y_error < tolerance and z_error < tolerance
        if not pair_symmetric:
            is_symmetric = False
            
        error = (dist_error, y_error, z_error)
        errors.append(error)
        
        print(f"  关节对 {id_to_name[pair[0]]:12} ↔ {id_to_name[pair[1]]:12}: "
              f"{'✓' if pair_symmetric else '✗'} "
              f"(x距离误差: {dist_error:.4f}, y误差: {y_error:.4f}, z误差: {z_error:.4f})")
    
    return is_symmetric, errors

def visualize_skeleton(joints, title="骨架可视化", highlight_pairs=None):
    """
    可视化骨架结构
    
    Args:
        joints: 关节坐标, shape (J, 3)
        title: 图表标题
        highlight_pairs: 需要高亮显示的关节对列表
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制骨架连接
    for i, parent in enumerate(parents):
        if parent is not None:
            ax.plot([joints[i, 0], joints[parent, 0]],
                    [joints[i, 1], joints[parent, 1]],
                    [joints[i, 2], joints[parent, 2]], 'gray', alpha=0.7)
    
    # 绘制关节点
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='blue', marker='o')
    
    # 标注关节名称
    for i, (x, y, z) in enumerate(joints):
        ax.text(x, y, z, f"{i}:{id_to_name[i]}", fontsize=8)
    
    # 高亮显示特定关节对
    if highlight_pairs:
        for pair in highlight_pairs:
            left_idx, right_idx = pair
            # 连接对称关节对
            ax.plot([joints[left_idx, 0], joints[right_idx, 0]],
                    [joints[left_idx, 1], joints[right_idx, 1]],
                    [joints[left_idx, 2], joints[right_idx, 2]], 'red', linestyle='--')
            # 高亮左右关节
            ax.scatter(joints[left_idx, 0], joints[left_idx, 1], joints[left_idx, 2], c='red', s=100)
            ax.scatter(joints[right_idx, 0], joints[right_idx, 1], joints[right_idx, 2], c='red', s=100)
    
    # 绘制坐标轴
    # X轴 - 红色
    ax.quiver(0, 0, 0, 0.2, 0, 0, color='red', arrow_length_ratio=0.1)
    ax.text(0.25, 0, 0, "X", color='red')
    # Y轴 - 绿色
    ax.quiver(0, 0, 0, 0, 0.2, 0, color='green', arrow_length_ratio=0.1)
    ax.text(0, 0.25, 0, "Y", color='green')
    # Z轴 - 蓝色
    ax.quiver(0, 0, 0, 0, 0, 0.2, color='blue', arrow_length_ratio=0.1)
    ax.text(0, 0, 0.25, "Z", color='blue')
    
    # 设置坐标轴范围和标签
    max_range = np.max(np.abs(joints)) * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title(title)
    
    # 添加spine位置指示
    spine_indices = [0, 1, 2, 3]
    spine_x = np.mean(joints[spine_indices, 0])
    spine_y = np.mean(joints[spine_indices, 1])
    spine_z = np.mean(joints[spine_indices, 2])
    ax.scatter(spine_x, spine_y, spine_z, c='green', s=200, marker='*', label='Spine中心')
    
    # 绘制spine的y-z平面
    xx, zz = np.meshgrid(
        np.linspace(spine_y-max_range/2, spine_y+max_range/2, 2),
        np.linspace(spine_z-max_range/2, spine_z+max_range/2, 2)
    )
    yy = np.ones_like(xx) * spine_x
    ax.plot_surface(yy, xx, zz, color='g', alpha=0.1)
    
    plt.legend()
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='检查骨骼数据的对称性')
    parser.add_argument('--data_root', type=str, default='data', help='数据根目录')
    parser.add_argument('--data_list', type=str, default='data/train_list.txt', help='数据列表文件')
    parser.add_argument('--max_samples', type=int, default=1000, help='最大检查样本数量')
    parser.add_argument('--tolerance', type=float, default=0.01, help='对称误差容忍度')
    parser.add_argument('--visualize', action='store_true', help='是否可视化骨架')
    parser.add_argument('--save_path', type=str, default='debug_output', help='保存结果的目录')
    parser.add_argument('--sampler_type', type=str, default='none', choices=['mix', 'fps', 'none'], help='采样器类型')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.save_path, exist_ok=True)
    
    # 创建数据加载器
    if args.sampler_type == 'none':
        # 不进行采样
        sampler = SamplerMix(num_samples=-1, vertex_samples=0)
    elif args.sampler_type == 'mix':
        sampler = SamplerMix(num_samples=2048, vertex_samples=1024)
    else:
        sampler = SamplerFPS(num_samples=2048)
    
    data_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.data_list,
        train=False,
        batch_size=1,  # 每次处理一个样本
        shuffle=False,
        sampler=sampler,
        transform=transform,
    )
    
    print(f"加载了 {len(data_loader)} 个数据样本")
    
    # 统计信息
    total_samples = len(data_loader)
    x_symmetric_count = 0
    spine_origin_count = 0
    spine_symmetric_count = 0
    
    # 结果汇总
    results = []
    
    # 遍历数据集
    for i, data in enumerate(data_loader):
        # if i >= args.max_samples:
        #     break
            
        joints = data['joints'][0].numpy()  # [J, 3]
        
        print(f"\n===== 样本 {i+1}/{total_samples} =====")
        
        # 1. 检查x轴对称性
        x_symmetric, x_errors = check_symmetry(joints, tolerance=args.tolerance)
        if x_symmetric:
            x_symmetric_count += 1
            
        # 2. 检查spine是否在原点
        spine_at_origin, spine_stats = check_spine_at_origin(joints, tolerance=args.tolerance)
        if spine_at_origin:
            spine_origin_count += 1
            
        # 3. 检查相对于spine的对称性
        spine_symmetric, spine_sym_errors = check_spine_symmetry(joints, tolerance=args.tolerance)
        if spine_symmetric:
            spine_symmetric_count += 1
            
        # 保存此样本的结果
        results.append({
            'sample_id': i,
            'x_symmetric': x_symmetric,
            'spine_at_origin': spine_at_origin,
            'spine_symmetric': spine_symmetric,
            'x_errors': x_errors,
            'spine_stats': spine_stats,
            'spine_sym_errors': spine_sym_errors
        })
        
        # 可视化
        if args.visualize:
            # 找出问题最大的关节对
            problem_pairs = []
            for j, (pair, error) in enumerate(zip(symmetric_joint_pairs, x_errors)):
                if max(error) > args.tolerance:
                    problem_pairs.append(pair)
            
            fig = visualize_skeleton(
                joints, 
                title=f"样本 {i+1} 骨架结构 (X对称: {'是' if x_symmetric else '否'}, Spine对称: {'是' if spine_symmetric else '否'})",
                highlight_pairs=problem_pairs
            )
            
            # 保存图像
            fig.savefig(os.path.join(args.save_path, f"sample_{i+1}_skeleton.png"))
            plt.close(fig)
    
    # 打印汇总结果
    print("\n===== 结果汇总 =====")
    print(f"检查了 {total_samples} 个样本")
    print(f"X轴对称的样本数量: {x_symmetric_count} ({x_symmetric_count/total_samples*100:.1f}%)")
    print(f"Spine在原点的样本数量: {spine_origin_count} ({spine_origin_count/total_samples*100:.1f}%)")
    print(f"相对于Spine对称的样本数量: {spine_symmetric_count} ({spine_symmetric_count/total_samples*100:.1f}%)")
    
    # 保存详细结果
    summary_path = os.path.join(args.save_path, "symmetry_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"检查了 {total_samples} 个样本\n")
        f.write(f"X轴对称的样本数量: {x_symmetric_count} ({x_symmetric_count/total_samples*100:.1f}%)\n")
        f.write(f"Spine在原点的样本数量: {spine_origin_count} ({spine_origin_count/total_samples*100:.1f}%)\n")
        f.write(f"相对于Spine对称的样本数量: {spine_symmetric_count} ({spine_symmetric_count/total_samples*100:.1f}%)\n")
        
        # 添加样本详细信息
        for i, result in enumerate(results):
            f.write(f"\n样本 {i+1} 详细信息:\n")
            f.write(f"  X轴对称: {'是' if result['x_symmetric'] else '否'}\n")
            f.write(f"  Spine在原点: {'是' if result['spine_at_origin'] else '否'}\n")
            f.write(f"  相对于Spine对称: {'是' if result['spine_symmetric'] else '否'}\n")
            
            # Spine位置统计
            f.write(f"  Spine平均位置: X={result['spine_stats']['x_avg']:.4f}, Y={result['spine_stats']['y_avg']:.4f}, Z={result['spine_stats']['z_avg']:.4f}\n")
    
    print(f"\n详细结果已保存至 {summary_path}")

if __name__ == "__main__":
    main()
