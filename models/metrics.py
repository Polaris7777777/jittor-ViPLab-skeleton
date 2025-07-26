import jittor as jt
from dataset.format import symmetric_bones
from dataset.format import symmetric_joint_pairs
def J2J(
    joints_a: jt.Var,
    joints_b: jt.Var,
) -> jt.Var:
    '''
    calculate J2J loss in [-1, 1]^3 cube
    
    joints_a: (J1, 3) joint

    joints_b: (J2, 3) joint
    '''
    assert isinstance(joints_a, jt.Var)
    assert isinstance(joints_b, jt.Var)
    assert joints_a.ndim == 2, "joints_a should be shape (J1, 3)"
    assert joints_b.ndim == 2, "joints_b should be shape (J2, 3)"
    dis1 = ((joints_a.unsqueeze(0) - joints_b.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss1 = dis1.min(dim=-1)
    dis2 = ((joints_b.unsqueeze(0) - joints_a.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss2 = dis2.min(dim=-1)
    return (loss1.mean() + loss2.mean()) / 2 / 2


def bone_length_symmetry_loss(pred, symmetric_bones=symmetric_bones):
    """
    对称骨骼长度损失。

    Args:
        pred: [B, J, 3]，预测的关键点位置

    Returns:
        scalar loss
    """
    loss = 0.0
    for (l_start, l_end), (r_start, r_end) in symmetric_bones:
        # 取出左右两段骨骼的端点
        l_vec = pred[:, l_end] - pred[:, l_start]  # [B, 3]
        r_vec = pred[:, r_end] - pred[:, r_start]  # [B, 3]

        # 计算长度（L2范数）
        l_len = jt.norm(l_vec, dim=1)  # [B]
        r_len = jt.norm(r_vec, dim=1)  # [B]

        # 损失：平方误差
        loss += jt.mean((l_len - r_len) ** 2)

    return loss / len(symmetric_bones)


def joint_symmetry_loss(
    joints: jt.Var, 
    symmetric_pairs: list[tuple[int, int]] = symmetric_joint_pairs, 
    spine_joints: list[int] = [0, 1, 2, 3, 4, 5]
) -> jt.Var:
    """
    对称骨架损失函数，约束左右 limb 成镜像对称（仅适用于 T-pose)
    同时约束脊柱上的点的x坐标为0
    
    Args:
        joints: shape (B, J, 3)  — 预测出的关键点位置
        symmetric_pairs: list of tuples — 对称关节点对
        spine_joints: list[int] — 脊柱上的关节点索引列表

    Returns:
        scalar 对称损失值
    """
    joints_mirror = joints.clone()
    joints_mirror[:, :, 0] *= -1  # X 轴翻转，构造镜像

    total_loss = 0.0
    for i, j in symmetric_pairs:
        loss_ij = jt.norm(joints[:, i] - joints_mirror[:, j], dim=-1).mean()
        loss_ji = jt.norm(joints[:, j] - joints_mirror[:, i], dim=-1).mean()
        total_loss += loss_ij + loss_ji
    
    mirror_loss = total_loss / (2 * len(symmetric_pairs))
    
    # 添加脊柱点x坐标一致的约束
    spine_loss = 0.0
    if spine_joints is not None and len(spine_joints) > 0:
        # 计算脊柱上点的x坐标的平均值
        spine_x_mean = joints[:, spine_joints, 0].mean(dim=1, keepdims=True)
        # 计算每个脊柱点的x坐标与平均值的差异
        spine_loss = jt.abs(joints[:, spine_joints, 0] - spine_x_mean).mean()
    
    return mirror_loss + spine_loss
