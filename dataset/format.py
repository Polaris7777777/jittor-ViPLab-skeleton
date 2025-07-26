id_to_name = {
  0: 'hips',
  1: 'spine',
  2: 'chest',
  3: 'upper_chest',
  4: 'neck',
  5: 'head',
  6: 'l_shoulder',
  7: 'l_upper_arm',
  8: 'l_lower_arm',
  9: 'l_hand',
  10: 'r_shoulder',
  11: 'r_upper_arm',
  12: 'r_lower_arm',
  13: 'r_hand',
  14: 'l_upper_leg',
  15: 'l_lower_leg',
  16: 'l_foot',
  17: 'l_toe_base',
  18: 'r_upper_leg',
  19: 'r_lower_leg',
  20: 'r_foot',
  21: 'r_toe_base',
}

parents = [None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20,]

symmetric_joint_pairs = [
    (6, 10),   # l_shoulder ↔ r_shoulder
    (7, 11),   # l_upper_arm ↔ r_upper_arm
    (8, 12),   # l_lower_arm ↔ r_lower_arm
    (9, 13),   # l_hand ↔ r_hand
    (14, 18),  # l_upper_leg ↔ r_upper_leg
    (15, 19),  # l_lower_leg ↔ r_lower_leg
    (16, 20),  # l_foot ↔ r_foot
    (17, 21),  # l_toe_base ↔ r_toe_base
]

symmetric_bones = [
    # ((左侧骨骼起点, 左侧骨骼终点), (右侧骨骼起点, 右侧骨骼终点))
    ((6, 7), (10, 11)),    # 肩→上臂
    ((7, 8), (11, 12)),    # 上臂→下臂
    ((8, 9), (12, 13)),    # 下臂→手
    ((14, 15), (18, 19)),  # 上腿→下腿
    ((15, 16), (19, 20)),  # 下腿→脚
    ((16, 17), (20, 21)),  # 脚→脚趾
]

