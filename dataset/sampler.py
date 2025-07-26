import numpy as np
from numpy import ndarray
from typing import Tuple
from abc import ABC, abstractmethod
import jittor as jt
import os  # 新增导入os模块用于路径操作
from .pointnet_ops import FurthestPointSampler

jt.flags.use_cuda = 1  # 确保在文件开头设置
class Sampler(ABC):
    def __init__(self):
        pass
    
    def _sample_barycentric(
        self,
        vertex_groups: ndarray,
        faces: ndarray,
        face_index: ndarray,
        random_lengths: ndarray,
    ):
        v_origins = vertex_groups[faces[face_index, 0]]
        v_vectors = vertex_groups[faces[face_index, 1:]]
        v_vectors -= v_origins[:, np.newaxis, :]
        
        sample_vector = (v_vectors * random_lengths).sum(axis=1)
        v_samples = sample_vector + v_origins
        return v_samples
    
    @abstractmethod
    def sample(
        self,
        vertices: ndarray,
        vertex_normals: ndarray,
        face_normals: ndarray,
        vertex_groups: dict[str, ndarray],
        faces: ndarray,
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        '''
        Args:
            vertices: (N, 3)
            vertex_normals: (N, 3)
            face_normals: (F, 3)
            vertex_groups: dict{name: shape (N, x)}
            face: (F, 3)
        Returns:
            vertex_samples: (num_samples, 3) - 采样得到的顶点坐标，包含原始顶点采样和表面采样
            normal_samples: (num_samples, 3) - 对应采样点的法向量
            vertex_groups_samples: dict{name: shape (num_samples, x)} - 对应采样点的顶点组属性
        '''
        return vertices, vertex_normals, vertex_groups

class SamplerMix(Sampler):
    '''
    Pick `vertex_samples` samples with SamplerOrigin, then pick `num_vertices`-`vertex_samples` samples
    with SamplerRandom.
    '''
    def __init__(self, num_samples: int, vertex_samples: int, cache_dir='datacache/mix'):
        super().__init__()
        self.num_samples = num_samples
        self.vertex_samples = vertex_samples
        self.cache_dir = cache_dir
    
    def sample(
        self,
        vertices: ndarray,
        vertex_normals: ndarray,
        face_normals: ndarray,
        vertex_groups: dict[str, ndarray],
        faces: ndarray,
        data_root=None,
        path=None,
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        '''
        Args:
            vertices: (N, 3)
            vertex_normals: (N, 3)
            face_normals: (F, 3)
            vertex_groups: dict{name: shape (N, x)}
            face: (F, 3)
            data_root: str, 数据根目录
            path: str, 数据相对路径，用于命名缓存文件
        Returns:
            vertex_samples: (num_samples, 3) - 采样得到的顶点坐标，包含原始顶点采样和表面采样
            normal_samples: (num_samples, 3) - 对应采样点的法向量
            vertex_groups_samples: dict{name: shape (num_samples, x)} - 对应采样点的顶点组属性
        '''
        if self.num_samples==-1:
            return vertices, vertex_normals, vertex_groups
            
        # # 生成缓存文件名 - 使用几何特征
        # vertices_count = vertices.shape[0]
        # faces_count = faces.shape[0]
        # # 计算顶点坐标的特征哈希值
        # vertices_mean = np.mean(vertices, axis=0)
        # vertices_std = np.std(vertices, axis=0)
        # vertices_feature = hash(f"{vertices_count}_{faces_count}_{np.sum(vertices_mean):.6f}_{np.sum(vertices_std):.6f}") % 1000000
        
        # # 如果有path使用path信息，否则使用生成的特征
        # if path:
        #     file_id = os.path.splitext(os.path.basename(path))[0]
        #     prefix = f"{file_id}_"
        # else:
        #     prefix = ""
            
        # # 生成缓存文件名，包含采样类型和参数
        # cache_filename = f"{prefix}v{vertices_count}_h{vertices_feature}_mix_{self.num_samples}_{self.vertex_samples}.npz"
        # cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # # 尝试从缓存加载采样结果
        # if os.path.exists(cache_path):
        #     try:
        #         cache_data = np.load(cache_path, allow_pickle=True)
        #         vertex_samples = cache_data['vertex_samples']
        #         normal_samples = cache_data['normal_samples']
        #         vertex_groups_samples = cache_data['vertex_groups_samples'].item()  # 将ndarray转回dict
        #         return vertex_samples, normal_samples, vertex_groups_samples
        #     except Exception as e:
        #         # print(f"Error loading mix sampling cache: {e}")
        #         pass  # 如果加载失败，继续原有采样逻辑
        
        # 1. sample vertices
        num_samples = self.num_samples
        perm = np.random.permutation(vertices.shape[0])
        vertex_samples = min(self.vertex_samples, vertices.shape[0])
        num_samples -= vertex_samples
        perm = perm[:vertex_samples]
        n_vertices = vertices[perm]
        n_normal = vertex_normals[perm]
        n_v = {name: v[perm] for name, v in vertex_groups.items()}
        
        # 2. sample surface
        perm = np.random.permutation(num_samples)
        vertex_samples, face_index, random_lengths = sample_surface(
            num_samples=num_samples,
            vertices=vertices,
            faces=faces,
            return_weight=True,
        )
        vertex_samples = np.concatenate([n_vertices, vertex_samples], axis=0)
        normal_samples = np.concatenate([n_normal, face_normals[face_index]], axis=0)
        vertex_groups_samples = {}
        for n, v in vertex_groups.items():
            g = self._sample_barycentric(
                vertex_groups=v,
                faces=faces,
                face_index=face_index,
                random_lengths=random_lengths,
            )
            vertex_groups_samples[n] = np.concatenate([n_v[n], g], axis=0)
        
        # # 保存结果到缓存
        # np.savez(
        #     cache_path,
        #     vertex_samples=vertex_samples,
        #     normal_samples=normal_samples,
        #     vertex_groups_samples=np.array([vertex_groups_samples], dtype=object)  # 将dict存储为ndarray
        # )
        
        return vertex_samples, normal_samples, vertex_groups_samples

class SamplerFPS(Sampler):
    '''
    使用最远点采样(FPS)进行顶点采样，使用Jittor加速
    '''
    def __init__(self, num_samples: int, cache_dir='datacache/fps'):
        super().__init__()
        self.num_samples = num_samples
        self.cache_dir = cache_dir
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def sample(
        self,
        vertices: ndarray,
        vertex_normals: ndarray,
        face_normals: ndarray,
        vertex_groups: dict[str, ndarray],
        faces: ndarray,
        data_root=None,
        path=None,
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        '''
        Args:
            vertices: (N, 3)
            vertex_normals: (N, 3)
            face_normals: (F, 3)
            vertex_groups: dict{name: shape (N, x)}
            face: (F, 3)
            data_root: str, 数据根目录
            path: str, 数据相对路径，用于命名缓存文件（现在可选）
        Returns:
            vertex_samples: (num_samples, 3) - FPS采样得到的顶点坐标
            normal_samples: (num_samples, 3) - 对应采样点的法向量
            vertex_groups_samples: dict{name: shape (num_samples, x)} - 对应采样点的顶点组属性
        '''
        if self.num_samples == -1:
            return vertices, vertex_normals, vertex_groups
        
        # # 生成缓存文件名 - 不依赖于path参数
        # # 使用顶点数量和顶点数据的特征生成唯一标识符
        # vertices_count = vertices.shape[0]
        # # 计算顶点坐标的特征哈希值 - 使用均值和标准差作为特征
        # vertices_mean = np.mean(vertices, axis=0)
        # vertices_std = np.std(vertices, axis=0)
        # # 使用这些特征生成一个简单的哈希值
        # vertices_feature = hash(f"{vertices_count}_{np.sum(vertices_mean):.6f}_{np.sum(vertices_std):.6f}") % 1000000
        
        # # 如果有path使用path信息，否则使用生成的特征
        # if path:
        #     file_id = os.path.splitext(os.path.basename(path))[0]
        #     prefix = f"{file_id}_"
        # else:
        #     prefix = ""
            
        # # 生成缓存文件名
        # cache_filename = f"{prefix}v{vertices_count}_h{vertices_feature}_fps_{self.num_samples}.npy"
        # cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # # 尝试从缓存加载采样索引
        # sampled_indices = None
        # if os.path.exists(cache_path):
        #     try:
        #         sampled_indices = np.load(cache_path)
        #         # print(f"Loaded FPS cache from {cache_path}")
        #     except Exception as e:
        #         # print(f"Error loading FPS cache: {e}")
        #         sampled_indices = None
        sampled_indices = None
        # 如果没有缓存或加载失败，执行FPS采样
        if sampled_indices is None:
            sampled_indices = sample_fps_jittor(
                vertices=vertices,
                num_samples=self.num_samples,
                vertex_groups=vertex_groups,
                return_indices=True
            )
            # 保存采样结果到缓存
            # np.save(cache_path, sampled_indices)
            # print(f"Saved FPS cache to {cache_path}")
        
        # 使用采样索引获取对应的顶点和属性
        sampled_vertices = vertices[sampled_indices]
        sampled_vertex_groups = {name: v[sampled_indices] for name, v in vertex_groups.items()}
        sampled_normals = vertex_normals[sampled_indices]
        
        return sampled_vertices, sampled_normals, sampled_vertex_groups

class SamplerFPSMix(Sampler):
    '''
    混合采样器：部分点使用FPS采样，剩余点在面片上随机采样
    '''
    def __init__(self, num_samples: int, fps_samples: int, cache_dir='datacache/fpsmix'):
        super().__init__()
        self.num_samples = num_samples
        self.fps_samples = min(fps_samples, num_samples)  # FPS采样点数不能超过总点数
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def sample(
        self,
        vertices: ndarray,
        vertex_normals: ndarray,
        face_normals: ndarray,
        vertex_groups: dict[str, ndarray],
        faces: ndarray,
        data_root=None,
        path=None,
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        if self.num_samples == -1:
            return vertices, vertex_normals, vertex_groups
            
        # # 生成缓存文件名
        # vertices_count = vertices.shape[0]
        # vertices_mean = np.mean(vertices, axis=0)
        # vertices_std = np.std(vertices, axis=0)
        # vertices_feature = hash(f"{vertices_count}_{np.sum(vertices_mean):.6f}_{np.sum(vertices_std):.6f}") % 1000000
        
        # prefix = os.path.splitext(os.path.basename(path))[0] + "_" if path else ""
        # cache_filename = f"{prefix}v{vertices_count}_h{vertices_feature}_fpsmix_{self.num_samples}_{self.fps_samples}.npz"
        # cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # # 尝试从缓存加载
        # if os.path.exists(cache_path):
        #     try:
        #         cache_data = np.load(cache_path, allow_pickle=True)
        #         vertex_samples = cache_data['vertex_samples']
        #         normal_samples = cache_data['normal_samples']
        #         vertex_groups_samples = cache_data['vertex_groups_samples'].item()
        #         return vertex_samples, normal_samples, vertex_groups_samples
        #     except:
        #         pass
        
        # 1. FPS采样
        fps_indices = sample_fps_jittor(
            vertices=vertices,
            num_samples=self.fps_samples,
            return_indices=True
        )
        fps_vertices = vertices[fps_indices]
        fps_normals = vertex_normals[fps_indices]
        fps_groups = {name: v[fps_indices] for name, v in vertex_groups.items()}
        
        # 2. 表面采样剩余点
        surface_samples = self.num_samples - self.fps_samples
        if surface_samples > 0:
            surf_vertices, face_index, random_lengths = sample_surface(
                num_samples=surface_samples,
                vertices=vertices,
                faces=faces,
                return_weight=True,
            )
            
            # 合并结果
            vertex_samples = np.concatenate([fps_vertices, surf_vertices], axis=0)
            normal_samples = np.concatenate([fps_normals, face_normals[face_index]], axis=0)
            
            vertex_groups_samples = {}
            for name, v in vertex_groups.items():
                surf_groups = self._sample_barycentric(
                    vertex_groups=v,
                    faces=faces,
                    face_index=face_index,
                    random_lengths=random_lengths,
                )
                vertex_groups_samples[name] = np.concatenate([fps_groups[name], surf_groups], axis=0)
        else:
            vertex_samples = fps_vertices
            normal_samples = fps_normals
            vertex_groups_samples = fps_groups
            
        # 保存到缓存
        # np.savez(
        #     cache_path,
        #     vertex_samples=vertex_samples,
        #     normal_samples=normal_samples,
        #     vertex_groups_samples=np.array([vertex_groups_samples], dtype=object)
        # )
        
        return vertex_samples, normal_samples, vertex_groups_samples

def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    return_weight: bool=False,
):
    '''
    Randomly pick samples according to face area.
    
    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py
    '''
    # get face area
    offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_weight = np.cross(offset_0, offset_1, axis=-1)
    face_weight = (face_weight * face_weight).sum(axis=1)
    
    weight_cum = np.cumsum(face_weight, axis=0)
    face_pick = np.random.rand(num_samples) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]
    
    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    if not return_weight:
        return vertex_samples
    return vertex_samples, face_index, random_lengths

def sample_fps_jittor(
    vertices: ndarray,
    num_samples: int,
    vertex_groups: dict[str, ndarray] = None,
    return_indices: bool = False,
):
    '''
    Farthest Point Sampling (FPS) using Jittor GPU acceleration.

    Args:
        vertices: (N, 3) numpy array
        num_samples: int
        vertex_groups: optional dict of {name: (N, x)}
        return_indices: bool

    Returns:
        Either (sampled_vertices, sampled_vertex_groups) or sampled_indices
    '''
    N = vertices.shape[0]

    if num_samples <= 0:
        if return_indices:
            return np.array([], dtype=int)
        return vertices[:0], {name: v[:0] for name, v in vertex_groups.items()}

    if num_samples >= N:
        sampled_indices = list(range(N))
        while len(sampled_indices) < num_samples:
            remaining = num_samples - len(sampled_indices)
            repeat_indices = sampled_indices[:min(remaining, N)]
            sampled_indices.extend(repeat_indices)
        sampled_indices = np.array(sampled_indices[:num_samples])
        if return_indices:
            return sampled_indices
        sampled_vertices = vertices[sampled_indices]
        sampled_vertex_groups = {name: v[sampled_indices] for name, v in vertex_groups.items()}
        return sampled_vertices, sampled_vertex_groups

    # ========= GPU FPS ==========

    vertices_jt = jt.array(vertices)  # (N, 3)
    min_dists = jt.full((N,), float('inf'))  # 每个点到采样集合的最小距离

    # 初始采样索引 (随机1个点)
    init_idx = np.random.randint(N)
    sampled_indices = jt.array([init_idx], dtype="int32")  # (1,)

    for _ in range(1, num_samples):
        last_point = vertices_jt[sampled_indices[-1]] # (1, 3)
        dists = jt.norm(vertices_jt - last_point, dim=1)  # (N,)
        min_dists = jt.minimum(min_dists, dists)  # 更新最短距离
        next_idx = jt.argmax(min_dists, dim=0)[0]  # 找最远点
        sampled_indices = jt.concat([sampled_indices, next_idx.reshape(1)], dim=0)

    sampled_indices_np = sampled_indices.numpy()  # 转为 numpy 索引
    if return_indices:
        return sampled_indices_np

    sampled_vertices = vertices[sampled_indices_np]
    sampled_vertex_groups = {name: v[sampled_indices_np] for name, v in vertex_groups.items()}

    return sampled_vertices, sampled_vertex_groups


class FPSWrapper(Sampler):
    '''
    Wrapper for PointNet's FPS implementation.
    '''
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples
        self.sampler = FurthestPointSampler(self.num_samples)

    def sample(
        self,
        vertices: ndarray,
        vertex_normals: ndarray,
        face_normals: ndarray,
        vertex_groups: dict[str, ndarray],
        faces: ndarray,
        data_root=None,
        path=None,
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        '''
        Args:
            vertices: (N, 3)
            vertex_normals: (N, 3)
            face_normals: (F, 3)
            vertex_groups: dict{name: shape (N, x)}
            face: (F, 3)
            data_root: str, 数据根目录
            path: str, 数据相对路径
        Returns:
            vertex_samples: (num_samples, 3)
            normal_samples: (num_samples, 3)
            vertex_groups_samples: dict{name: shape (num_samples, x)}
        '''
        if self.num_samples == -1:
            return vertices, vertex_normals, vertex_groups

        # 转换为jittor tensor并添加batch维度
        vertices_jt = jt.array(vertices).unsqueeze(0)  # (1, N, 3)
        
        # 执行FPS采样并获取索引
        sampled_indices_jt = self.sampler.execute_with_indices(vertices_jt)  # (1, num_samples)
        sampled_indices = sampled_indices_jt.squeeze(0).numpy()  # (num_samples,)

        # 直接使用索引获取采样点的顶点、法向量和属性
        sampled_vertices = vertices[sampled_indices]
        sampled_normals = vertex_normals[sampled_indices]
        sampled_vertex_groups = {name: v[sampled_indices] for name, v in vertex_groups.items()}
        
        return sampled_vertices, sampled_normals, sampled_vertex_groups
