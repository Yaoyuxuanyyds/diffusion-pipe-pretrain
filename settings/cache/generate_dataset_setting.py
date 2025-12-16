# import math
# import os

# # -----------------------
# # 参数
# # -----------------------
# N = 2174  # 总子目录数
# K = 1    # 生成 32 个 toml
# OUTPUT_DIR = "/inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/cache/dataset"
# BASE_PATH = "/inspire/hdd/project/chineseculture/public/yuxuan/datasets/cc12m-unpacked"
# PREFIX = "cc12m-train-"

# # -----------------------
# # 创建输出目录
# # -----------------------
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------
# # 计算每个 shard 的大小
# # -----------------------
# chunk_size = math.ceil(N / K)

# print(f"Total dirs: {N}, Shards: {K}, Chunk size: {chunk_size}")

# # -----------------------
# # 生成 32 个 toml 文件
# # -----------------------
# for shard_id in range(K):
#     start = shard_id * chunk_size
#     end = min(N, (shard_id + 1) * chunk_size)

#     toml_path = os.path.join(OUTPUT_DIR, f"cc12m_dataset_{shard_id:04d}.toml")

#     with open(toml_path, "w") as f:
#         # 写 resolutions
#         f.write("resolutions = [256]\n\n")
        
#         # 写每个 directory 块
#         for i in range(start, end):
#             dirname = f"{PREFIX}{i:04d}"
#             full_path = os.path.join(BASE_PATH, dirname)

#             f.write(f"[[directory]]\n")
#             f.write(f"path = '{full_path}'\n")
#             f.write("num_repeats = 1\n\n")

#     print(f"Generated: {toml_path} ({start} - {end - 1})")

# print("DONE.")



import math
import os

# -----------------------
# 参数（可调）
# -----------------------
START_IDX = 1700        # cc12m-train-0001
END_IDX = 1767       # cc12m-train-2174（包含）
K = 8                # 生成多少个 toml

OUTPUT_DIR = "/inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/cache/dataset/extra"
BASE_PATH = "/inspire/hdd/project/chineseculture/public/yuxuan/datasets/cc12m-unpacked"
PREFIX = "cc12m-train-"

# -----------------------
# 派生参数
# -----------------------
N = END_IDX - START_IDX + 1
chunk_size = math.ceil(N / K)

print(f"Dirs: [{START_IDX:04d} - {END_IDX:04d}]")
print(f"Total dirs: {N}, Shards: {K}, Chunk size: {chunk_size}")

# -----------------------
# 创建输出目录
# -----------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# 生成 toml
# -----------------------
for shard_id in range(K):
    local_start = shard_id * chunk_size
    local_end = min(N, (shard_id + 1) * chunk_size)

    if local_start >= local_end:
        break

    global_start = START_IDX + local_start
    global_end = START_IDX + local_end - 1

    toml_path = os.path.join(
        OUTPUT_DIR,
        f"cc12m_dataset_25-{shard_id:02d}.toml"
    )

    with open(toml_path, "w") as f:
        f.write("resolutions = [256]\n\n")

        for idx in range(global_start, global_end + 1):
            dirname = f"{PREFIX}{idx:04d}"
            full_path = os.path.join(BASE_PATH, dirname)

            f.write("[[directory]]\n")
            f.write(f"path = '{full_path}'\n")
            f.write("num_repeats = 1\n\n")

    print(
        f"Generated: {toml_path} "
        f"({global_start:04d} - {global_end:04d})"
    )

print("DONE.")
