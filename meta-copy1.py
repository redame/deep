import os
os.environ["KERAS_BACKEND"] = "torch"  # 必须在导入 keras 前设置
import sys
import glob
import pandas as pd
import numpy as np
import torch
import gc
import logging
from typing import List, Tuple, Dict, Optional
import shutil
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from keras.models import Model, load_model
import keras
# =============================================================================
# 初始化logger（解决NameError）
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FACTORY] %(levelname)s: %(message)s"
)
logger = logging.getLogger("factory_train_final")
# =============================================================================
# 修复：替换 __file__ 路径导入方式（适配交互式环境和脚本运行）
# =============================================================================
current_dir = os.getcwd()
sys.path.append(current_dir)
logger.info(f"已添加当前目录到系统路径：{current_dir}")
# 验证 deeptrain.py 是否存在
deeptrain_path = os.path.join(current_dir, "deeptrain.py")
if not os.path.exists(deeptrain_path):
    raise FileNotFoundError(f"未找到 deeptrain.py，请确保它与当前文件在同一目录：{current_dir}")
# 导入 deeptrain.py 中的核心函数和配置
from deeptrain1 import (
    ModelConfig, set_all_seeds, load_or_fit_label_encoders, preprocess_data_deepcrossing,
    UpdatedDeepCrossingDataGenerator, build_model, train_model, evaluate_predictions,
    get_data_paths, concat_parquet_files, huber_loss, evaluate_test_folder
)
# =============================================================================
# 配置参数（按需调整）
# =============================================================================
class FactoryConfig:
    SELECTED_FOLDER_INDEXES = [36,37,38,39,40,41,42,43]
    #SELECTED_FOLDER_INDEXES = [33,34, 35,36,37,38,39,40,41,42,43] # 目标文件夹索引（可修改为其他索引如53、363）
    CANDIDATE_COUNT = 50        # 每组候选训练数据数量
    TIME_WINDOW_KEYS = ["1130", "1500"]  # 支持的时间窗口类型（与selected_folder文件名匹配）
    FACTORY_CACHE_DIR = "pt/factory_cache"  # 聚合特征和索引存储目录
    INDEX_CSV_PATH = os.path.join(FACTORY_CACHE_DIR, "offline_enum_index.csv")  # 评估结果索引文件
    ONE_HOT_MAP = {
        "systematic": [1, 0, 0],
        "sliding": [0, 1, 0],
        "topk": [0, 0, 1]
    }  # 候选选择方案的one-hot编码
    SEED = 42  # 全局随机种子（与deeptrain保持一致）
# 设置全局随机种子（确保结果可复现）
set_all_seeds(FactoryConfig.SEED)
# =============================================================================
# 工具函数：文件夹与时间处理
# =============================================================================
def normalize_factory_features(feat_df: pd.DataFrame):
    """
    分开处理全量数据和候选数据：
    - 第一行（全量数据，user_id）：保持原始值
    - 后续行（候选数据，item_id）：归一化
    """
    save_path = os.path.join(FactoryConfig.FACTORY_CACHE_DIR, "factory_scaler.pkl")
    # 归一化列（0~325 + 328）
    normalize_cols = [f"feat_{i}" for i in range(326)] + ["feat_328"]
    keep_cols = [f"feat_{i}" for i in range(326, 328)] + [f"feat_{i}" for i in range(329, 332)]
    # 分离全量数据（第一行）和候选数据（其余行）
    full_row = feat_df.iloc[[0]].copy()
    candidate_rows = feat_df.iloc[1:].copy()
    # 初始化或加载Scaler
    if os.path.exists(save_path):
        scaler = joblib.load(save_path)
    else:
        scaler = RobustScaler()
        scaler.fit(candidate_rows[normalize_cols])
        joblib.dump(scaler, save_path)
    # 对候选数据归一化
    candidate_rows[normalize_cols] = scaler.transform(candidate_rows[normalize_cols])
    # 合并回去
    feat_df = pd.concat([full_row, candidate_rows], axis=0)
    logger.info(f"候选数据归一化完成 (RobustScaler)，全量数据保持原始值，参数文件: {save_path}")
    return feat_df
def extract_folder_datetime(folder_name: str) -> Tuple[str, str]:
    """
    修复版：直接从文件夹名截取，保持原始位数 (如 1500, 0940)
    返回：(datetime_str: "2025-1-7-1500", time_window: "1500")
    """
    try:
        # 逻辑：Q9_n-sh000300-2025-1-7-1500 -> 提取 2025-1-7-1500
        parts = folder_name.split('-')
        # 1. 提取时间窗口（最后一部分）
        time_window = parts[-1] 
        # 2. 提取日期+时间部分（倒数第四部分到最后）
        date_parts = parts[-4:]
        datetime_str = "-".join(date_parts)
        # 3. 验证时间窗口是否在配置列表中 
        if time_window not in FactoryConfig.TIME_WINDOW_KEYS:
            raise ValueError(f"时间窗口 {time_window} 不在支持列表 {FactoryConfig.TIME_WINDOW_KEYS} 中")
        return datetime_str, time_window
    except Exception as e:
        logger.error(f"提取文件夹时间失败：{folder_name}, 错误：{e}") 
        raise
def get_candidate_files(inter_data_dir: str, time_window: str, inter_feat_df: pd.DataFrame) -> List[str]:
    """
    根据 interact_feat.parquet 的 item_id 过滤候选文件：
    - 直接从文件名字符串截取 ID，确保时间位数（如1500）保持原样
    """
    # 1. 确定时间窗口文件夹路径 [cite: 6]
    time_window_dir = os.path.join(inter_data_dir, "all", time_window)
    if not os.path.exists(time_window_dir):
        raise FileNotFoundError(f"时间窗口文件夹不存在：{time_window_dir}")
    # 2. 获取目录下所有匹配的 parquet 文件 [cite: 7]
    all_files = glob.glob(os.path.join(time_window_dir, "sxy-sh000300-*.parquet"))
    if not all_files:
        raise ValueError(f"时间窗口 {time_window} 下无候选文件：{time_window_dir}")
    # 3. 提取有效 item_id 集合 [cite: 7]
    valid_item_ids = set(inter_feat_df["item_id"].unique())
    # 4. 建立 ID 到文件路径的映射（字符串截取逻辑） 
    file_map = {}
    for file_path in all_files:
        file_name = os.path.basename(file_path).replace(".parquet", "")
        # 逻辑：sxy-sh000300-2025-1-7-1500 -> 2025-1-7-1500
        # 取 '-' 分割后的最后四段：年-月-日-时间
        parts = file_name.split('-')
        item_id_str = "-".join(parts[-4:]) 
        file_map[item_id_str] = file_path
    # 5. 过滤：只保留在 interact_feat 中出现的 item_id 
    filtered_files = [file_map[item_id] for item_id in valid_item_ids if item_id in file_map]
    if not filtered_files:
        # 打印调试信息，方便对比 ID 格式 
        sample_feat_id = list(valid_item_ids)[0] if valid_item_ids else "None"
        sample_file_id = list(file_map.keys())[0] if file_map else "None"
        logger.error(f"匹配失败！interact_feat 样例: {sample_feat_id}, 文件截取样例: {sample_file_id}")
        raise ValueError("过滤后无有效候选文件，请检查 interact_feat 与文件名的对应关系")
# 6. 核心修复：将字符串转换为 datetime 对象进行排序
    def sort_key(file_path):
        file_name = os.path.basename(file_path).replace(".parquet", "")
        # 提取日期部分，例如 "2023-10-9-1500"
        parts = file_name.split('-')
        item_id_str = "-".join(parts[-4:])
        # 使用 strptime 解析日期，这样 10-9 会被正确识别为早于 10-23
        return datetime.strptime(item_id_str, "%Y-%m-%d-%H%M")
    candidate_files_sorted = sorted(filtered_files, key=sort_key)
    logger.info(f"找到 {len(candidate_files_sorted)} 个候选文件（已按时间线校准排序）")
    return candidate_files_sorted
# =============================================================================
# 工具函数：候选数据选择方案（3种方案，核心修正Top-K逻辑）
# =============================================================================
from collections import defaultdict
def generate_systematic_groups(candidate_files: List[str], count: int = 50) -> List[List[str]]:
    """
    方案1：工作日无限循环提取（直到数据耗尽）
    规则：
    1. 识别最老数据对应的星期（起始点） 。
    2. 按 [起始星期, 起始+1, ..., 周五, 周一, ...] 的顺序循环。
    3. 每次从当前星期的“文件池”中提取最前的 count 个文件。
    4. 只要任何一个星期的剩余文件不足 count 个，循环停止。
    """
    if not candidate_files:
        return []
    # 1. 将所有文件按星期几归类 (0=周一, 4=周五)
    weekday_pools = defaultdict(list)
    parsed_data = []
    for file_path in candidate_files:
        file_name = os.path.basename(file_path).replace(".parquet", "")
        parts = file_name.split('-') 
        # 提取日期部分：sxy-sh000300-2025-1-7-1500 -> 2025-1-7 [cite: 5]
        date_str = "-".join(parts[-4:-1]) 
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            weekday = dt.weekday()
            if weekday <= 4:  # 仅保留工作日 [cite: 11]
                weekday_pools[weekday].append(file_path)
                parsed_data.append((dt, weekday))
        except Exception as e:
            logger.warning(f"解析日期失败 {date_str}: {e}")
    if not parsed_data:
        return [] 
    # 2. 找到最老文件对应的星期几作为起始点 
    # candidate_files 已按时间排序，取第一个有效解析的文件即可
    first_dt, start_weekday = min(parsed_data, key=lambda x: x[0])
    # 3. 准备循环变量
    groups = []
    current_weekday = start_weekday
    # 记录每个星期池已经提取到了第几个索引
    pool_pointers = defaultdict(int) 
    weekday_names = ["周一", "周二", "周三", "周四", "周五"]
    logger.info(f"起始数据星期：{weekday_names[start_weekday]}，开始全量循环提取...")
    # 4. 开启无限循环提取，直到数据不足
    while True:
        # 获取当前星期池
        pool = weekday_pools[current_weekday]
        start_idx = pool_pointers[current_weekday]
        end_idx = start_idx + count
        # 检查当前星期池是否还有足够数据
        if end_idx > len(pool):
            logger.info(f"停止提取：{weekday_names[current_weekday]} 数据已耗尽 (剩余 {len(pool)-start_idx} 条)")
            break
        # 提取一组数据
        group = pool[start_idx:end_idx]
        groups.append(group)
        # 更新指针：此步决定是“滑动提取”还是“跳跃提取”
        # 若要完全不重复提取，使用 pool_pointers[current_weekday] += count
        # 若要按步长滑动（如每次往后移1天），使用 += 1。这里建议使用 += 1 以获得更多训练样本
        pool_pointers[current_weekday] += 1 
        # 移动到下一个工作日 (0->1->2->3->4->0)
        current_weekday = (current_weekday + 1) % 5
        # 如果下一个星期完全没数据（比如数据集中根本没周五），也需要跳出避免死循环
        if not weekday_pools[current_weekday] and any(weekday_pools.values()):
             # 尝试再找下一个有数据的日子
             search_count = 0
             while not weekday_pools[current_weekday] and search_count < 5:
                 current_weekday = (current_weekday + 1) % 5
                 search_count += 1
             if search_count >= 5: break
    logger.info(f"循环提取完成，共生成 {len(groups)} 组候选数据")
    return groups
def generate_sliding_groups(candidate_files: List[str], count: int = 50, step: int = 5) -> List[List[str]]:
    """
    方案2：时间滑动窗口（连续50条，步长5）
    规则：连续选取50条数据，窗口每次后移5条，捕捉时间连续性特征
    返回：每组50条文件路径的列表
    """
    total = len(candidate_files)
    if total < count:
        logger.warning(f"滑动窗口候选文件不足（总文件数：{total} < {count}），跳过该方案")
        return []
    groups = []
    for start in range(0, total - count + 1, step):
        end = start + count
        group = candidate_files[start:end]
        groups.append(group)
    logger.info(f"生成滑动窗口组：{len(groups)} 组")
    return groups
def generate_topk_groups(inter_feat_df: pd.DataFrame, candidate_files: List[str], count: int = 50) -> List[List[str]]:
    """
    方案3：Top-K相似度（基于字符串截取匹配 ID，确保时间位数一致）
    规则：特征均值越小越相似，分3组（最相似/中间/最不相似），每组50条 [cite: 13]
    """
    total_candidates = len(candidate_files)
    if total_candidates < count * 3:
        logger.warning(f"Top-K候选文件不足（总文件数：{total_candidates} < {count*3}），跳过该方案")
        return []
    # 1. 筛选相似性特征列 [cite: 13]
    similar_feat_cols = [col for col in inter_feat_df.columns if 'NC-n' in col or 'nad-n' in col]
    if not similar_feat_cols:
        raise ValueError("interact_feat.parquet中未找到11NC-n/13NC-n等相似性特征列") 
    logger.info(f"Top-K排序使用 {len(similar_feat_cols)} 个相似性特征")
    # 2. 计算每条候选数据的特征均值并排序（均值越小越相似） 
    inter_feat_df['similarity_mean'] = inter_feat_df[similar_feat_cols].mean(axis=1)
    inter_feat_sorted = inter_feat_df.sort_values('similarity_mean', ascending=True).reset_index(drop=True)
    # 3. 建立 item_id 到文件路径的映射（关键修复：使用字符串截取逻辑） [cite: 8, 22]
    item_id_to_file = {}
    for file_path in candidate_files:
        file_name = os.path.basename(file_path).replace(".parquet", "")
        # 逻辑：sxy-sh000300-2025-1-7-1500 -> 提取 2025-1-7-1500
        # 直接截取最后 4 段，确保像 1500 这样的字符串不会因转化为数字而丢失末尾的 0 [cite: 8, 22]
        parts = file_name.split('-')
        item_id_str = "-".join(parts[-4:]) 
        item_id_to_file[item_id_str] = file_path
    # 4. 生成排序后的有效文件路径列表 [cite: 15]
    sorted_files = []
    for _, row in inter_feat_sorted.iterrows():
        item_id_str = row['item_id']
        if item_id_str in item_id_to_file:
            sorted_files.append(item_id_to_file[item_id_str])
        else:
            # 如果依然匹配不到，这里可以方便地打印出尝试匹配的 ID 格式进行调试
            logger.debug(f"未找到匹配项: {item_id_str}") 
    # 5. 确保有效文件数满足分组需求 [cite: 15, 16]
    if len(sorted_files) < count * 3:
        logger.warning(f"Top-K有效匹配文件不足（有效数：{len(sorted_files)} < {count*3}）")
        return []
    # 6. 分3组：最相似（均值小）、中间、最不相似（均值大） [cite: 16]
    groups = [
        sorted_files[:count],  # Top-K最小（最相似） [cite: 16]
        sorted_files[len(sorted_files)//2 - count//2 : len(sorted_files)//2 + count//2],  # 中间组 [cite: 16]
        sorted_files[-count:]  # Top-K最大（最不相似） [cite: 16]
    ]
    logger.info(f"成功生成Top-K组：3组，匹配成功数：{len(sorted_files)}")
    return groups
def generate_all_candidate_groups(inter_feat_df: pd.DataFrame, candidate_files: List[str]) -> Dict[str, List[List[str]]]:
    """
    生成所有3种方案的候选组，过滤空组
    返回：{方案名: 组列表}
    """
    groups = {
        "systematic": generate_systematic_groups(candidate_files, FactoryConfig.CANDIDATE_COUNT),
        "sliding": generate_sliding_groups(candidate_files, FactoryConfig.CANDIDATE_COUNT),
        "topk": generate_topk_groups(inter_feat_df, candidate_files, FactoryConfig.CANDIDATE_COUNT)
    }
    # 过滤空组（仅保留有有效组的方案）
    groups = {k: v for k, v in groups.items() if v}
    total_groups = sum(len(v) for v in groups.values())
    logger.info(f"最终生成有效候选组：{total_groups} 组（方案：{list(groups.keys())}）")
    return groups
# =============================================================================
# 工具函数：断点续跑（避免重复处理）
# =============================================================================
def init_index_csv():
    """初始化评估结果索引CSV（若不存在）"""
    os.makedirs(FactoryConfig.FACTORY_CACHE_DIR, exist_ok=True)
    if not os.path.exists(FactoryConfig.INDEX_CSV_PATH):
        # 索引文件字段：文件夹名、方案类型、组编号、特征路径、各项评估指标、时间戳
        df = pd.DataFrame(columns=[
            "selected_folder", "scheme", "group_id", "feature_path",
            "test_loss", "precision@5", "penalty@5", "comprehensive_score", "timestamp"
        ])
        df.to_csv(FactoryConfig.INDEX_CSV_PATH, index=False)
        logger.info(f"初始化索引文件：{FactoryConfig.INDEX_CSV_PATH}")
def is_group_processed(selected_folder: str, scheme: str, group_id: int) -> bool:
    """检查组是否已处理（通过索引CSV判断）"""
    if not os.path.exists(FactoryConfig.INDEX_CSV_PATH):
        return False
    df = pd.read_csv(FactoryConfig.INDEX_CSV_PATH)
    mask = (
        (df["selected_folder"] == selected_folder) &
        (df["scheme"] == scheme) &
        (df["group_id"] == group_id)
    )
    return mask.any()
def write_to_index(selected_folder: str, scheme: str, group_id: int, feature_path: str,
                  test_loss: float, precision5: float, penalty5: float, comprehensive_score: float):
    """将组的评估结果写入索引CSV"""
    init_index_csv()
    # 构造新记录
    new_row = pd.DataFrame({
        "selected_folder": [selected_folder],
        "scheme": [scheme],
        "group_id": [group_id],
        "feature_path": [feature_path],
        "test_loss": [round(test_loss, 6)],
        "precision@5": [round(precision5, 6)],
        "penalty@5": [round(penalty5, 6)],
        "comprehensive_score": [round(comprehensive_score, 6)],
        "timestamp": [pd.Timestamp.now().isoformat()]
    })
    # 追加写入CSV
    new_row.to_csv(FactoryConfig.INDEX_CSV_PATH, mode="a", header=False, index=False)
    logger.info(f"已记录组 {scheme}-{group_id} 到索引：综合得分 {comprehensive_score:.4f}")
# =============================================================================
# 工具函数：聚合特征提取与存储（51条×333列：item_id + 332维特征）
# =============================================================================
def extract_agg_features(selected_folder_path: str, candidate_group: List[str], 
                        scheme: str, datetime_str: str) -> pd.DataFrame:
    """
    提取51条数据（1条全量+50条候选），保持 item_id 位数一致性 
    返回：333列DataFrame（item_id + 332维特征）
    """
    # 1. 读取该文件夹的聚合特征文件 
    agg_feat_file = os.path.join(selected_folder_path, f"inter_feat_{datetime_str}.parquet")
    if not os.path.exists(agg_feat_file):
        raise FileNotFoundError(f"聚合特征文件不存在：{agg_feat_file}")
    agg_df = pd.read_parquet(agg_feat_file)
    agg_df = agg_df.rename(columns={"id": "item_id"})  # 统一列名 
    # 2. 提取候选组中每个文件的原始 item_id 字符串 [cite: 21]
    candidate_item_ids = []
    for file_path in candidate_group:
        file_name = os.path.basename(file_path).replace(".parquet", "")
        # 同样使用字符串截取逻辑，确保 1500 不会变成 150 [cite: 21]
        parts = file_name.split('-')
        item_id_str = "-".join(parts[-4:])
        candidate_item_ids.append(item_id_str)
    # 3. 匹配特征数据 [cite: 21, 22]
    candidate_items = []
    for item_id in candidate_item_ids:
        mask = agg_df["item_id"] == item_id
        if mask.any():
            row = agg_df[mask].iloc[0]
            item_id_val = row["item_id"]
            feat_row = row.drop("item_id").values.astype(np.float32) 
        else:
            logger.warning(f"聚合特征中未找到 item_id {item_id}，全零填充")
            item_id_val = item_id
            feat_row = np.zeros(329, dtype=np.float32)
        candidate_items.append((item_id_val, feat_row))
    # 4. 提取第1行作为全量数据 [cite: 23]
    total_row = agg_df.iloc[0]
    total_item_id = total_row["item_id"]
    total_feat = total_row.drop("item_id").values.astype(np.float32) 
    # 5. 组合并添加 One-Hot 辅助特征 [cite: 23, 24]
    one_hot = FactoryConfig.ONE_HOT_MAP[scheme]
    all_data = [(total_item_id, total_feat)] + candidate_items
    final_rows = []
    for item_id_val, core_feat in all_data:
        feat_with_aux = np.concatenate([core_feat, one_hot])  # 329+3=332维 [cite: 24]
        final_rows.append([item_id_val] + feat_with_aux.tolist())
    # 6. 构造最终 DataFrame [cite: 24]
    columns = ["item_id"] + [f"feat_{i}" for i in range(332)]
    feat_df = pd.DataFrame(final_rows, columns=columns)
    logger.info(f"聚合特征提取完成：{len(feat_df)}行，ID格式样例：{feat_df['item_id'].iloc[0]}")
    return feat_df
def save_agg_features(feat_df: pd.DataFrame, selected_folder: str, scheme: str, group_id: int) -> str:
    """保存归一化后的聚合特征到 factory_cache"""
    feat_df = normalize_factory_features(feat_df)
    file_name = f"{selected_folder}_{scheme}_{group_id}_set.parquet"
    save_path = os.path.join(FactoryConfig.FACTORY_CACHE_DIR, file_name)
    feat_df.to_parquet(save_path, index=False, compression="snappy")
    logger.info(f"归一化后的聚合特征保存路径：{save_path}")
    return save_path
# =============================================================================
# 工具函数：评估指标计算（Precision@5 + Over-prediction Penalty）
# =============================================================================

def calculate_prediction_penalty(top_k_df: pd.DataFrame) -> float:
    """
    计算惩罚性指标：当真实评分低于预测评分时，计算其差值的平均数。
    仅针对 Top-k 数据进行计算。
    """
    # 计算差值 (预测 - 真实)，只保留预测更高的部分（即差值 > 0）
    diff = top_k_df["prediction"] - top_k_df["true_rating"]
    penalty_values = diff[diff > 0]
    
    if penalty_values.empty:
        return 0.0
    return penalty_values.mean()

def calculate_precision_and_penalty(pred_df: pd.DataFrame, k=5, precision_threshold=9.0) -> Tuple[float, float]:
    """
    计算 Precision@k 和 Top-k 预测惩罚项。
    - Precision@k：Top-k 预测中，true_rating >= threshold 的占比。
    - Penalty：Top-k 中，(预测分 - 真实分) 的平均值（仅当预测 > 真实时计算）。
    """
    precision_list = []
    penalty_list = []
    
    for user_id, group in pred_df.groupby("user_id"):
        # 按预测分降序选 Top-k
        group_sorted = group.sort_values("prediction", ascending=False).head(k)
        
        # 1. 计算 Precision
        hits = (group_sorted["true_rating"] >= precision_threshold).sum()
        precision_list.append(hits / k if k > 0 else 0.0)
        
        # 2. 计算惩罚项 (Top-5 差值平均)
        penalty = calculate_prediction_penalty(group_sorted)
        penalty_list.append(penalty)
        
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_penalty = np.mean(penalty_list) if penalty_list else 0.0
    
    logger.info(f"Precision@{k}: {avg_precision:.4f}, Prediction Penalty: {avg_penalty:.4f}")
    return avg_precision, avg_penalty
# =============================================================================
# 工具函数：候选数据硬链接到copy文件夹（避免复制大文件）
# =============================================================================
def link_candidate_to_copy(candidate_group: List[str], copy_folder: str):
    """
    将候选组的文件硬链接到selected_folder的copy文件夹（训练集目录）
    先清空原有文件，再创建硬链接（节省磁盘空间）
    """
    # 清空copy文件夹
    if os.path.exists(copy_folder):
        for file in glob.glob(os.path.join(copy_folder, "*.parquet")):
            os.remove(file)
            logger.debug(f"删除旧链接：{os.path.basename(file)}")
    else:
        os.makedirs(copy_folder, exist_ok=True)
    # 为候选组中每个文件创建硬链接
    for src_file in candidate_group:
        dst_file = os.path.join(copy_folder, os.path.basename(src_file))
        if not os.path.exists(dst_file):
            os.link(src_file, dst_file)  # 硬链接（仅创建引用，不复制数据）
    logger.info(f"已创建 {len(candidate_group)} 个硬链接到训练集目录：{copy_folder}")
# =============================================================================
# 核心函数：单组候选数据的训练、评估与特征存储
# =============================================================================
def process_single_group(selected_folder_path: str, selected_folder_name: str,
                        candidate_group: List[str], scheme: str, group_id: int,
                        datetime_str: str, time_window: str) -> Tuple[float, float, float, float]:
    """
    处理单组候选数据的完整流程（适配代码1逻辑）：
    1. 硬链接 -> 2. 数据预处理 -> 3. 模型训练 -> 4. 保存模型 -> 5. 评估测试集
    """
    # 1. 硬链接候选数据到训练集目录
    copy_folder = os.path.join(selected_folder_path, "copy")
    link_candidate_to_copy(candidate_group, copy_folder)

    # 2. 环境准备
    slopeorpt = ModelConfig.SLOPE_OR_PT
    sub_dir = ModelConfig.SUB_DIR
    user_le, item_le = load_or_fit_label_encoders(slopeorpt, sub_dir)
    
    # 3. 数据加载
    train_inter_df = concat_parquet_files(copy_folder, files_per_batch=ModelConfig.FILES_PER_BATCH)
    if train_inter_df.empty:
        raise ValueError("训练集为空")

    # 4. 数据预处理
    inter_df, user_df, item_df, user_le, item_le, train_idx, test_idx = preprocess_data_deepcrossing(
        train_inter_df, slopeorpt, sub_dir
    )

    # 5. 构建生成器
    train_df = inter_df.iloc[train_idx]
    val_df = inter_df.iloc[test_idx] 
    train_generator = UpdatedDeepCrossingDataGenerator(train_df, user_df, item_df)
    val_generator = UpdatedDeepCrossingDataGenerator(val_df, user_df, item_df, batch_size=ModelConfig.EVALUATION_BATCH_SIZE)
    
    user_dim, item_dim, inter_dim = train_generator.get_input_dims()

    # ======================== 模型加载/构建逻辑 ========================
    model_name_tag = selected_folder_name.replace('Q9_n-sh000300-', '')
    model_save_dir = os.path.join(slopeorpt, "model", sub_dir)
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"model_{model_name_tag}.keras")

    if ModelConfig.MODEL_LOAD and os.path.exists(model_path):
        logger.info(f"正在加载预训练模型: {model_path}")
        model = load_model(model_path, custom_objects={'huber_loss': huber_loss})
        model.optimizer.learning_rate = ModelConfig.LEARNING_RATE
    else:
        if ModelConfig.MODEL_LOAD:
            logger.warning(f"未找到可加载模型 {model_path}，将构建新模型")
        model = build_model(user_dim, item_dim, inter_dim)
        logger.info("已构建新模型")
    # ======================================================================

    # 6. 训练模型
    csv_log_path = os.path.join(FactoryConfig.FACTORY_CACHE_DIR, f"train_log_{selected_folder_name}_{scheme}_{group_id}.csv")
    run_id = int(pd.Timestamp.now().timestamp())
    
    model, history = train_model(
        model, train_generator, val_generator, user_le, item_le,
        csv_log_path=csv_log_path, 
        run_id=run_id
    )

    # ======================== 关键修正：先保存模型！ ========================
    # 必须先保存，evaluate_test_folder 才能加载到刚才训练好的模型
    try:
        model.save(model_path)
        logger.info(f"模型已保存至: {model_path}")
    except Exception as e:
        logger.error(f"模型保存失败: {e}")
    # ======================================================================

    # 7. 核心修改：评估 test 文件夹
    # 此时加载的才是刚才训练保存的最新模型
    test_evaluation, pred_df = evaluate_test_folder(selected_folder_name)
    
    # 8. 计算指标
    test_loss = test_evaluation[0] if test_evaluation else 999.0
    precision5, penalty = calculate_precision_and_penalty(pred_df, k=5)
    
    loss_component = 1.0 / (1.0 + test_loss)
    comprehensive_score = (0.3 * loss_component + 0.5 * precision5 - 0.2 * penalty)

    # 9. 特征提取与索引记录
    feat_df = extract_agg_features(selected_folder_path, candidate_group, scheme, datetime_str)
    feature_path = save_agg_features(feat_df, selected_folder_name, scheme, group_id)
    write_to_index(selected_folder_name, scheme, group_id, feature_path, test_loss, precision5, penalty, comprehensive_score)

    # 10. 资源清理
    try:
        del model, train_generator, val_generator, history, pred_df, train_df, inter_df
        gc.collect()
        torch.cuda.empty_cache()
        keras.backend.clear_session()
        logger.info("资源清理完成")
    except Exception as e:
        logger.warning(f"资源清理异常: {e}")

    return test_loss, precision5, penalty, comprehensive_score
# =============================================================================
# 主流程：遍历所有候选组，断点续跑处理
# =============================================================================
def main():
    try:
        # 1. 初始化索引文件
        init_index_csv()
        # 2. 获取数据根路径和 selected_folder 列表
        slopeorpt = ModelConfig.SLOPE_OR_PT
        sub_dir = ModelConfig.SUB_DIR
        data_paths = get_data_paths(slopeorpt, sub_dir)
        inter_data_dir = data_paths["INTER_DATA_DIR"]
        logger.info(f"交互数据根目录：{inter_data_dir}")
        folder_list = [d for d in os.listdir(inter_data_dir) if os.path.isdir(os.path.join(inter_data_dir, d))]
        # 3. 循环处理多个索引
        for folder_index in FactoryConfig.SELECTED_FOLDER_INDEXES:
            if len(folder_list) <= folder_index:
                logger.warning(f"索引 {folder_index} 超出范围（总共有 {len(folder_list)} 个 selected_folder）")
                continue
            selected_folder_name = folder_list[folder_index]
            selected_folder_path = os.path.join(inter_data_dir, selected_folder_name)
            logger.info(f"\n{'='*80}")
            logger.info(f"当前处理 selected_folder：{selected_folder_name}")
            logger.info(f"文件夹路径：{selected_folder_path}")
            logger.info(f"{'='*80}\n")
            try:
                # 4. 提取时间信息
                datetime_str, time_window = extract_folder_datetime(selected_folder_name)
                logger.info(f"提取到时间窗口：{time_window}，datetime_str：{datetime_str}")
                # 5. 读取 interact_feat.parquet
                interact_feat_path = os.path.join(selected_folder_path, "interact_feat.parquet")
                if not os.path.exists(interact_feat_path):
                    logger.error(f"交互特征文件不存在：{interact_feat_path}")
                    continue
                inter_feat_df = pd.read_parquet(interact_feat_path)
                logger.info(f"加载 interact_feat.parquet：{len(inter_feat_df)} 条交互特征数据")
                # 6. 获取候选文件（基于 inter_feat_df 过滤）
                candidate_files = get_candidate_files(inter_data_dir, time_window, inter_feat_df)
                # 7. 生成所有候选组
                candidate_groups = generate_all_candidate_groups(inter_feat_df, candidate_files)
                if not candidate_groups:
                    logger.error("无有效候选组，跳过该文件夹")
                    continue
                total_groups = sum(len(groups) for groups in candidate_groups.values())
                logger.info(f"\n开始处理所有候选组（总计：{total_groups} 组）")
                # 8. 遍历所有组
                for scheme, groups in candidate_groups.items():
                    for group_id, candidate_group in enumerate(groups):
                        if is_group_processed(selected_folder_name, scheme, group_id):
                            logger.info(f"组 {scheme}-{group_id} 已处理，跳过")
                            continue
                        process_single_group(
                            selected_folder_path, selected_folder_name,
                            candidate_group, scheme, group_id,
                            datetime_str, time_window
                        )
            except Exception as e:
                logger.error(f"处理文件夹 {selected_folder_name} 失败：{e}")
                continue
        logger.info("所有组处理完成！")
    except Exception as e:
        logger.error(f"程序执行失败：{e}")
if __name__ == "__main__":
    main()
