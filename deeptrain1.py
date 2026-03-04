#代码1
import os
os.environ["KERAS_BACKEND"] = "torch"  # 必须在导入 keras 前设置
import random
# =============================================================================
# 全局随机种子设置（保持原有逻辑）
# =============================================================================
def set_all_seeds(seed=42):
    import numpy as _np
    import torch as _torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
#set_all_seeds(42)  # 在导入其他库之前调用
# =============================================================================
# 统一参数配置类（参考代码2重构）
# =============================================================================
class ModelConfig:
    """代码1参数配置类，所有参数集中管理"""
    # 基础路径配置
    SLOPE_OR_PT = 'pt'  # 'slope' 或 'pt'
    SUB_DIR = '0000'
    # 模型加载开关
    MODEL_LOAD = False  # True=检查并加载已有模型（重置学习率），False=重新构建模型
    # 数据加载配置
    FILES_PER_BATCH = 0  # 0=一次性加载所有文件，>0=每次加载指定数量的文件
    # 内存优化配置
    CLEAR_CUDA_CACHE_EACH_EPOCH = True   # 是否每个epoch清理显存
    SAVE_PREDICTION_HISTORY = False      # 是否保存预测历史（默认关闭防内存泄漏）
    EVALUATION_BATCH_SIZE = 80008        # 评估时的batch size（可单独设置）
    # 模型架构参数
    EMBEDDING_SIZE = 228
    DENSE_UNITS = [512, 256, 128, 64, 32]  # 保持原有参数，未添加16（避免改变运行结果）
    ATTENTION_UNITS = 300
    L2_REGULARIZATION = 0.001  # L2正则化系数
    # 训练参数
    TRAIN_BATCH_SIZE = 130008
    LEARNING_RATE = 0.003
    DROPOUT_RATE = 0.2
    OPTIMIZER_CHOICE = 2  # 1=Adam, 2=RMSprop
    EPOCHS = 35
    PATIENCE_EARLY_STOPPING = 9
    PATIENCE_LR_SCHEDULER = 3
    LR_FACTOR = 0.3
    # 损失函数参数（补充到配置类）
    HUBER_LOSS_DELTA = 20
    HUBER_NEGATIVE_WEIGHT = 1.5
    HUBER_POSITIVE_WEIGHT = 0.5
    # 优化器参数（补充到配置类）
    ADAM_BETA1 = 0.8
    ADAM_BETA2 = 0.99
    ADAM_EPSILON = 1e-6
    # 评估模式配置
    SCALER_SAVE_DIR = "scalers"
    LABEL_ENCODER_SAVE_NAME = {"user": "user_le.pkl", "item": "item_le.pkl"}
# =============================================================================
# 导入依赖库（保持原有顺序）
# =============================================================================
import torch
import sys
import glob
import gc
import logging
import joblib
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from keras.models import Model, load_model
from keras.layers import (
    Input, Flatten, Dense, Dropout, Multiply, Concatenate, BatchNormalization
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.optimizers import Adam, RMSprop
from keras.metrics import MeanAbsoluteError
from keras.losses import Huber, log_cosh
from keras.utils import Sequence
from keras.regularizers import l2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from keras.initializers import HeUniform
# ======================== 日志与字体配置（整合代码2） ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
def setup_chinese_font():
    """自动查找并设置中文字体"""
    # Windows 常见中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    # 查找系统中已安装的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = None
    for font in chinese_fonts:
        if font in available_fonts:
            font_found = font
            break
    if font_found:
        plt.rcParams['font.sans-serif'] = [font_found]
        logger.info(f"已设置中文字体: {font_found}")
    else:
        logger.warning("未找到常见中文字体，图表可能无法正确显示中文")
        logger.warning("建议安装 SimHei 或 Microsoft YaHei 字体")
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
# 初始化中文字体
setup_chinese_font()
# ======================== 自定义显存清理回调（代码1） ========================
class ClearCacheCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if ModelConfig.CLEAR_CUDA_CACHE_EACH_EPOCH:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(f"\nEpoch {epoch+1}: 清理显存完成")
# ======================== 新增：CSV日志回调（适配工厂模式） ========================
class CSVHistoryLRCallback(Callback):
    """
    将每个 epoch 的训练数据写入 CSV 文件
    """
    def __init__(self, csv_path, run_id=None):
        super().__init__()
        self.csv_path = csv_path
        self.run_id = run_id if run_id is not None else int(pd.Timestamp.now().timestamp())
        self._header_written = os.path.exists(csv_path)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # 获取当前学习率
        try:
            lr = float(self.model.optimizer.learning_rate)
        except:
            lr = float('nan')
            
        row = {
            'run_id': self.run_id,
            'epoch': epoch + 1,
            'learning_rate': lr,
            'timestamp': pd.Timestamp.now().isoformat(),
            'loss': logs.get('loss', float('nan')),
            'val_loss': logs.get('val_loss', float('nan')),
            'mean_absolute_error': logs.get('mean_absolute_error', float('nan')),
            'val_mean_absolute_error': logs.get('val_mean_absolute_error', float('nan'))
        }
        
        df = pd.DataFrame([row])
        
        # 写入 CSV
        if not self._header_written:
            df.to_csv(self.csv_path, index=False, mode='w')
            self._header_written = True
        else:
            df.to_csv(self.csv_path, index=False, mode='a', header=False)
            
        logger.info(f"Epoch {epoch+1} - Run ID: {self.run_id}, LR: {lr:.6f}, Loss: {logs.get('loss'):.4f}, Val Loss: {logs.get('val_loss', 'N/A')}")

# ======================== 损失函数（代码1，适配ModelConfig） ========================
def huber_loss(y_true, y_pred):
    delta = ModelConfig.HUBER_LOSS_DELTA
    error = y_true - y_pred
    abs_error = keras.ops.abs(error)
    quadratic = keras.ops.minimum(abs_error, delta)
    linear = abs_error - quadratic
    base_loss = 0.5 * keras.ops.square(quadratic) + delta * linear
    weights = keras.ops.where(
        error < 0,
        ModelConfig.HUBER_NEGATIVE_WEIGHT,
        ModelConfig.HUBER_POSITIVE_WEIGHT
    )
    return keras.ops.mean(base_loss * weights)
# ======================== 工具函数（整合代码1+代码2，适配ModelConfig） ========================
def get_data_paths(slopeorpt=None, sub_dir=None):
    """统一数据路径获取（兼容代码1和代码2）"""
    # 优先使用传入参数，无则使用配置类默认值
    slopeorpt = slopeorpt or ModelConfig.SLOPE_OR_PT
    sub_dir = sub_dir or ModelConfig.SUB_DIR
    return {
        "INTER_DATA_DIR": f"{slopeorpt}/inter_csv1/{sub_dir}",
        "NEW_INTER_DATA_DIR": f"{slopeorpt}/inter_csv1/{sub_dir}/new",
        "USER_DATA_FILE": f"{slopeorpt}/user1.parquet",
        "ITEM_DATA_FILE": f"{slopeorpt}/item1.parquet",
        "NEW_USER_DATA_FILE": f"{slopeorpt}/neuser1.parquet",
        "NEW_ITEM_DATA_FILE": f"{slopeorpt}/neitem1.parquet"
    }
def load_label_encoders(slopeorpt=None, sub_dir=None):
    """代码1原函数（兼容评估模式，适配ModelConfig）"""
    slopeorpt = slopeorpt or ModelConfig.SLOPE_OR_PT
    sub_dir = sub_dir or ModelConfig.SUB_DIR
    user_le_path = os.path.join(slopeorpt, ModelConfig.LABEL_ENCODER_SAVE_NAME["user"])
    item_le_path = os.path.join(slopeorpt, ModelConfig.LABEL_ENCODER_SAVE_NAME["item"])
    if os.path.exists(user_le_path) and os.path.exists(item_le_path):
        user_le = joblib.load(user_le_path)
        item_le = joblib.load(item_le_path)
    else:
        data_paths = get_data_paths(slopeorpt, sub_dir) 
        user_df = pd.read_parquet(data_paths["USER_DATA_FILE"])
        item_df = pd.read_parquet(data_paths["ITEM_DATA_FILE"])
        if os.path.exists(data_paths["NEW_USER_DATA_FILE"]):
            new_user_df = pd.read_parquet(data_paths["NEW_USER_DATA_FILE"])
            user_df = pd.concat([user_df, new_user_df], ignore_index=True)
        if os.path.exists(data_paths["NEW_ITEM_DATA_FILE"]):
            new_item_df = pd.read_parquet(data_paths["NEW_ITEM_DATA_FILE"])
            item_df = pd.concat([item_df, new_item_df], ignore_index=True)
        user_le = LabelEncoder()
        item_le = LabelEncoder()
        user_le.fit(user_df['user_id'])
        item_le.fit(item_df['item_id'])
        joblib.dump(user_le, user_le_path)
        joblib.dump(item_le, item_le_path)
    return user_le, item_le
def load_or_fit_label_encoders(slopeorpt=None, sub_dir=None):
    """评估模式专用（确保编码器已存在，适配ModelConfig）"""
    slopeorpt = slopeorpt or ModelConfig.SLOPE_OR_PT
    sub_dir = sub_dir or ModelConfig.SUB_DIR
    user_le_path = os.path.join(slopeorpt, ModelConfig.LABEL_ENCODER_SAVE_NAME["user"])
    item_le_path = os.path.join(slopeorpt, ModelConfig.LABEL_ENCODER_SAVE_NAME["item"])
    if not (os.path.exists(user_le_path) and os.path.exists(item_le_path)):
        logger.error(f"未找到已保存的 LabelEncoder，请先运行训练: {user_le_path}, {item_le_path}")
        sys.exit(1)
    user_le = joblib.load(user_le_path)
    item_le = joblib.load(item_le_path)
    logger.info("已加载已有的 LabelEncoder")
    return user_le, item_le
def load_interaction_data(slopeorpt=None, sub_dir=None, selected_folder=None):
    """代码1原函数：加载copy文件夹数据，适配ModelConfig"""
    slopeorpt = slopeorpt or ModelConfig.SLOPE_OR_PT
    sub_dir = sub_dir or ModelConfig.SUB_DIR
    data_paths = get_data_paths(slopeorpt, sub_dir)
    folder_path = os.path.join(data_paths["INTER_DATA_DIR"], selected_folder)
    folder_path = os.path.join(folder_path, 'copy')
    # 获取所有parquet文件
    parquet_files = glob.glob(f"{folder_path}/*.parquet")
    all_parquet_files = parquet_files
    # 根据FILES_PER_BATCH参数确定批次大小
    batch_size = ModelConfig.FILES_PER_BATCH if ModelConfig.FILES_PER_BATCH > 0 else len(parquet_files)
    print(f"Batch size (files): {batch_size}")
    # 计算总行数
    total_rows = 0
    for parquet_file in parquet_files:
        df_temp = pd.read_parquet(parquet_file)
        total_rows += len(df_temp)
    # 分批加载
    for i in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[i:i + batch_size]
        inter_df = pd.DataFrame()
        for parquet_file in batch_files:
            chunk = pd.read_parquet(parquet_file)
            inter_df = pd.concat([inter_df, chunk])
        yield inter_df, total_rows
def concat_parquet_files(folder_path, files_per_batch=None):
    """代码2函数：加载test文件夹数据，适配ModelConfig"""
    files_per_batch = files_per_batch or ModelConfig.FILES_PER_BATCH
    parquet_files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
    if not parquet_files:
        return pd.DataFrame()
    if files_per_batch == 0:
        list_dfs = [pd.read_parquet(p) for p in parquet_files]
        return pd.concat(list_dfs, ignore_index=True)
    list_dfs = []
    for start in range(0, len(parquet_files), files_per_batch):
        batch = parquet_files[start:start + files_per_batch]
        for p in batch:
            list_dfs.append(pd.read_parquet(p))
    return pd.concat(list_dfs, ignore_index=True)
def parse_feature(x):
    """代码1原函数：解析特征"""
    if isinstance(x, str):
        return np.fromstring(x.strip("[]"), sep=' ', dtype=np.float32)
    return x
def fast_parse_feature_series(series):
    """代码2函数：快速解析特征序列"""
    def _parse_one(x):
        if isinstance(x, (np.ndarray, list)):
            return np.array(x, dtype=np.float32)
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            if "," in s:
                parts = s.replace(",", " ").split()
            else:
                parts = s.split()
            return np.array([float(v) for v in parts], dtype=np.float32)
        return np.array([], dtype=np.float32)
    return series.map(_parse_one)
def batch_normalize_features(df, feature_columns, scalers_dict, scope_key=None):
    """统一批次归一化函数（兼容代码1和代码2）"""
    if scope_key is None:
        # 代码1原逻辑
        for feature in feature_columns:
            if feature in scalers_dict:
                scaler = scalers_dict[feature]
                feature_data = np.stack(df[feature].values)
                if feature_data.shape[1] == scaler.n_features_in_:
                    transformed = scaler.transform(feature_data)
                else:
                    transformed = np.zeros((len(df), scaler.n_features_in_))
                df[feature] = list(transformed)
    else:
        # 代码2评估逻辑
        for feature in feature_columns:
            scaler = scalers_dict[scope_key].get(feature, None)
            if scaler is None:
                continue
            feature_data = np.stack(df[feature].values)
            if feature_data.shape[1] == scaler.n_features_in_:
                transformed = scaler.transform(feature_data)
            else:
                transformed = np.zeros((len(df), scaler.n_features_in_), dtype=np.float32)
            df[feature] = list(transformed)
    return df
def preprocess_data_deepcrossing(inter_df, slopeorpt=None, sub_dir=None):
    """代码1原函数：预处理copy文件夹数据，适配ModelConfig"""
    slopeorpt = slopeorpt or ModelConfig.SLOPE_OR_PT
    sub_dir = sub_dir or ModelConfig.SUB_DIR
    data_paths = get_data_paths(slopeorpt, sub_dir)
    user_df = pd.read_parquet(data_paths["USER_DATA_FILE"])
    item_df = pd.read_parquet(data_paths["ITEM_DATA_FILE"])
    # 标签编码
    user_le = LabelEncoder()
    item_le = LabelEncoder()
    user_df['user_id'] = user_le.fit_transform(user_df['user_id'])
    item_df['item_id'] = item_le.fit_transform(item_df['item_id'])
    inter_df['user_id'] = user_le.transform(inter_df['user_id'])
    inter_df['item_id'] = item_le.transform(inter_df['item_id'])
    # 分割数据集
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=32)
    train_idx, test_idx = next(gss.split(inter_df, groups=inter_df['user_id']))
    train_df = inter_df.iloc[train_idx]
    test_df = inter_df.iloc[test_idx]
    # 获取训练集的用户和商品范围
    train_user_ids = train_df['user_id'].unique()
    train_item_ids = train_df['item_id'].unique()
    # 用户特征归一化（仅使用训练用户数据）
    user_features = user_df.columns[1:]
    for col in user_features:
        user_df[col] = user_df[col].apply(parse_feature)
    # 预解析商品特征
    item_features = item_df.columns[1:]
    for col in item_features:
        item_df[col] = item_df[col].apply(parse_feature)
    # 用户特征归一化（批量处理）
    user_scalers = {}
    for feature in user_features:
        # 提取训练用户的特征数据
        train_mask = user_df['user_id'].isin(train_user_ids)
        train_data = np.stack(user_df.loc[train_mask, feature].values)
        if len(train_data) > 0:
            scaler = StandardScaler()
            scaler.fit(train_data)
            user_scalers[feature] = scaler
    # 批量归一化所有用户特征
    user_df = batch_normalize_features(user_df, user_features, user_scalers)
    # 商品特征归一化（批量处理）
    item_scalers = {}
    for feature in item_features:
        # 提取训练商品的特征数据
        train_mask = item_df['item_id'].isin(train_item_ids)
        train_data = np.stack(item_df.loc[train_mask, feature].values)
        if len(train_data) > 0:
            scaler = StandardScaler()
            scaler.fit(train_data)
            item_scalers[feature] = scaler
    # 批量归一化所有商品特征
    item_df = batch_normalize_features(item_df, item_features, item_scalers)
    # 交互特征归一化
    interaction_features = [col for col in inter_df.columns if col not in ['user_id', 'item_id', 'rating']]
    if interaction_features:
        inter_scaler = StandardScaler()
        inter_scaler.fit(train_df[interaction_features])
        train_df[interaction_features] = inter_scaler.transform(train_df[interaction_features])
        test_df[interaction_features] = inter_scaler.transform(test_df[interaction_features])
    else:
        inter_scaler = None
        print("警告: 未找到交互特征")
    # 合并处理后的数据
    inter_df = pd.concat([train_df, test_df])
    # 保存归一化器
    scalers_dir = os.path.join(slopeorpt, ModelConfig.SCALER_SAVE_DIR, sub_dir)
    os.makedirs(scalers_dir, exist_ok=True)
    # 保存用户/商品/交互特征归一化器（兼容代码2）
    scalers_dict = {
        "user": user_scalers,
        "item": item_scalers,
        "inter": inter_scaler,
        "inter_features": interaction_features
    }
    joblib.dump(user_scalers, os.path.join(scalers_dir, "user_scalers.pkl"))
    joblib.dump(item_scalers, os.path.join(scalers_dir, "item_scalers.pkl"))
    if inter_scaler is not None:
        joblib.dump(inter_scaler, os.path.join(scalers_dir, "inter_scaler.pkl"))
        joblib.dump(interaction_features, os.path.join(scalers_dir, "inter_feature_names.pkl"))
        joblib.dump(scalers_dict, os.path.join(scalers_dir, "scalers_dict.pkl"))  # 代码2需要的整合scaler
        print(f"保存了 {len(interaction_features)} 个交互特征名称")
    else:
        print("未保存交互特征归一化器，因为没有交互特征")
    # 保存标签编码器
    joblib.dump(user_le, os.path.join(slopeorpt, ModelConfig.LABEL_ENCODER_SAVE_NAME["user"]))
    joblib.dump(item_le, os.path.join(slopeorpt, ModelConfig.LABEL_ENCODER_SAVE_NAME["item"]))
    # 打印数据统计信息
    print("\n=== 数据预处理统计 ===")
    print(f"总交互数: {len(inter_df)}")
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"用户数: {len(user_le.classes_)}")
    print(f"商品数: {len(item_le.classes_)}")
    print(f"用户特征数: {len(user_features)}")
    print(f"商品特征数: {len(item_features)}")
    print(f"交互特征数: {len(interaction_features) if interaction_features else 0}")
    return inter_df, user_df, item_df, user_le, item_le, train_idx, test_idx
# ======================== 数据生成器（整合代码1+代码2，适配ModelConfig） ========================
class UpdatedDeepCrossingDataGenerator(Sequence):
    def __init__(self, inter_df, user_df, item_df, batch_size=None):
        self.batch_size = batch_size or ModelConfig.TRAIN_BATCH_SIZE
        self.inter_df = inter_df.reset_index(drop=True)
        self.user_df = user_df
        self.item_df = item_df
        # 自动推断特征列（去掉 id 列）
        self.user_features = [c for c in user_df.columns if c != 'user_id']
        self.item_features = [c for c in item_df.columns if c != 'item_id']
        self.interaction_features = [c for c in inter_df.columns if c not in ['user_id', 'item_id', 'rating']]
        # 存储用户ID和物品ID用于评估
        self.user_ids = self.inter_df['user_id'].values if not self.inter_df.empty else np.array([], dtype=np.int32)
        self.item_ids = self.inter_df['item_id'].values if not self.inter_df.empty else np.array([], dtype=np.int32)
        # 构建特征字典（保证每个 user_id/item_id 对应一个拼接后的向量）
        self.user_feature_dict = {}
        if len(user_df) > 0:
            for user_id, row in user_df.set_index('user_id').iterrows():
                self.user_feature_dict[user_id] = np.hstack([row[feat] for feat in self.user_features])
        self.item_feature_dict = {}
        if len(item_df) > 0:
            for item_id, row in item_df.set_index('item_id').iterrows():
                self.item_feature_dict[item_id] = np.hstack([row[feat] for feat in self.item_features])
        # 记录维度，方便调试
        self.user_dim = len(next(iter(self.user_feature_dict.values()))) if len(self.user_feature_dict) > 0 else 0
        self.item_dim = len(next(iter(self.item_feature_dict.values()))) if len(self.item_feature_dict) > 0 else 0
        self.inter_dim = len(self.interaction_features)
        print(f"[DataGenerator] user_dim={self.user_dim}, item_dim={self.item_dim}, inter_dim={self.inter_dim}")
    def __len__(self):
        if self.inter_df.empty:
            return 0
        return int(np.ceil(len(self.inter_df) / float(self.batch_size)))
    def __getitem__(self, idx):
        batch_inter = self.inter_df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        user_ids = batch_inter['user_id'].values
        item_ids = batch_inter['item_id'].values
        # 提取特征（兼容代码2的空值处理）
        user_features = np.array([
            self.user_feature_dict.get(uid, np.zeros(self.user_dim, dtype=np.float32)) 
            for uid in user_ids
        ], dtype=np.float32)
        item_features = np.array([
            self.item_feature_dict.get(iid, np.zeros(self.item_dim, dtype=np.float32)) 
            for iid in item_ids
        ], dtype=np.float32)
        inter_features = batch_inter[self.interaction_features].values.astype(np.float32) if len(self.interaction_features) > 0 else np.zeros((len(batch_inter), 0), dtype=np.float32)
        ratings = batch_inter['rating'].values.astype(np.float32)
        return (user_features, item_features, inter_features), ratings
    def get_input_dims(self):
        """代码2需要的维度获取函数"""
        return self.user_dim, self.item_dim, self.inter_dim
# ======================== 模型构建与训练（代码1，适配ModelConfig） ========================
def build_model(user_dim, item_dim, inter_dim):

    # 定义输入层
    user_features_input = Input(shape=(user_dim,), name='user_features')
    item_features_input = Input(shape=(item_dim,), name='item_features')
    interaction_features_input = Input(shape=(inter_dim,), name='interaction_features')
    # 特征处理
    user_embedding = Dense(ModelConfig.EMBEDDING_SIZE, activation='relu')(user_features_input)
    user_embedding = Dense(ModelConfig.EMBEDDING_SIZE, activation='relu')(user_embedding)
    item_embedding = Dense(ModelConfig.EMBEDDING_SIZE, activation='relu')(item_features_input)
    item_embedding = Dense(ModelConfig.EMBEDDING_SIZE, activation='relu')(item_embedding)
    interaction_embedding = Dense(ModelConfig.EMBEDDING_SIZE, activation='relu')(interaction_features_input)
    interaction_embedding = Dense(ModelConfig.EMBEDDING_SIZE, activation='relu')(interaction_embedding)
    # 注意力机制
    user_combined = Concatenate(name='user_combined_embedding')([Flatten()(user_embedding), user_embedding, interaction_embedding])
    item_combined = Concatenate(name='item_combined_embedding')([Flatten()(item_embedding), item_embedding, interaction_embedding])
    attention_input = Multiply(name='attention_multiply')([user_combined, item_combined])
    attention_vector = Dense(ModelConfig.ATTENTION_UNITS, activation='linear', name='attention_dense_linear')(attention_input)
    attention_vector = Dense(ModelConfig.ATTENTION_UNITS, activation='sigmoid', name='attention_dense_sigmoid')(attention_vector)
    # 合并特征
    mf_vector = Concatenate()([user_embedding, item_embedding, interaction_embedding, attention_vector])
    mlp_vector = Concatenate()([user_features_input, item_features_input, interaction_features_input])
    # MLP层（使用统一的L2正则化参数）
    for units in ModelConfig.DENSE_UNITS:
        mlp_vector = Dense(units, activation='relu',kernel_initializer=HeUniform(),
                          kernel_regularizer=l2(ModelConfig.L2_REGULARIZATION))(mlp_vector)
        #mlp_vector = Dense(units, activation='relu', kernel_regularizer=l2(ModelConfig.L2_REGULARIZATION))(mlp_vector)
        mlp_vector = BatchNormalization()(mlp_vector)
        mlp_vector = Dropout(ModelConfig.DROPOUT_RATE)(mlp_vector)
    # 最终预测
    concatenated = Concatenate()([mf_vector, mlp_vector])
    predictions = Dense(1, activation='linear')(concatenated)
    # 定义模型
    model = Model(inputs=[user_features_input, item_features_input, interaction_features_input], outputs=predictions)
    return model
def train_model(model, train_generator, val_generator, user_le, item_le, 
                csv_log_path=None, run_id=None):
    """
    代码1训练函数（已适配工厂模式参数）：
    - 新增 csv_log_path, run_id 参数
    - 新增 val_generator 参数，若提供则优先作为验证集使用
    """
    # 1. 确定验证数据和监控指标
    # 如果传入了独立的验证集(val_generator)，则使用它；否则回退到代码1原有逻辑(使用test_generator作为验证集)

    monitor_metric = 'val_loss' if val_generator is not None else 'loss'
    
    # 2. 选择优化器
    if ModelConfig.OPTIMIZER_CHOICE == 1:
        optimizer = Adam(
            learning_rate=ModelConfig.LEARNING_RATE, 
            beta_1=ModelConfig.ADAM_BETA1, 
            beta_2=ModelConfig.ADAM_BETA2, 
            epsilon=ModelConfig.ADAM_EPSILON
        )
    elif ModelConfig.OPTIMIZER_CHOICE == 2:
        optimizer = RMSprop(learning_rate=ModelConfig.LEARNING_RATE)
    else:
        raise ValueError("Invalid optimizer choice. Please choose 1 for Adam or 2 for RMSprop.")

    # 3. 学习率调度器 (使用动态确定的监控指标)
    lr_scheduler = ReduceLROnPlateau(
        monitor=monitor_metric, 
        factor=ModelConfig.LR_FACTOR, 
        patience=ModelConfig.PATIENCE_LR_SCHEDULER
    )

    # 使用自定义 huber_loss
    model.compile(optimizer=optimizer, loss=huber_loss, metrics=[MeanAbsoluteError()])

    # 4. 构建回调列表
    callbacks = [
        EarlyStopping(
            monitor=monitor_metric, 
            patience=ModelConfig.PATIENCE_EARLY_STOPPING, 
            restore_best_weights=True
        ),
        lr_scheduler
    ]

    # 根据参数添加显存清理回调
    if ModelConfig.CLEAR_CUDA_CACHE_EACH_EPOCH:
        callbacks.append(ClearCacheCallback())
    
    # 新增：如果指定了 CSV 日志路径，添加回调
    if csv_log_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
        callbacks.append(CSVHistoryLRCallback(csv_path=csv_log_path, run_id=run_id))

    # 5. 训练模型
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=ModelConfig.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

# ======================== 评估函数（整合代码1+代码2，适配ModelConfig） ========================
def evaluate_predictions(test_generator, model, user_le, item_le):
    """统一评估预测函数（兼容代码1和代码2）"""
    if len(test_generator) == 0:
        return pd.DataFrame(columns=['user_id', 'item_id', 'prediction', 'true_rating'])
    logger.info("开始生成预测...")
    # 使用单独的评估batch size
    predictions = model.predict(test_generator, batch_size=ModelConfig.EVALUATION_BATCH_SIZE, verbose=1)
    start_idx = 0
    user_ids = []
    item_ids = []
    true_ratings = []
    for i in range(len(test_generator)):
        batch = test_generator[i]
        batch_size = len(batch[1])
        end_idx = start_idx + batch_size
        # 从生成器的属性中提取当前批次的ID
        user_ids.extend(test_generator.user_ids[start_idx:end_idx])
        item_ids.extend(test_generator.item_ids[start_idx:end_idx])
        # 获取当前批次的真实评分
        true_ratings.extend(batch[1])
        start_idx = end_idx
    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    true_ratings = np.array(true_ratings)
    df = pd.DataFrame({
        'user_id': user_le.inverse_transform(user_ids),
        'item_id': item_le.inverse_transform(item_ids),
        'prediction': predictions.flatten(),
        'true_rating': true_ratings
    })
    logger.info(f"预测完成，共生成 {len(df)} 条结果")
    return df
def plot_top_n_predictions(df, n=5):
    """统一可视化函数（兼容代码1和代码2）"""
    if df.empty:
        logger.info("没有预测结果可视化（df 为空）。")
        return
    df_sorted = df.sort_values(by=['user_id', 'prediction'], ascending=[True, False])
    top_n_df = df_sorted.groupby('user_id').head(n)
    for user_id, group in top_n_df.groupby('user_id'):
        print(f"User ID: {user_id}")
        for i, row in group.iterrows():
            print(f"  Item ID: {row['item_id']}, Predicted Rating: {row['prediction']:.4f}, True Rating: {row['true_rating']:.4f}")
        print()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df['user_id'].unique())))
    color_dict = dict(zip(df['user_id'].unique(), colors))
    plt.figure(figsize=(12, 10))
    for user_id in df['user_id'].unique():
        user_df = top_n_df[top_n_df['user_id'] == user_id]
        plt.scatter(user_df['true_rating'], user_df['prediction'], alpha=0.5, color=color_dict[user_id], label=user_id)
    plt.title('Actual vs Predicted Ratings for Top N Items')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.legend(title='User ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
def evaluate_test_folder(selected_folder):
    """代码2核心逻辑：评估test文件夹数据，适配ModelConfig"""
    logger.info("=" * 60)
    logger.info("开始评估 test 文件夹数据")
    logger.info("=" * 60)
    evaluation = None
    df = pd.DataFrame()
    # 1. 加载 LabelEncoder
    logger.info("步骤1: 加载 LabelEncoder...")
    user_le, item_le = load_or_fit_label_encoders()
    # 2. 加载测试数据
    logger.info("步骤2: 加载test文件夹数据...")
    data_paths = get_data_paths()
    folder_base = os.path.join(data_paths["INTER_DATA_DIR"], selected_folder)
    test_folder = os.path.join(folder_base, 'test')
    if not os.path.exists(test_folder):
        logger.error(f"测试文件夹不存在: {test_folder}")
        return
    test_inter_df = concat_parquet_files(test_folder)
    if test_inter_df.empty:
        logger.error("test文件夹数据为空，无法评估")
        return
    logger.info(f"成功加载test数据，共 {len(test_inter_df)} 条交互记录")
    # 3. 加载用户和物品特征数据
    logger.info("步骤3: 加载用户和物品特征数据...")
    user_df = pd.read_parquet(data_paths["USER_DATA_FILE"])
    item_df = pd.read_parquet(data_paths["ITEM_DATA_FILE"])
    # 4. 标签编码
    logger.info("步骤4: 对test数据进行标签编码...")
    try:
        user_df['user_id'] = user_le.transform(user_df['user_id'])
        item_df['item_id'] = item_le.transform(item_df['item_id'])
        test_inter_df['user_id'] = user_le.transform(test_inter_df['user_id'])
        test_inter_df['item_id'] = item_le.transform(test_inter_df['item_id'])
    except ValueError as e:
        logger.error(f"标签编码失败，test集包含未在训练集中出现的ID: {e}")
        return
    # 5. 解析特征字符串
    logger.info("步骤5: 解析特征字符串...")
    user_features = [c for c in user_df.columns if c != 'user_id']
    item_features = [c for c in item_df.columns if c != 'item_id']
    for col in user_features:
        user_df[col] = fast_parse_feature_series(user_df[col])
    for col in item_features:
        item_df[col] = fast_parse_feature_series(item_df[col])
    # 6. 加载已保存的 scaler
    logger.info("步骤6: 加载已保存的 scaler...")
    scalers_dir = os.path.join(ModelConfig.SLOPE_OR_PT, ModelConfig.SCALER_SAVE_DIR, ModelConfig.SUB_DIR)
    scalers_path = os.path.join(scalers_dir, "scalers_dict.pkl")
    if not os.path.exists(scalers_path):
        logger.error(f"未找到已保存的 scaler 文件: {scalers_path}")
        return
    scalers_dict = joblib.load(scalers_path)
    logger.info("成功加载 scaler")
    # 7. 应用归一化
    logger.info("步骤7: 应用特征归一化...")
    if scalers_dict.get('user'):
        user_df = batch_normalize_features(user_df, user_features, scalers_dict, 'user')
    if scalers_dict.get('item'):
        item_df = batch_normalize_features(item_df, item_features, scalers_dict, 'item')
    interaction_features = scalers_dict.get('inter_features', [])
    if interaction_features and scalers_dict.get('inter') is not None:
        missing_features = set(interaction_features) - set(test_inter_df.columns)
        if missing_features:
            logger.warning(f"test数据缺少以下交互特征: {missing_features}")
            for f in missing_features:
                test_inter_df[f] = 0.0
        test_inter_df[interaction_features] = scalers_dict['inter'].transform(test_inter_df[interaction_features])
        logger.info(f"已归一化 {len(interaction_features)} 个交互特征")
    else:
        logger.warning("未找到交互特征 scaler，跳过交互特征归一化")
    # 8. 创建test数据生成器
    logger.info("步骤8: 创建test数据生成器...")
    test_generator = UpdatedDeepCrossingDataGenerator(test_inter_df, user_df, item_df, ModelConfig.EVALUATION_BATCH_SIZE)
    user_dim, item_dim, inter_dim = test_generator.get_input_dims()
    logger.info(f"test数据维度 - user: {user_dim}, item: {item_dim}, inter: {inter_dim}")
    # 9. 加载模型（使用训练好的模型）
    logger.info("步骤9: 加载已训练模型...")
    model_name = selected_folder.replace('Q9_n-sh000300-', '')
    model_save_dir = f"{ModelConfig.SLOPE_OR_PT}/model/{ModelConfig.SUB_DIR}"
    model_path = f"{model_save_dir}/model_{model_name}.keras"
    if not os.path.exists(model_path):
        logger.error(f"未找到模型文件: {model_path}")
        return
    try:
        model = load_model(model_path, custom_objects={'huber_loss': huber_loss})
        logger.info(f"成功加载模型: {model_path}")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return
    # 10. 执行评估
    logger.info("步骤10: 执行test集评估...")
    if len(test_generator) > 0:
        evaluation = model.evaluate(test_generator, verbose=1)
        logger.info(f"test集评估结果 - Loss: {evaluation[0]:.6f}, MAE: {evaluation[1]:.6f}")
    else:
        logger.warning("test数据生成器为空，无法评估")
        return
    # 11. 生成预测并可视化
    logger.info("步骤11: 生成test集预测结果并可视化...")
    df = evaluate_predictions(test_generator, model, user_le, item_le)
    if not df.empty:
        logger.info(f"test集预测完成，共生成 {len(df)} 条结果")
        print("\n" + "=" * 60)
        print("test集 - 每个用户 Top-5 预测结果:")
        print("=" * 60)
        plot_top_n_predictions(df, n=5)
    else:
        logger.warning("test集预测结果为空")
    # 12. 清理资源
    logger.info("步骤12: 清理test评估资源...")
    try:
        del model, test_generator, user_df, item_df, test_inter_df
    except Exception:
        pass
    gc.collect()
    keras.backend.clear_session()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    logger.info("=" * 60)
    logger.info("test集评估完成 ✅")
    logger.info("=" * 60)
    return evaluation, df
# ======================== 主函数（整合训练+test评估，适配ModelConfig） ========================
def main():
    # 加载标签编码器
    user_le, item_le = load_label_encoders()
    slopeorpt = ModelConfig.SLOPE_OR_PT
    # 获取文件夹列表
    data_paths = get_data_paths()
    folder_list = [d for d in os.listdir(data_paths["INTER_DATA_DIR"]) if os.path.isdir(os.path.join(data_paths["INTER_DATA_DIR"], d))]
    if not folder_list:
        print("No folders found in INTER_DATA_DIR.")
        sys.exit(1)
    else:
        print("Available folders and their indices:")
        for i, folder in enumerate(folder_list):
            print(f"{i}: {folder}")
        selected_folder = folder_list[33]  # 选择文件夹
        print(f"Selected folder: {selected_folder}")
    # 逐批次加载交互数据（copy文件夹）
    for inter_df, total_rows in load_interaction_data(selected_folder=selected_folder):
        print(f"Total rows: {total_rows}")
        # 数据预处理（copy文件夹）
        inter_df, user_df, item_df, user_le, item_le, train_idx, test_idx = preprocess_data_deepcrossing(inter_df)
        train_df = inter_df.iloc[train_idx]
        test_df = inter_df.iloc[test_idx]
        # 创建数据生成器（copy文件夹）
        train_generator = UpdatedDeepCrossingDataGenerator(train_df, user_df, item_df)
        val_generator = UpdatedDeepCrossingDataGenerator(test_df, user_df, item_df)
        # 构建/加载模型（新增MODEL_LOAD逻辑）
        user_dim, item_dim, inter_dim = train_generator.get_input_dims()
        model_name = selected_folder.replace('Q9_n-sh000300-', '')
        model_save_dir = f"{ModelConfig.SLOPE_OR_PT}/model/{ModelConfig.SUB_DIR}"
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = f"{model_save_dir}/model_{model_name}.keras"
        if ModelConfig.MODEL_LOAD:
            # 检查模型是否存在，存在则加载并重置学习率
            if os.path.exists(model_path):
                print(f"MODEL_LOAD=True，加载已有模型: {model_path}")
                model = load_model(model_path, custom_objects={'huber_loss': huber_loss})
                # 重置学习率（重新编译模型）
                if ModelConfig.OPTIMIZER_CHOICE == 1:
                    optimizer = Adam(
                        learning_rate=ModelConfig.LEARNING_RATE, 
                        beta_1=ModelConfig.ADAM_BETA1, 
                        beta_2=ModelConfig.ADAM_BETA2, 
                        epsilon=ModelConfig.ADAM_EPSILON
                    )
                elif ModelConfig.OPTIMIZER_CHOICE == 2:
                    optimizer = RMSprop(learning_rate=ModelConfig.LEARNING_RATE)
                else:
                    raise ValueError("Invalid optimizer choice")
                model.compile(optimizer=optimizer, loss=huber_loss, metrics=[MeanAbsoluteError()])
                print(f"模型学习率已重置为: {ModelConfig.LEARNING_RATE}")
            else:
                print(f"MODEL_LOAD=True 但未找到模型 {model_path}，重新构建模型")
                model = build_model(user_dim, item_dim, inter_dim)
        else:
            # 不加载，重新构建模型
            print("MODEL_LOAD=False，重新构建模型")
            model = build_model(user_dim, item_dim, inter_dim)
        csv_log_path = os.path.join(slopeorpt, f"training_log_mode{model_name}.csv")
        run_id = int(pd.Timestamp.now().timestamp())
        # 训练模型（copy文件夹）
        model, history = train_model(model, train_generator, val_generator, user_le, item_le, 
                csv_log_path=csv_log_path, run_id=run_id)
        # 根据参数保存训练历史
        if ModelConfig.SAVE_PREDICTION_HISTORY:
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(f'{ModelConfig.SLOPE_OR_PT}/training_history_{model_name}.csv', index=False)
        # 保存模型
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        # 评估copy文件夹的测试集（原有逻辑）
        evaluation = model.evaluate(val_generator, batch_size=ModelConfig.EVALUATION_BATCH_SIZE)
        print(f'Test Loss (copy folder): {evaluation}')
        # 生成copy文件夹的评估报告
        df = evaluate_predictions(val_generator, model, user_le, item_le)
        plot_top_n_predictions(df)
        # 评估test文件夹数据（新增逻辑）
        evaluate_test_folder(selected_folder)
        # 清理显存
        import gc
        from keras import backend as K
        del model, train_generator, val_generator, history, df
        gc.collect()
        K.clear_session()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("显存已清理完毕 ✅")
if __name__ == "__main__":
    main()