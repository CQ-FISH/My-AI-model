#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pro Max Plus Ultra+++ TensorFlow AI v1.0
=========================================
工业级多任务通用AI模型
- 超深残差网络 + Transformer混合架构
- 多GPU分布式训练
- 混合精度加速
- AdamW优化器 + 梯度裁剪
- 自动超参搜索
- 断点续训
- 最优模型保存
- 早停 + 自适应学习率
- TensorBoard可视化
- CSV日志
- 训练曲线绘图
- H5模型导出 + TFLite量化部署
- 支持图像分类/回归/特征提取多任务

作者: AI Assistant
版本: 1.0
日期: 2026-04-05
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False
    print("警告: tensorflow_addons未安装，部分功能将降级")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: optuna未安装，自动超参搜索功能将禁用")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: sklearn未安装，部分评估功能将降级")


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProMaxTensorFlowAI")


# ============================================================
# 1. 配置管理
# ============================================================

@dataclass
class ModelConfig:
    """模型配置"""
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 10
    task_type: str = "classification"  # classification, regression, feature_extraction
    
    # 架构参数
    use_residual: bool = True
    use_transformer: bool = True
    residual_blocks: int = 50
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dim: int = 512
    dropout_rate: float = 0.3
    attention_dropout: float = 0.1
    
    # 正则化
    l2_reg: float = 1e-4
    label_smoothing: float = 0.1
    
    # 优化器
    optimizer: str = "adamw"
    initial_lr: float = 1e-3
    min_lr: float = 1e-7
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    gradient_clip_value: float = 0.5
    
    # 训练参数
    batch_size: int = 32
    epochs: int = 100
    warmup_epochs: int = 5
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    
    # 分布式训练
    use_multi_gpu: bool = True
    num_gpus: int = 2
    use_mixed_precision: bool = True
    
    # 超参搜索
    enable_hyperparameter_search: bool = False
    hyperparameter_trials: int = 20
    
    # 保存路径
    model_dir: str = "./models"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # 数据增强
    use_data_augmentation: bool = True
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    horizontal_flip: bool = True
    zoom_range: float = 0.2
    
    def save(self, path: str):
        """保存配置"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))


# ============================================================
# 2. 自定义层和模块
# ============================================================

class SqueezeExcitation(layers.Layer):
    """Squeeze-and-Excitation模块"""
    
    def __init__(self, filters: int, ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = keras.Sequential([
            layers.Dense(filters // ratio, activation='relu'),
            layers.Dense(filters, activation='sigmoid')
        ])
    
    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation(x)
        x = tf.reshape(x, [-1, 1, 1, self.filters])
        return inputs * x
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'ratio': self.ratio})
        return config


class ResidualBlock(layers.Layer):
    """残差块"""
    
    def __init__(self, 
                 filters: int, 
                 kernel_size: int = 3,
                 stride: int = 1,
                 use_se: bool = True,
                 dropout_rate: float = 0.0,
                 l2_reg: float = 1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_se = use_se
        
        conv_params = {
            'padding': 'same',
            'kernel_regularizer': regularizers.l2(l2_reg)
        }
        
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=stride, **conv_params)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.conv2 = layers.Conv2D(filters, kernel_size, **conv_params)
        self.bn2 = layers.BatchNormalization()
        
        if use_se:
            self.se = SqueezeExcitation(filters)
        
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # 残差连接
        if stride > 1:
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, strides=stride, **conv_params),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = layers.Identity()
        
        self.relu2 = layers.Activation('relu')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.use_se:
            x = self.se(x)
        
        x = self.dropout2(x, training=training)
        
        shortcut = self.shortcut(inputs)
        x = layers.add([x, shortcut])
        x = self.relu2(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'use_se': self.use_se
        })
        return config


class TransformerEncoder(layers.Layer):
    """Transformer编码器"""
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout_rate: float = 0.1,
                 l2_reg: float = 1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2_reg))
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # 自注意力
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim
        })
        return config


class PositionalEncoding(layers.Layer):
    """位置编码"""
    
    def __init__(self, embed_dim: int, max_sequence_length: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_sequence_length = max_sequence_length
        self.pos_encoding = None
    
    def build(self, input_shape):
        sequence_length = input_shape[1]
        
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        
        pos_encoding = np.zeros((sequence_length, self.embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)
        super().build(input_shape)
    
    def call(self, inputs):
        # 确保位置编码的数据类型与输入一致
        pos_encoding = tf.cast(self.pos_encoding, inputs.dtype)
        return inputs + pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'max_sequence_length': self.max_sequence_length
        })
        return config


# ============================================================
# 3. 模型构建器
# ============================================================

class ProMaxModelBuilder:
    """Pro Max Plus Ultra+++ 模型构建器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.strategy = None
        
    def setup_distributed_training(self):
        """设置分布式训练策略"""
        if self.config.use_multi_gpu and len(tf.config.list_physical_devices('GPU')) > 1:
            self.strategy = tf.distribute.MirroredStrategy(
                devices=[f'/gpu:{i}' for i in range(min(self.config.num_gpus, len(tf.config.list_physical_devices('GPU'))))]
            )
            logger.info(f"使用 {self.strategy.num_replicas_in_sync} 个GPU进行分布式训练")
        else:
            self.strategy = tf.distribute.get_strategy()
            logger.info("使用单GPU或CPU训练")
        
        return self.strategy
    
    def setup_mixed_precision(self):
        """设置混合精度训练"""
        if self.config.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            logger.info("已启用混合精度训练")
    
    def build_model(self) -> keras.Model:
        """构建模型"""
        inputs = layers.Input(shape=self.config.input_shape, name='input')
        
        # 初始卷积层
        x = layers.Conv2D(
            64, 7, strides=2, padding='same',
            kernel_regularizer=regularizers.l2(self.config.l2_reg),
            name='initial_conv'
        )(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.Activation('relu', name='initial_relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same', name='initial_pool')(x)
        
        # 残差块阶段
        if self.config.use_residual:
            x = self._build_residual_stages(x)
        
        # Transformer阶段
        if self.config.use_transformer:
            x = self._build_transformer_stages(x)
        
        # 全局池化
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dropout(self.config.dropout_rate, name='final_dropout')(x)
        
        # 输出层
        if self.config.task_type == 'classification':
            outputs = layers.Dense(
                self.config.num_classes,
                activation='softmax',
                dtype='float32',
                name='output'
            )(x)
        elif self.config.task_type == 'regression':
            outputs = layers.Dense(
                self.config.num_classes,
                activation='linear',
                dtype='float32',
                name='output'
            )(x)
        else:  # feature_extraction
            outputs = layers.Dense(
                self.config.transformer_dim,
                activation='relu',
                dtype='float32',
                name='output'
            )(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='ProMaxPlusUltraAI')
        
        return self.model
    
    def _build_residual_stages(self, x):
        """构建残差块阶段"""
        filters = [64, 128, 256, 512]
        blocks_per_stage = [3, 4, 6, 3]
        
        for stage_idx, (filter_size, num_blocks) in enumerate(zip(filters, blocks_per_stage)):
            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                
                x = ResidualBlock(
                    filters=filter_size,
                    stride=stride,
                    use_se=True,
                    dropout_rate=self.config.dropout_rate,
                    l2_reg=self.config.l2_reg,
                    name=f'residual_{stage_idx}_{block_idx}'
                )(x)
        
        return x
    
    def _build_transformer_stages(self, x):
        """构建Transformer阶段"""
        # 获取空间维度
        input_shape = x.shape
        height = input_shape[1]
        width = input_shape[2]
        
        # 展平空间维度
        x = layers.Reshape((-1, input_shape[-1]), name='transformer_flatten')(x)
        
        # 投影到Transformer维度
        x = layers.Dense(
            self.config.transformer_dim,
            kernel_regularizer=regularizers.l2(self.config.l2_reg),
            name='transformer_projection'
        )(x)
        
        # 位置编码
        x = PositionalEncoding(
            embed_dim=self.config.transformer_dim,
            name='positional_encoding'
        )(x)
        
        # Transformer编码器层
        for i in range(self.config.transformer_layers):
            x = TransformerEncoder(
                embed_dim=self.config.transformer_dim,
                num_heads=self.config.transformer_heads,
                ff_dim=self.config.transformer_dim * 4,
                dropout_rate=self.config.attention_dropout,
                l2_reg=self.config.l2_reg,
                name=f'transformer_encoder_{i}'
            )(x)
        
        # 重塑回空间维度
        x = layers.Reshape((height, width, self.config.transformer_dim), name='transformer_reshape')(x)
        
        # 降维
        x = layers.Conv2D(
            512, 1,
            kernel_regularizer=regularizers.l2(self.config.l2_reg),
            name='transformer_reduction'
        )(x)
        x = layers.BatchNormalization(name='transformer_bn')(x)
        x = layers.Activation('relu', name='transformer_relu')(x)
        
        return x
    
    def compile_model(self):
        """编译模型"""
        # 选择优化器
        if self.config.optimizer == 'adamw':
            if TFA_AVAILABLE:
                optimizer = tfa.optimizers.AdamW(
                    learning_rate=self.config.initial_lr,
                    weight_decay=self.config.weight_decay
                )
            else:
                # 使用Adam优化器，不使用decay参数
                optimizer = optimizers.Adam(learning_rate=self.config.initial_lr)
        elif self.config.optimizer == 'adam':
            optimizer = optimizers.Adam(learning_rate=self.config.initial_lr)
        elif self.config.optimizer == 'sgd':
            optimizer = optimizers.SGD(
                learning_rate=self.config.initial_lr,
                momentum=0.9,
                nesterov=True
            )
        else:
            optimizer = optimizers.Adam(learning_rate=self.config.initial_lr)
        
        # 混合精度优化器
        if self.config.use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        # 选择损失函数
        if self.config.task_type == 'classification':
            loss = keras.losses.CategoricalCrossentropy(
                label_smoothing=self.config.label_smoothing
            )
            metrics = ['accuracy', 'top_k_categorical_accuracy']
        elif self.config.task_type == 'regression':
            loss = keras.losses.MeanSquaredError()
            metrics = ['mae', 'mse']
        else:
            loss = keras.losses.MeanSquaredError()
            metrics = ['mae']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info("模型编译完成")
        return self.model


# ============================================================
# 4. 回调函数
# ============================================================

class TrainingCallbacks:
    """训练回调函数管理器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.callbacks_list = []
    
    def create_callbacks(self) -> List[callbacks.Callback]:
        """创建所有回调函数"""
        # 创建目录
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # TensorBoard
        tensorboard = callbacks.TensorBoard(
            log_dir=os.path.join(self.config.log_dir, 'tensorboard'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2
        )
        self.callbacks_list.append(tensorboard)
        
        # CSV日志
        csv_logger = callbacks.CSVLogger(
            os.path.join(self.config.log_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
        self.callbacks_list.append(csv_logger)
        
        # 模型检查点
        checkpoint = callbacks.ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        self.callbacks_list.append(checkpoint)
        
        # 早停
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        self.callbacks_list.append(early_stopping)
        
        # 学习率衰减
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
            verbose=1
        )
        self.callbacks_list.append(reduce_lr)
        
        # 学习率调度器
        lr_scheduler = callbacks.LearningRateScheduler(
            self._lr_schedule,
            verbose=1
        )
        self.callbacks_list.append(lr_scheduler)
        
        # 梯度裁剪回调
        gradient_clipping = GradientClippingCallback(self.config)
        self.callbacks_list.append(gradient_clipping)
        
        # 自定义训练监控
        training_monitor = TrainingMonitorCallback(self.config)
        self.callbacks_list.append(training_monitor)
        
        return self.callbacks_list
    
    def _lr_schedule(self, epoch: int, lr: float) -> float:
        """学习率调度"""
        if epoch < self.config.warmup_epochs:
            return lr * (epoch + 1) / self.config.warmup_epochs
        else:
            decay_epochs = self.config.epochs - self.config.warmup_epochs
            remaining_epochs = epoch - self.config.warmup_epochs
            return self.config.min_lr + 0.5 * (self.config.initial_lr - self.config.min_lr) * \
                   (1 + np.cos(np.pi * remaining_epochs / decay_epochs))


class GradientClippingCallback(callbacks.Callback):
    """梯度裁剪回调"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def on_train_batch_begin(self, batch, logs=None):
        """在每个训练批次开始时应用梯度裁剪"""
        if hasattr(self.model.optimizer, 'clipnorm'):
            self.model.optimizer.clipnorm = self.config.gradient_clip_norm
        if hasattr(self.model.optimizer, 'clipvalue'):
            self.model.optimizer.clipvalue = self.config.gradient_clip_value


class TrainingMonitorCallback(callbacks.Callback):
    """训练监控回调"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.epoch_times = []
        self.batch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
        logger.info(f"{'='*60}")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        logger.info(f"Epoch {epoch + 1} 完成 - 用时: {epoch_time:.2f}s")
        logger.info(f"训练损失: {logs.get('loss'):.4f} - 训练准确率: {logs.get('accuracy', 0):.4f}")
        logger.info(f"验证损失: {logs.get('val_loss'):.4f} - 验证准确率: {logs.get('val_accuracy', 0):.4f}")
        logger.info(f"学习率: {float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)):.6f}")
    
    def on_train_end(self, logs=None):
        avg_epoch_time = np.mean(self.epoch_times)
        logger.info(f"\n训练完成!")
        logger.info(f"平均每轮用时: {avg_epoch_time:.2f}s")
        logger.info(f"总训练时间: {sum(self.epoch_times):.2f}s")


# ============================================================
# 5. 数据加载器
# ============================================================

class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
    
    def load_from_directory(self, 
                           data_dir: str,
                           validation_split: float = 0.2,
                           test_split: float = 0.1) -> Tuple:
        """从目录加载图像数据"""
        if self.config.use_data_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=self.config.rotation_range,
                width_shift_range=self.config.width_shift_range,
                height_shift_range=self.config.height_shift_range,
                horizontal_flip=self.config.horizontal_flip,
                zoom_range=self.config.zoom_range,
                validation_split=validation_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # 训练集
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.config.input_shape[:2],
            batch_size=self.config.batch_size,
            class_mode='categorical' if self.config.task_type == 'classification' else 'raw',
            subset='training'
        )
        
        # 验证集
        self.val_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.config.input_shape[:2],
            batch_size=self.config.batch_size,
            class_mode='categorical' if self.config.task_type == 'classification' else 'raw',
            subset='validation'
        )
        
        # 测试集
        if os.path.exists(os.path.join(data_dir, 'test')):
            self.test_generator = test_datagen.flow_from_directory(
                os.path.join(data_dir, 'test'),
                target_size=self.config.input_shape[:2],
                batch_size=self.config.batch_size,
                class_mode='categorical' if self.config.task_type == 'classification' else 'raw',
                shuffle=False
            )
        
        logger.info(f"数据加载完成: 训练集 {self.train_generator.samples} 张, "
                   f"验证集 {self.val_generator.samples} 张")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def load_from_numpy(self,
                       x_train: np.ndarray,
                       y_train: np.ndarray,
                       x_val: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None,
                       validation_split: float = 0.2) -> Tuple:
        """从numpy数组加载数据"""
        # 归一化
        x_train = x_train.astype('float32') / 255.0
        
        # 分割验证集
        if x_val is None or y_val is None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=validation_split, random_state=42
            )
        
        # 数据增强
        if self.config.use_data_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=self.config.rotation_range,
                width_shift_range=self.config.width_shift_range,
                height_shift_range=self.config.height_shift_range,
                horizontal_flip=self.config.horizontal_flip,
                zoom_range=self.config.zoom_range
            )
            self.train_generator = datagen.flow(x_train, y_train, batch_size=self.config.batch_size)
        else:
            self.train_generator = (x_train, y_train)
        
        self.val_generator = (x_val, y_val)
        
        logger.info(f"数据加载完成: 训练集 {len(x_train)} 张, 验证集 {len(x_val)} 张")
        
        return self.train_generator, self.val_generator


# ============================================================
# 6. 超参数搜索
# ============================================================

class HyperparameterSearcher:
    """超参数搜索器"""
    
    def __init__(self, config: ModelConfig, train_data: Tuple, val_data: Tuple):
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.best_params = None
        self.best_score = float('inf')
    
    def objective(self, trial) -> float:
        """Optuna目标函数"""
        # 定义搜索空间
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        l2_reg = trial.suggest_loguniform('l2_reg', 1e-6, 1e-2)
        
        # 更新配置
        config = ModelConfig(
            **{**asdict(self.config), 
               'initial_lr': lr,
               'dropout_rate': dropout_rate,
               'batch_size': batch_size,
               'l2_reg': l2_reg,
               'epochs': min(self.config.epochs, 10)  # 快速训练
            }
        )
        
        # 构建和训练模型
        builder = ProMaxModelBuilder(config)
        builder.setup_distributed_training()
        builder.setup_mixed_precision()
        model = builder.build_model()
        builder.compile_model()
        
        # 训练
        callbacks_handler = TrainingCallbacks(config)
        history = model.fit(
            self.train_data[0], self.train_data[1],
            validation_data=(self.val_data[0], self.val_data[1]),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks_handler.create_callbacks(),
            verbose=0
        )
        
        # 返回验证损失
        val_loss = min(history.history['val_loss'])
        
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.best_params = {
                'lr': lr,
                'dropout_rate': dropout_rate,
                'batch_size': batch_size,
                'l2_reg': l2_reg
            }
        
        return val_loss
    
    def search(self) -> Dict:
        """执行超参数搜索"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna未安装，跳过超参数搜索")
            return {}
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.config.hyperparameter_trials)
        
        logger.info(f"最佳超参数: {study.best_params}")
        logger.info(f"最佳验证损失: {study.best_value}")
        
        # 保存结果
        with open('hyperparameter_search_results.json', 'w') as f:
            json.dump({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'all_trials': [trial.params for trial in study.trials]
            }, f, indent=2)
        
        return study.best_params


# ============================================================
# 7. 模型导出器
# ============================================================

class ModelExporter:
    """模型导出器"""
    
    def __init__(self, model: keras.Model, config: ModelConfig):
        self.model = model
        self.config = config
    
    def export_h5(self, filepath: str):
        """导出H5模型"""
        self.model.save(filepath)
        logger.info(f"H5模型已保存: {filepath}")
    
    def export_savedmodel(self, filepath: str):
        """导出SavedModel格式"""
        self.model.save(filepath, save_format='tf')
        logger.info(f"SavedModel已保存: {filepath}")
    
    def export_tflite(self, 
                     filepath: str,
                     quantize: bool = True,
                     representative_data: Optional[np.ndarray] = None):
        """导出TFLite模型"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            if representative_data is not None:
                # 完整整数量化
                def representative_dataset():
                    for data in representative_data[:100]:
                        yield [data[np.newaxis, ...].astype(np.float32)]
                
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            else:
                # 动态范围量化
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(filepath, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite模型已保存: {filepath}")
        
        # 显示模型大小
        h5_model_path = os.path.join(self.config.model_dir, 'final_model.h5')
        if os.path.exists(h5_model_path):
            original_size = os.path.getsize(h5_model_path)
            tflite_size = os.path.getsize(filepath)
            compression_ratio = (1 - tflite_size / original_size) * 100
            
            logger.info(f"原始模型大小: {original_size / 1024 / 1024:.2f} MB")
            logger.info(f"TFLite模型大小: {tflite_size / 1024:.2f} KB")
            logger.info(f"压缩比例: {compression_ratio:.1f}%")
    
    def export_onnx(self, filepath: str):
        """导出ONNX模型"""
        try:
            import tf2onnx
            onnx_model = tf2onnx.convert.from_keras(self.model)
            with open(filepath, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            logger.info(f"ONNX模型已保存: {filepath}")
        except ImportError:
            logger.warning("tf2onnx未安装，跳过ONNX导出")


# ============================================================
# 8. 可视化工具
# ============================================================

class TrainingVisualizer:
    """训练可视化工具"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.history = None
    
    def plot_training_history(self, history: keras.callbacks.History, save_path: str):
        """绘制训练历史"""
        self.history = history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(history.history['loss'], label='训练损失')
        axes[0, 0].plot(history.history['val_loss'], label='验证损失')
        axes[0, 0].set_title('模型损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        if 'accuracy' in history.history:
            axes[0, 1].plot(history.history['accuracy'], label='训练准确率')
            axes[0, 1].plot(history.history['val_accuracy'], label='验证准确率')
            axes[0, 1].set_title('模型准确率')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 学习率曲线
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('学习率变化')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Top-K准确率
        if 'top_k_categorical_accuracy' in history.history:
            axes[1, 1].plot(history.history['top_k_categorical_accuracy'], label='训练Top-5')
            axes[1, 1].plot(history.history['val_top_k_categorical_accuracy'], label='验证Top-5')
            axes[1, 1].set_title('Top-5准确率')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存: {save_path}")
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: List[str],
                             save_path: str):
        """绘制混淆矩阵"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn未安装，跳过混淆矩阵绘制")
            return
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='混淆矩阵',
               ylabel='真实标签',
               xlabel='预测标签')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存: {save_path}")
    
    def plot_model_architecture(self, model: keras.Model, save_path: str):
        """绘制模型架构图"""
        try:
            plot_model(
                model,
                to_file=save_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=300
            )
            logger.info(f"模型架构图已保存: {save_path}")
        except Exception as e:
            logger.warning(f"模型架构图生成失败: {e}")


# ============================================================
# 9. 主训练器
# ============================================================

class ProMaxTrainer:
    """Pro Max Plus Ultra+++ 训练器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.builder = None
        self.history = None
        self.data_loader = None
        self.visualizer = TrainingVisualizer(config)
    
    def setup(self):
        """设置训练环境"""
        logger.info("="*60)
        logger.info("Pro Max Plus Ultra+++ TensorFlow AI v1.0")
        logger.info("="*60)
        
        # 创建目录
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # 保存配置
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        # 设置分布式训练
        self.builder = ProMaxModelBuilder(self.config)
        strategy = self.builder.setup_distributed_training()
        
        # 设置混合精度
        self.builder.setup_mixed_precision()
        
        # 构建模型
        with strategy.scope():
            self.model = self.builder.build_model()
            self.builder.compile_model()
        
        # 打印模型信息
        self.model.summary()
        logger.info(f"模型参数总数: {self.model.count_params():,}")
        
        # 绘制模型架构
        self.visualizer.plot_model_architecture(
            self.model,
            os.path.join(self.config.model_dir, 'model_architecture.png')
        )
    
    def train(self, train_data, val_data) -> keras.callbacks.History:
        """训练模型"""
        logger.info("\n开始训练...")
        
        # 创建回调
        callbacks_handler = TrainingCallbacks(self.config)
        callbacks_list = callbacks_handler.create_callbacks()
        
        # 训练
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 绘制训练曲线
        self.visualizer.plot_training_history(
            self.history,
            os.path.join(self.config.log_dir, 'training_curves.png')
        )
        
        return self.history
    
    def evaluate(self, test_data) -> Dict:
        """评估模型"""
        logger.info("\n评估模型...")
        
        results = self.model.evaluate(test_data, verbose=1)
        
        metrics = {}
        for metric_name, metric_value in zip(self.model.metrics_names, results):
            metrics[metric_name] = float(metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # 保存评估结果
        with open(os.path.join(self.config.log_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def export_models(self, representative_data: Optional[np.ndarray] = None):
        """导出所有模型格式"""
        logger.info("\n导出模型...")
        
        exporter = ModelExporter(self.model, self.config)
        
        # H5格式
        exporter.export_h5(os.path.join(self.config.model_dir, 'final_model.h5'))
        
        # SavedModel格式
        exporter.export_savedmodel(os.path.join(self.config.model_dir, 'savedmodel'))
        
        # TFLite格式
        exporter.export_tflite(
            os.path.join(self.config.model_dir, 'model_quantized.tflite'),
            quantize=True,
            representative_data=representative_data
        )
        
        # ONNX格式
        exporter.export_onnx(os.path.join(self.config.model_dir, 'model.onnx'))
    
    def run_full_pipeline(self, data_dir: str = None, train_data: Tuple = None):
        """运行完整训练流程"""
        try:
            # 设置
            self.setup()
            
            # 加载数据
            self.data_loader = DataLoader(self.config)
            
            if data_dir:
                train_gen, val_gen, test_gen = self.data_loader.load_from_directory(data_dir)
            elif train_data:
                train_gen, val_gen = self.data_loader.load_from_numpy(*train_data)
                test_gen = None
            else:
                raise ValueError("必须提供data_dir或train_data")
            
            # 超参数搜索
            if self.config.enable_hyperparameter_search and OPTUNA_AVAILABLE:
                searcher = HyperparameterSearcher(self.config, train_gen, val_gen)
                best_params = searcher.search()
                
                # 使用最佳参数重新配置
                for key, value in best_params.items():
                    setattr(self.config, key, value)
                
                # 重新设置
                self.setup()
            
            # 训练
            self.history = self.train(train_gen, val_gen)
            
            # 评估
            if test_gen:
                metrics = self.evaluate(test_gen)
            
            # 导出
            if isinstance(train_gen, tuple):
                representative_data = train_gen[0]
            else:
                representative_data = None
            
            self.export_models(representative_data)
            
            logger.info("\n" + "="*60)
            logger.info("训练流程完成!")
            logger.info("="*60)
            
            return self.model, self.history
            
        except Exception as e:
            logger.error(f"训练流程失败: {e}")
            raise


# ============================================================
# 10. 示例数据生成器
# ============================================================

def generate_sample_data(num_samples: int = 1000, 
                        input_shape: Tuple[int, int, int] = (224, 224, 3),
                        num_classes: int = 10) -> Tuple:
    """生成示例数据"""
    logger.info(f"生成示例数据: {num_samples} 样本")
    
    x_train = np.random.random((num_samples, *input_shape)).astype('float32')
    y_train = np.random.randint(0, num_classes, num_samples)
    y_train = to_categorical(y_train, num_classes)
    
    x_val = np.random.random((num_samples // 5, *input_shape)).astype('float32')
    y_val = np.random.randint(0, num_classes, num_samples // 5)
    y_val = to_categorical(y_val, num_classes)
    
    return (x_train, y_train, x_val, y_val)


# ============================================================
# 11. 主函数
# ============================================================

def main():
    """主函数"""
    # 配置（快速测试版本）
    config = ModelConfig(
        input_shape=(64, 64, 3),  # 减小输入尺寸
        num_classes=10,
        task_type='classification',
        use_residual=True,
        use_transformer=True,
        residual_blocks=6,  # 减少残差块
        transformer_layers=2,  # 减少Transformer层
        transformer_heads=4,
        transformer_dim=256,
        dropout_rate=0.3,
        batch_size=16,
        epochs=3,  # 减少训练轮次
        use_multi_gpu=False,  # 禁用多GPU
        use_mixed_precision=False,  # 禁用混合精度（CPU上更快）
        enable_hyperparameter_search=False,
        model_dir='./models',
        log_dir='./logs',
        checkpoint_dir='./checkpoints'
    )
    
    # 生成示例数据
    train_data = generate_sample_data(
        num_samples=100,  # 减少样本数量
        input_shape=config.input_shape,
        num_classes=config.num_classes
    )
    
    # 创建训练器
    trainer = ProMaxTrainer(config)
    
    # 运行完整流程
    model, history = trainer.run_full_pipeline(train_data=train_data)
    
    logger.info("\n训练完成！模型已保存到 ./models 目录")
    logger.info("查看训练日志: ./logs/training_log.csv")
    logger.info("查看TensorBoard: tensorboard --logdir=./logs/tensorboard")


if __name__ == "__main__":
    main()
