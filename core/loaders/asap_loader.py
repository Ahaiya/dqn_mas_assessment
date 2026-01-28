"""
ASAP Dataset Loader (Decoupled Version)
=======================================
功能：
1. 从 data/raw_submissions 读取 TSV 数据
2. 从 data/metadata 读取 JSON 配置 (原文、题目)
3. 组装 EvaluationSubject 对象
"""
import pandas as pd
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Any
from core.schemas import EvaluationSubject, AssessmentArtifact, ArtifactType


class ASAPLoader:
    def __init__(self, tsv_path: str, metadata_path: str):
        """
        初始化加载器
        :param tsv_path: 训练数据路径 (.tsv)
        :param metadata_path: 元数据配置路径 (.json)
        """
        self.tsv_path = tsv_path
        self.metadata_path = metadata_path
        self.df = None

        # 缓存配置数据
        self.context_data: Dict[str, Any] = {}
        self._load_metadata()

    def _load_metadata(self):
        """加载 JSON 配置文件"""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"❌ 元数据文件缺失: {self.metadata_path}")

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.context_data = json.load(f)
        print(f"✅ Metadata loaded from {os.path.basename(self.metadata_path)}")

    def load_dataset(self):
        """加载 TSV 数据集"""
        if not os.path.exists(self.tsv_path):
            raise FileNotFoundError(f"❌ 数据集缺失: {self.tsv_path}")

        print(f"📂 Loading ASAP dataset from {self.tsv_path}...")
        try:
            # ASAP 数据集通常是 ISO-8859-1 编码
            self.df = pd.read_csv(self.tsv_path, sep='\t', encoding='ISO-8859-1')
            # 过滤掉没有 domain1_score 的行
            self.df = self.df.dropna(subset=['domain1_score'])
            print(f"✅ Loaded {len(self.df)} essays.")
        except Exception as e:
            print(f"❌ Read Error: {e}")
            self.df = pd.DataFrame()

    def get_split_indices(self, split: str = 'train', seed: int = 42) -> List[int]:
        """获取切分索引 (80/20 split)"""
        if self.df is None:
            self.load_dataset()

        total_size = len(self.df)
        indices = np.arange(total_size)

        np.random.seed(seed)
        np.random.shuffle(indices)

        split_point = int(total_size * 0.8)

        if split == 'train':
            return indices[:split_point]
        else:
            return indices[split_point:]

    def get_subject_by_index(self, index: int) -> Tuple[EvaluationSubject, float]:
        """
        获取指定行号的数据，并封装为对象
        """
        if self.df is None:
            self.load_dataset()

        row = self.df.iloc[index]

        # 注意：JSON 中的 key 都是字符串，而 DataFrame 里的 set_id 是 int
        set_id_int = int(row['essay_set'])
        set_id_str = str(set_id_int)

        raw_score = float(row['domain1_score'])
        essay_text = str(row['essay'])
        essay_id = str(row['essay_id'])

        # 1. 从 JSON 配置中获取参数
        # 满分范围
        score_ranges = self.context_data.get("score_ranges", {})
        max_score = score_ranges.get(set_id_str, 10)

        # 题目背景
        prompts = self.context_data.get("prompts", {})
        prompt_text = prompts.get(set_id_str, "Unknown Topic")

        # 阅读原文 (仅部分 Set 有)
        source_texts = self.context_data.get("source_texts", {})
        source_text = source_texts.get(set_id_str, None)

        # 2. 分数归一化 (0-5)
        norm_score = (raw_score / max_score) * 5.0
        norm_score = max(0.0, min(5.0, norm_score))

        # 3. 构建对象
        subject = EvaluationSubject(
            subject_id=f"Set{set_id_int}_ID{essay_id}",
            reference_text=source_text,  # 注入原文
            metadata={
                "set_id": set_id_int,
                "raw_max_score": max_score,
                "context": prompt_text,
                "original_score": raw_score
            },
            artifacts=[
                AssessmentArtifact(
                    type=ArtifactType.TEXT_CONTENT,
                    filename=f"essay_set_{set_id_int}.txt",
                    content=essay_text,
                    description=f"Student Essay (Set {set_id_int})"
                )
            ]
        )

        return subject, norm_score