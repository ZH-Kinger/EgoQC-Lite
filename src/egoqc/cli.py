from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .pipeline import run
from .extract import extract_samples
from .annotated_video import render_annotated_episode
from .episode_qc import inspect_episode
from .mano import HaworManoBackend, ManoOverlayRenderer
from .temporal_plot import write_temporal_plot
from .robot20 import inspect_robot_pair
from .robot_mesh import render_robot
from .report import write_json
from .ops import doctor, self_test
from .dashboard import write_registry_dashboard
from .estimate import estimate_manifest
from .tuner import write_tuner
from .experiment_evidence import build_experiment_evidence
from .decisions import create_retry_plan
from .repair import write_repair_preview
from .adapters import inspect_adapter
from .live_server import serve_review
from .review_db import ReviewStore, load_event_file
from .review_pg_server import serve_postgres_review
from .gold_review import build_phase_a_review_events
from .hand_screen import screen_rekadaily_hands
from .completion import plan_public_completion, build_completion_overlay
from .training_views import build_rekadaily_training_views
from .vla_dataset import smoke_vla_loader
from .vla_train import smoke_vla_train
from .distillation import (
    audit_qc_training_data,
    build_distillation_manifest,
    evaluate_qc_predictions,
    smoke_train_qc_student,
)
from .research_evaluation import evaluate_qc_research_protocol
from .clip_selection import plan_qc_clips
from .adapter_clip_selection import plan_adapter_clips
from .teacher_api import run_teacher_api
from .teacher_queue import (
    build_overlay_teacher_queue,
    merge_teacher_queues,
    normalize_teacher_queue_provenance,
)
from .training_partition import freeze_training_partition
from .queue_gold import build_queue_gold_review, render_queue_gold_overlays
from .teacher_training_pool import build_teacher_training_pool
from .synthetic_qc import build_synthetic_qc_training
from .interventions import plan_qc_interventions, run_qc_interventions
from .undistortion import plan_vitra_undistortion, run_vitra_undistortion, verify_vitra_undistortion
from .egodex_overlay import render_egodex_overlay
from .egodex_batch import build_egodex_training_candidates
from .egodex_review_batch import build_egodex_review_batch
from .multisource_discovery import discover_lerobot_roots
from .generic_ego import build_generic_ego_views
from .task_taxonomy import classify_task_records
from .registry import (
    create_manifest,
    register_datasets,
    registry_status,
    run_manifest,
)


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="egoqc",
        description="低成本、增量式 LeRobot v3 ego/MANO 数据质量检查",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="扫描并生成质量报告")
    scan.add_argument("dataset", type=Path)
    scan.add_argument("--output", type=Path, default=Path(".egoqc"))
    default_config = Path(__file__).with_name("default.json")
    scan.add_argument("--config", type=Path, default=default_config)
    scan.add_argument("--hash-mode", choices=("none", "headtail"), default="headtail")
    scan.add_argument(
        "--video-check", choices=("header", "count", "sample-quality"),
        help="视频检查成本档位；默认读取配置（生产全量建议 header）",
    )
    extract = sub.add_parser("extract-samples", help="按抽帧计划顺序读取聚合 MP4 并生成证据图")
    extract.add_argument("dataset", type=Path)
    extract.add_argument("--plan", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--video-key", default="observation.images.ego")
    extract.add_argument(
        "--hawor-root",
        type=Path,
        help="HaWoR 仓库根目录；提供后启用 MANO mesh/joints 重投影",
    )
    extract.add_argument(
        "--mano-data-root",
        type=Path,
        help="HaWoR _DATA 根目录，默认使用 <hawor-root>/_DATA",
    )
    extract.add_argument("--mano-alpha", type=float, default=0.48)
    extract.add_argument(
        "--no-seek",
        action="store_true",
        help="禁用 PTS 关键帧 seek，始终从聚合视频开头顺序解码",
    )
    extract.add_argument("--seek-margin-s", type=float, default=2.0)
    extract.add_argument("--seek-min-frame", type=int, default=300)
    extract.add_argument(
        "--annotated-root", type=Path,
        help="可选 repaired annotated MP4 目录，用于复检页的 MANO 参考标签页",
    )
    annotated = sub.add_parser("render-annotated-video", help="把 MANO 和审核信息烧录到派生 MP4")
    annotated.add_argument("dataset", type=Path)
    annotated.add_argument("--episode", type=int, required=True)
    annotated.add_argument("--output", type=Path, required=True)
    annotated.add_argument("--hawor-root", type=Path, required=True)
    annotated.add_argument("--mano-data-root", type=Path)
    annotated.add_argument("--mano-alpha", type=float, default=0.48)
    annotated.add_argument("--video-key", default="observation.images.ego")
    annotated.add_argument("--batch-size", type=int, default=32)
    annotated.add_argument("--start-frame", type=int, default=0)
    annotated.add_argument("--max-frames", type=int)
    annotated.add_argument("--review-labels", type=Path)
    inspect_ep = sub.add_parser("inspect-episode", help="只检查大型 LeRobot 根中的一个 episode")
    inspect_ep.add_argument("dataset", type=Path)
    inspect_ep.add_argument("--episode", type=int, required=True)
    inspect_ep.add_argument("--config", type=Path, default=default_config)
    inspect_ep.add_argument("--output", type=Path)
    temporal_plot = sub.add_parser("temporal-plot", help="生成单个 episode 的抖动时序 SVG")
    temporal_plot.add_argument("dataset", type=Path)
    temporal_plot.add_argument("--episode", type=int, required=True)
    temporal_plot.add_argument("--output", type=Path, required=True)
    temporal_plot.add_argument("--config", type=Path, default=default_config)
    repair = sub.add_parser("repair-preview", help="生成断点感知的手部平滑预览，不覆盖源数据")
    repair.add_argument("dataset", type=Path)
    repair.add_argument("--episode", type=int, required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--config", type=Path, default=default_config)
    repair.add_argument("--video-key", default="observation.images.ego")
    repair.add_argument("--position-min-cutoff", type=float)
    repair.add_argument("--position-beta", type=float)
    repair.add_argument("--rotation-min-cutoff", type=float)
    repair.add_argument("--rotation-beta", type=float)
    repair.add_argument("--derivative-cutoff", type=float)
    repair.add_argument("--hawor-root", type=Path, help="可选；启用 repaired MANO 标注视频")
    repair.add_argument("--mano-data-root", type=Path)
    repair.add_argument("--mano-alpha", type=float, default=0.48)
    repair.add_argument("--start-frame", type=int, default=0)
    repair.add_argument("--max-frames", type=int)
    adapter = sub.add_parser("inspect-adapter", help="自动识别来源并输出只读 Canonical Episode 视图")
    adapter.add_argument("dataset", type=Path)
    adapter.add_argument(
        "--episode",
        help="LeRobot episode_index、EgoDex 相对路径，或 RekaDaily video_id；R​​ekaDaily 省略时只汇总索引",
    )
    adapter.add_argument("--confidence-threshold", type=float, default=0.5)
    adapter.add_argument(
        "--video-check", choices=("header", "count", "sample-quality"), default="header"
    )
    adapter.add_argument("--config", type=Path, default=default_config)
    adapter.add_argument("--output", type=Path)
    generic_ego = sub.add_parser(
        "build-generic-ego-views",
        help="只读发现通用 raw ego 视频，探测能力并生成缺失模态感知的训练/质检 manifest",
    )
    generic_ego.add_argument("dataset", type=Path)
    generic_ego.add_argument("--output", type=Path, required=True)
    generic_ego.add_argument("--source-dataset")
    generic_ego.add_argument(
        "--source-class",
        choices=("generic_ego", "public_dataset", "supplier_dataset", "internal_dataset"),
        default="generic_ego",
    )
    generic_ego.add_argument("--license-id")
    generic_ego.add_argument("--workers", type=int, default=16)
    generic_ego.add_argument("--maximum-depth", type=int)
    generic_ego.add_argument("--limit", type=int)
    generic_ego.add_argument(
        "--video-check", choices=("header", "count", "sample-quality"), default="header"
    )
    generic_ego.add_argument("--config", type=Path, default=default_config)
    completion_plan = sub.add_parser(
        "plan-completion",
        help="分析公开 LeRobot 数据的非关键缺失字段并生成安全补齐计划",
    )
    completion_plan.add_argument("dataset", type=Path)
    completion_plan.add_argument("--config", type=Path, default=default_config)
    completion_plan.add_argument("--output", type=Path, required=True)
    completion_build = sub.add_parser(
        "build-completion-overlay",
        help="把确定性补齐字段写成独立 Parquet overlay，不覆盖 raw",
    )
    completion_build.add_argument("dataset", type=Path)
    completion_build.add_argument("--plan", type=Path, required=True)
    completion_build.add_argument("--output", type=Path, required=True)
    hand_screen = sub.add_parser(
        "screen-rekadaily-hands",
        help="用 GPU 手检测器对 RekaDaily tar 做可见率、离画和疑似多人预筛",
    )
    hand_screen.add_argument("dataset", type=Path)
    hand_screen.add_argument("--video-id", action="append", required=True)
    hand_screen.add_argument("--output", type=Path, required=True)
    hand_screen.add_argument("--weights", type=Path, required=True)
    hand_screen.add_argument("--sample-fps", type=float, default=5.0)
    hand_screen.add_argument("--confidence", type=float, default=0.2)
    hand_screen.add_argument("--batch-size", type=int, default=32)
    hand_screen.add_argument("--device", default="0")
    hand_screen.add_argument("--workers", type=int, default=1)
    hand_screen.add_argument("--no-resume", action="store_true")
    training_views = sub.add_parser(
        "build-rekadaily-views",
        help="生成 RekaDaily 视频预训练与 MANO Silver 分阶段 manifest",
    )
    training_views.add_argument("dataset", type=Path)
    training_views.add_argument("--output", type=Path, required=True)
    training_views.add_argument("--materialized-only", action="store_true")
    training_views.add_argument("--hand-screen-root", type=Path)
    training_views.add_argument("--mano-root", type=Path)
    training_views.add_argument("--alignment-root", type=Path)
    training_views.add_argument("--minimum-duration-s", type=float, default=5.0)
    training_views.add_argument("--project", action="append", dest="projects")
    training_views.add_argument("--limit", type=int)
    training_views.add_argument(
        "--license-id",
        help="许可证或内部审批编号；缺失时只产出 technical candidates，不进入 training-ready",
    )
    vla_smoke = sub.add_parser(
        "smoke-vla-loader",
        help="真实解码 VLA manifest，验证 clip、batch、文本与 loss mask",
    )
    vla_smoke.add_argument("--manifest", type=Path, required=True)
    vla_smoke.add_argument("--output", type=Path, required=True)
    vla_smoke.add_argument("--split", choices=("train", "validation", "test"), default="train")
    vla_smoke.add_argument("--batch-size", type=int, default=2)
    vla_smoke.add_argument("--seed", type=int, default=0)
    vla_smoke.add_argument(
        "--allow-technical-candidates",
        action="store_true",
        help="仅用于数据加载调试；不代表已获得训练许可",
    )
    vla_train = sub.add_parser(
        "smoke-vla-train",
        help="在真实视频 batch 上运行轻量 PyTorch 前向、反向与参数更新",
    )
    vla_train.add_argument("--manifest", type=Path, required=True)
    vla_train.add_argument("--output", type=Path, required=True)
    vla_train.add_argument("--split", choices=("train", "validation", "test"), default="train")
    vla_train.add_argument("--batch-size", type=int, default=2)
    vla_train.add_argument("--steps", type=int, default=5)
    vla_train.add_argument("--learning-rate", type=float, default=3e-4)
    vla_train.add_argument("--device", default="cuda")
    vla_train.add_argument("--seed", type=int, default=0)
    vla_train.add_argument("--allow-technical-candidates", action="store_true")
    distill_build = sub.add_parser(
        "build-qc-distillation",
        help="合并程序化弱标签、本地VLM教师标签和人工Gold Set",
    )
    distill_build.add_argument("--records", type=Path, required=True)
    distill_build.add_argument("--task-config", type=Path, default=Path("config/visual_model_tasks.json"))
    distill_build.add_argument("--output", type=Path, required=True)
    distill_build.add_argument("--hand-screen-root", type=Path)
    distill_build.add_argument("--teacher-root", type=Path)
    distill_build.add_argument("--gold-labels", type=Path)
    teacher_pool = sub.add_parser(
        "build-teacher-training-pool",
        help="把 API 教师 clip 弱标签整理为仅训练用的高置信蒸馏数据",
    )
    teacher_pool.add_argument("--queue", type=Path, required=True)
    teacher_pool.add_argument("--task-config", type=Path, default=Path("config/visual_model_tasks.json"))
    teacher_pool.add_argument("--output", type=Path, required=True)
    teacher_pool.add_argument("--minimum-overall-confidence", type=float, default=0.85)
    teacher_pool.add_argument("--minimum-task-confidence", type=float, default=0.70)
    synthetic_qc = sub.add_parser(
        "build-synthetic-qc-training",
        help="从高置信干净 clip 构建只用于训练的在线受控异常增强",
    )
    synthetic_qc.add_argument("--manifest", type=Path, required=True)
    synthetic_qc.add_argument("--output", type=Path, required=True)
    synthetic_qc.add_argument("--maximum-per-task", type=int, default=400)
    synthetic_qc.add_argument("--seed", type=int, default=31)
    intervention_plan = sub.add_parser(
        "plan-qc-interventions",
        help="为 Phase A 生成不复制、不覆盖源数据的可控干预虚拟视图",
    )
    intervention_plan.add_argument("dataset", type=Path)
    intervention_plan.add_argument("--output", type=Path, required=True)
    intervention_plan.add_argument(
        "--intervention-config",
        type=Path,
        default=Path("config/qc_interventions_phase_a.json"),
    )
    intervention_plan.add_argument(
        "--episode", type=int, action="append", dest="episodes"
    )
    intervention_plan.add_argument(
        "--family", action="append", dest="families"
    )
    intervention_plan.add_argument("--maximum-episodes", type=int, default=32)
    intervention_plan.add_argument("--seed", type=int, default=20260819)
    intervention_run = sub.add_parser(
        "run-qc-interventions",
        help="重放虚拟干预并记录规则/几何专家的前后 evidence delta",
    )
    intervention_run.add_argument("dataset", type=Path)
    intervention_run.add_argument("--manifest", type=Path, required=True)
    intervention_run.add_argument("--output", type=Path, required=True)
    intervention_run.add_argument("--config", type=Path, default=default_config)
    intervention_run.add_argument("--maximum-interventions", type=int)
    distill_audit = sub.add_parser(
        "audit-qc-training",
        help="检查 Gold Set 覆盖、分组切分、跨 split 泄漏和训练治理状态",
    )
    distill_audit.add_argument("--manifest", type=Path, required=True)
    distill_audit.add_argument("--task-config", type=Path, default=Path("config/visual_model_tasks.json"))
    distill_audit.add_argument("--output", type=Path, required=True)
    distill_smoke = sub.add_parser(
        "smoke-qc-student",
        help="用蒸馏 manifest 训练轻量时序 QC student 工程样机",
    )
    distill_smoke.add_argument("--manifest", type=Path, required=True)
    distill_smoke.add_argument("--output", type=Path, required=True)
    distill_smoke.add_argument("--steps", type=int, default=20)
    distill_smoke.add_argument("--batch-size", type=int, default=4)
    distill_smoke.add_argument("--learning-rate", type=float, default=5e-4)
    distill_smoke.add_argument("--device", default="cuda")
    distill_smoke.add_argument("--seed", type=int, default=0)
    distill_smoke.add_argument("--image-size", type=int, default=192)
    distill_smoke.add_argument("--temporal-stride", type=int, default=4)
    distill_eval = sub.add_parser(
        "evaluate-qc-student",
        help="用人工 Gold Set 搜索每类高精度阈值并决定是否允许自动拒收",
    )
    distill_eval.add_argument("--predictions", type=Path, required=True)
    distill_eval.add_argument("--gold-labels", type=Path, required=True)
    distill_eval.add_argument("--task-config", type=Path, default=Path("config/visual_model_tasks.json"))
    distill_eval.add_argument("--output", type=Path, required=True)
    research_eval = sub.add_parser(
        "evaluate-qc-research",
        help="按论文协议在 validation 冻结阈值，并在隔离 test 上报告置信区间和 worst-group",
    )
    research_eval.add_argument("--validation-predictions", type=Path, required=True)
    research_eval.add_argument("--validation-gold", type=Path, required=True)
    research_eval.add_argument("--test-predictions", type=Path, required=True)
    research_eval.add_argument("--test-gold", type=Path, required=True)
    research_eval.add_argument(
        "--task-config", type=Path, default=Path("config/visual_model_tasks.json")
    )
    research_eval.add_argument("--output", type=Path, required=True)
    research_eval.add_argument(
        "--group-field",
        action="append",
        dest="group_fields",
        help="重复传入以指定切片字段；默认 supplier_id/camera_id/source_dataset",
    )
    research_eval.add_argument("--bootstrap-replicates", type=int, default=1000)
    research_eval.add_argument("--minimum-group-samples", type=int, default=30)
    research_eval.add_argument("--seed", type=int, default=20260819)
    clip_plan = sub.add_parser(
        "plan-qc-clips",
        help="把逐帧异常自动合并为 4–8 秒视觉模型候选片段",
    )
    clip_plan.add_argument("dataset", type=Path)
    clip_plan.add_argument("--quality-root", type=Path, required=True)
    clip_plan.add_argument("--output", type=Path, required=True)
    clip_plan.add_argument(
        "--task-config",
        type=Path,
        default=Path("config/visual_model_tasks.json"),
    )
    clip_plan.add_argument("--video-key", default="observation.images.ego")
    clip_plan.add_argument("--minimum-s", type=float, default=4.0)
    clip_plan.add_argument("--maximum-s", type=float, default=8.0)
    clip_plan.add_argument("--context-s", type=float, default=1.5)
    clip_plan.add_argument("--merge-gap-s", type=float, default=1.0)
    clip_plan.add_argument("--control-ratio", type=float, default=0.25)
    clip_plan.add_argument("--minimum-control-clips", type=int, default=8)
    clip_plan.add_argument("--maximum-clips", type=int)
    clip_plan.add_argument("--seed", type=int, default=17)
    clip_plan.add_argument(
        "--source-class",
        choices=("public_dataset", "supplier_dataset", "internal_dataset"),
        help="数据治理类型；默认根据 supplier-id 推断",
    )
    clip_plan.add_argument("--source-dataset")
    clip_plan.add_argument("--supplier-id")
    queue_merge = sub.add_parser(
        "merge-teacher-queues",
        help="按数据源和召回类型轮转合并、去重视觉教师队列",
    )
    queue_merge.add_argument("queues", type=Path, nargs="+")
    queue_merge.add_argument("--output", type=Path, required=True)
    queue_merge.add_argument("--maximum-requests", type=int)
    queue_merge.add_argument("--seed", type=int, default=17)
    queue_normalize = sub.add_parser(
        "normalize-teacher-queue",
        help="为旧教师队列补齐 raw 溯源、数据治理字段和 split_group",
    )
    queue_normalize.add_argument("--queue", type=Path, required=True)
    queue_normalize.add_argument("--output", type=Path, required=True)
    queue_normalize.add_argument(
        "--source-class",
        choices=("public_dataset", "supplier_dataset", "internal_dataset"),
    )
    queue_normalize.add_argument("--source-dataset")
    queue_normalize.add_argument("--supplier-id")
    partition = sub.add_parser(
        "freeze-training-partition",
        help="按原视频组冻结 teacher-train、Gold validation 与 Gold test，阻断数据泄漏",
    )
    partition.add_argument("--teacher-queue", type=Path, required=True)
    partition.add_argument("--gold-events", type=Path, required=True)
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--validation-fraction", type=float, default=0.5)
    partition.add_argument("--seed", type=int, default=29)
    overlay_queue = sub.add_parser(
        "build-overlay-teacher-queue",
        help="把 Gold MANO 叠加短片绑定到教师请求，保留 raw 溯源",
    )
    overlay_queue.add_argument("--queue", type=Path, required=True)
    overlay_queue.add_argument("--review-events", type=Path, required=True)
    overlay_queue.add_argument("--output", type=Path, required=True)
    queue_gold = sub.add_parser(
        "build-queue-gold-review",
        help="从多源教师候选分层生成 4–8 秒人工 Gold 复检任务",
    )
    queue_gold.add_argument("--queue", type=Path, required=True)
    queue_gold.add_argument("--output", type=Path, required=True)
    queue_gold.add_argument("--maximum-events", type=int, default=180)
    queue_gold.add_argument("--seed", type=int, default=23)
    queue_gold.add_argument("--materialize-media", action="store_true")
    queue_gold.add_argument("--workers", type=int, default=8)
    queue_gold_overlay = sub.add_parser(
        "render-queue-gold-overlays",
        help="为 clip 级 Gold 任务批量渲染 MANO mesh 和骨骼参考视图",
    )
    queue_gold_overlay.add_argument("--events", type=Path, required=True)
    queue_gold_overlay.add_argument("--output", type=Path, required=True)
    queue_gold_overlay.add_argument("--hawor-root", type=Path, required=True)
    queue_gold_overlay.add_argument("--mano-data-root", type=Path)
    queue_gold_overlay.add_argument("--workers", type=int, default=4)
    queue_gold_overlay.add_argument("--mano-alpha", type=float, default=0.48)
    queue_gold_overlay.add_argument("--maximum-events", type=int)
    adapter_clip_plan = sub.add_parser(
        "plan-adapter-clips",
        help="从 EgoDex 等只读 adapter 样本生成低成本视觉教师队列",
    )
    adapter_clip_plan.add_argument("dataset", type=Path)
    adapter_clip_plan.add_argument("--episode", required=True)
    adapter_clip_plan.add_argument("--output", type=Path, required=True)
    adapter_clip_plan.add_argument(
        "--task-config", type=Path, default=Path("config/visual_model_tasks.json")
    )
    egodex_candidates = sub.add_parser(
        "build-egodex-candidates",
        help="按任务均匀抽样 EgoDex，并生成 clean/review/hard-negative 弱标签候选集",
    )
    egodex_candidates.add_argument("dataset", type=Path)
    egodex_candidates.add_argument("--output", type=Path, required=True)
    egodex_candidates.add_argument("--episodes-per-task", type=int, default=2)
    egodex_candidates.add_argument("--seed", type=int, default=17)
    egodex_candidates.add_argument("--workers", type=int, default=8)
    egodex_candidates.add_argument("--confidence-threshold", type=float, default=0.5)
    egodex_candidates.add_argument("--minimum-duration-s", type=float, default=5.0)
    egodex_candidates.add_argument("--minimum-fps", type=float, default=29.97)
    egodex_candidates.add_argument("--minimum-width", type=int, default=1280)
    egodex_candidates.add_argument("--minimum-height", type=int, default=720)
    egodex_candidates.add_argument("--maximum-absence-s", type=float, default=1.0)
    egodex_candidates.add_argument("--clean-quantile", type=float, default=0.67)
    egodex_candidates.add_argument("--hard-negative-quantile", type=float, default=0.20)
    egodex_candidates.add_argument(
        "--resume", action="store_true", help="复用 output 中已完成 episode，只处理新增抽样"
    )
    egodex_review = sub.add_parser(
        "build-egodex-review-batch",
        help="从分层画像生成任务均衡的人工审核与视觉教师队列，不解码或复制源视频",
    )
    egodex_review.add_argument("--profiles", type=Path, required=True)
    egodex_review.add_argument("--output", type=Path, required=True)
    egodex_review.add_argument("--dataset", type=Path)
    egodex_review.add_argument(
        "--task-config", type=Path, default=Path("config/visual_model_tasks.json")
    )
    egodex_review.add_argument("--window-s", type=float, default=6.0)
    egodex_review.add_argument("--maximum-clean", type=int)
    egodex_review.add_argument("--maximum-hard-negative", type=int)
    egodex_review.add_argument("--maximum-review", type=int, default=128)
    egodex_review.add_argument("--maximum-hand-absence", type=int, default=0)
    egodex_review.add_argument("--seed", type=int, default=17)
    discover_sources = sub.add_parser(
        "discover-lerobot-roots",
        help="任务均衡发现 task/dataset LeRobot 根，清单只写 workspace",
    )
    discover_sources.add_argument("source_root", type=Path)
    discover_sources.add_argument("--output", type=Path, required=True)
    discover_sources.add_argument("--maximum-per-task", type=int, default=2)
    discover_sources.add_argument(
        "--maximum-tasks",
        type=int,
        help="按 seed 确定性抽取最多多少个任务目录；默认全部任务",
    )
    discover_sources.add_argument("--workers", type=int, default=16)
    discover_sources.add_argument("--seed", type=int, default=17)
    task_taxonomy = sub.add_parser(
        "classify-tasks",
        help="对唯一任务文本做可复现多标签分类，并保留 unknown 供语义复核",
    )
    task_taxonomy.add_argument("--input", type=Path, required=True)
    task_taxonomy.add_argument("--output", type=Path, required=True)
    task_taxonomy.add_argument("--task-field", required=True)
    task_taxonomy.add_argument("--source-id", required=True)
    task_taxonomy.add_argument(
        "--taxonomy", type=Path, default=Path("config/task_taxonomy.json")
    )
    egodex_candidates.add_argument(
        "--full-profile", action="store_true",
        help="加载完整关节变换；默认仅读 confidence 和视频头以降低 OSS I/O",
    )
    egodex_candidates.add_argument("--checkpoint-every", type=int, default=25)
    egodex_candidates.add_argument(
        "--inventory-cache", type=Path,
        help="workspace 中可复用的 OSS 文件清单；默认 <output>/inventory.jsonl",
    )
    egodex_candidates.add_argument(
        "--refresh-inventory", action="store_true", help="重新枚举源目录并刷新 inventory"
    )
    adapter_clip_plan.add_argument("--window-s", type=float, default=6.0)
    adapter_clip_plan.add_argument("--maximum-clips", type=int, default=3)
    adapter_clip_plan.add_argument("--confidence-threshold", type=float, default=0.5)
    adapter_clip_plan.add_argument(
        "--visual-source", type=Path,
        help="可选：使用骨骼/mesh 叠加视频作为教师证据，同时保留 raw source 溯源",
    )
    teacher_run = sub.add_parser(
        "run-teacher-api",
        help="断点执行视觉教师队列；密钥只从环境变量读取",
    )
    teacher_run.add_argument("--queue", type=Path, required=True)
    teacher_run.add_argument("--output", type=Path, required=True)
    teacher_run.add_argument(
        "--provider",
        choices=("openai-compatible", "bailian"),
        default="openai-compatible",
    )
    teacher_run.add_argument(
        "--region",
        choices=("beijing", "singapore", "virginia"),
        help="百炼 Key 所属地域，默认读取 BAILIAN_REGION 或使用 beijing",
    )
    teacher_run.add_argument("--workspace-id", help="可选；百炼业务空间 ID")
    teacher_run.add_argument("--base-url", help="默认读取 TEACHER_API_BASE_URL")
    teacher_run.add_argument("--model", help="默认读取 TEACHER_API_MODEL")
    teacher_run.add_argument("--api-key-env")
    teacher_run.add_argument("--dry-run", action="store_true")
    teacher_run.add_argument("--overwrite", action="store_true")
    teacher_run.add_argument("--no-response-format", action="store_true")
    teacher_run.add_argument("--concurrency", type=int, default=2)
    teacher_run.add_argument(
        "--cost-profile",
        choices=("low", "balanced", "quality"),
        default="low",
    )
    teacher_run.add_argument("--sample-fps", type=float)
    teacher_run.add_argument("--max-frames", type=int)
    teacher_run.add_argument("--max-edge", type=int)
    teacher_run.add_argument("--jpeg-quality", type=int)
    teacher_run.add_argument("--timeout-s", type=float, default=120.0)
    teacher_run.add_argument("--max-retries", type=int, default=3)
    teacher_run.add_argument("--max-requests", type=int)
    teacher_run.add_argument(
        "--allow-external-supplier-data",
        action="store_true",
        help="明确授权将供应商视频的抽样帧发送给外部教师 API",
    )
    teacher_run.add_argument("--input-price-per-million", type=float)
    teacher_run.add_argument("--output-price-per-million", type=float)
    undistort_plan = sub.add_parser(
        "plan-vitra-undistortion",
        help="按 Microsoft VITRA 官方几何约定生成只读去畸变任务清单",
    )
    undistort_plan.add_argument("--dataset-kind", choices=("ego4d", "egoexo4d"), required=True)
    undistort_plan.add_argument("--video-root", type=Path, required=True)
    undistort_plan.add_argument("--intrinsics-root", type=Path, required=True)
    undistort_plan.add_argument("--save-root", type=Path, required=True)
    undistort_plan.add_argument("--output", type=Path, required=True)
    undistort_plan.add_argument("--selection-list", type=Path)
    undistort_plan.add_argument("--aria-name-map", type=Path)
    undistort_run = sub.add_parser(
        "run-vitra-undistortion",
        help="按固定 commit 调用 Microsoft VITRA 官方实现，支持断点和稳定分片",
    )
    undistort_run.add_argument("--manifest", type=Path, required=True)
    undistort_run.add_argument("--vitra-root", type=Path, required=True)
    undistort_run.add_argument("--output", type=Path, required=True)
    undistort_run.add_argument("--shard-index", type=int, default=0)
    undistort_run.add_argument("--shard-count", type=int, default=1)
    undistort_run.add_argument("--max-tasks", type=int)
    undistort_run.add_argument("--batch-size", type=int, default=2_000_000_000)
    undistort_run.add_argument("--crf", type=int, default=22)
    undistort_run.add_argument("--overwrite", action="store_true")
    undistort_verify = sub.add_parser(
        "verify-vitra-undistortion",
        help="严格核验 VITRA 去畸变源/产物可读性、帧数、FPS和输出尺寸",
    )
    undistort_verify.add_argument("--manifest", type=Path, required=True)
    undistort_verify.add_argument("--output", type=Path, required=True)
    egodex_overlay = sub.add_parser(
        "render-egodex-overlay",
        help="按 Apple EgoDex 官方坐标变换流式渲染双手骨骼回投视频",
    )
    egodex_overlay.add_argument("dataset", type=Path)
    egodex_overlay.add_argument("--episode", required=True)
    egodex_overlay.add_argument("--output", type=Path, required=True)
    egodex_overlay.add_argument("--start-frame", type=int, default=0)
    egodex_overlay.add_argument("--max-frames", type=int, default=300)
    egodex_overlay.add_argument("--stride", type=int, default=1)
    gold_review = sub.add_parser(
        "build-phase-a-review-events",
        help="把 Phase A 基线证据和派生视频接入 PostgreSQL Gold 复检面板",
    )
    gold_review.add_argument("dataset", type=Path)
    gold_review.add_argument("--baseline-evidence", type=Path, required=True)
    gold_review.add_argument("--output", type=Path, required=True)
    gold_review.add_argument("--annotated-root", type=Path)
    gold_review.add_argument("--video-key", default="observation.images.ego")
    gold_review.add_argument(
        "--no-materialize-media",
        action="store_true",
        help="只生成事件，不从聚合 MP4 派生 episode 短视频（仅调试）",
    )
    serve = sub.add_parser("serve-review", help="启动可滚动更新的本地人工复检 Web 服务")
    serve.add_argument("--evidence-root", type=Path, required=True)
    serve.add_argument("--quality-root", type=Path, required=True)
    serve.add_argument("--registry", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    review_db = sub.add_parser("init-review-db", help="初始化 PostgreSQL 人工复检表")
    review_db.add_argument("--database-url", help="默认读取 DATABASE_URL")
    review_import = sub.add_parser("import-review-events", help="幂等导入异常事件到 PostgreSQL")
    review_import.add_argument("events", type=Path)
    review_import.add_argument("--dataset-name", required=True)
    review_import.add_argument("--run-name", required=True)
    review_import.add_argument("--source-root")
    review_import.add_argument("--detector-json", type=Path)
    review_import.add_argument("--database-url", help="默认读取 DATABASE_URL")
    review_serve = sub.add_parser("serve-postgres-review", help="启动 PostgreSQL 动态多人复检服务")
    review_serve.add_argument("--evidence-root", type=Path, required=True)
    review_serve.add_argument("--database-url", help="默认读取 DATABASE_URL")
    review_serve.add_argument("--host", default="127.0.0.1")
    review_serve.add_argument("--port", type=int, default=8767)
    urdf = sub.add_parser("inspect-urdf", help="检查左右手 URDF 的 Robot DOF、限位和资源")
    urdf.add_argument("--left", type=Path, required=True)
    urdf.add_argument("--right", type=Path, required=True)
    urdf.add_argument("--expected-dof", type=int, default=20)
    urdf.add_argument("--output", type=Path)
    render = sub.add_parser("render-robot20", help="用 URDF FK 渲染 Robot20 STL mesh")
    render.add_argument("--urdf", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--pose", choices=("neutral", "midrange"), default="neutral")
    render.add_argument("--q", help="逗号分隔的 20 个关节角（弧度），提供后覆盖 --pose")
    render.add_argument("--width", type=int, default=900)
    render.add_argument("--height", type=int, default=900)
    render.add_argument("--max-triangles", type=int)
    register = sub.add_parser("register", help="登记挂载目录中的显式 LeRobot 数据集")
    register.add_argument("datasets", type=Path, nargs="*")
    register.add_argument(
        "--dataset-list",
        type=Path,
        help="UTF-8 文本文件，每行一个数据集根目录；空行和 # 注释会忽略",
    )
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--source", required=True, help="稳定的数据源名称，例如 oss-prod")
    register.add_argument(
        "--source-root",
        type=Path,
        help="挂载根目录；用于生成不受挂载点变化影响的逻辑路径",
    )
    register.add_argument("--video-key", default="observation.images.ego")
    register.add_argument(
        "--require-marker",
        help="只登记包含该封板标记的数据集，例如 _SUCCESS",
    )
    plan = sub.add_parser("plan", help="为新增或变化的数据集生成增量任务 manifest")
    plan.add_argument("--registry", type=Path, required=True)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--config", type=Path, default=default_config)
    plan.add_argument("--dataset-id", action="append", dest="dataset_ids")
    execute = sub.add_parser("run-manifest", help="幂等执行挂载目录任务 manifest")
    execute.add_argument("--registry", type=Path, required=True)
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--config", type=Path, default=default_config)
    execute.add_argument("--hash-mode", choices=("none", "headtail"), default="none")
    execute.add_argument("--results", type=Path)
    execute.add_argument("--continue-on-error", action="store_true")
    execute.add_argument("--cache-root", type=Path)
    execute.add_argument("--worker-id")
    execute.add_argument("--lease-seconds", type=int, default=3600)
    execute.add_argument("--workers", type=int, default=1)
    execute.add_argument("--progress", action="store_true", help="向 stderr 输出实时 JSONL 进度和 ETA")
    estimate = sub.add_parser("estimate", help="根据 manifest、历史吞吐和缓存估算处理时间")
    estimate.add_argument("--registry", type=Path, required=True)
    estimate.add_argument("--manifest", type=Path, required=True)
    estimate.add_argument("--config", type=Path, default=default_config)
    estimate.add_argument("--workers", type=int, default=1)
    tune = sub.add_parser("tune", help="生成离线交互式阈值调优页面")
    tune.add_argument("--quality-root", type=Path, required=True, nargs="+")
    tune.add_argument("--config", type=Path, default=default_config)
    tune.add_argument("--output", type=Path, required=True)
    experiment = sub.add_parser(
        "build-experiment-evidence",
        help="生成可审计的 raw/MANO 错误与对照图片、视频和实验 manifest",
    )
    experiment.add_argument("--events", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--experiment-id", required=True)
    experiment.add_argument("--code-root", type=Path)
    experiment.add_argument("--config", type=Path)
    experiment.add_argument("--run-results", type=Path)
    experiment.add_argument("--maximum-rule-positive", type=int, default=12)
    experiment.add_argument("--maximum-clean-control", type=int, default=6)
    experiment.add_argument("--maximum-low-event-control", type=int, default=6)
    experiment.add_argument("--workers", type=int, default=4)
    experiment.add_argument("--seed", type=int, default=43)
    retry_plan = sub.add_parser("plan-retry", help="汇总失败 shard，生成去重重试计划")
    retry_plan.add_argument("--quality-root", type=Path, required=True, nargs="+")
    retry_plan.add_argument("--output", type=Path, required=True)
    status = sub.add_parser("status", help="查看 Registry 数据集和任务状态")
    status.add_argument("--registry", type=Path, required=True)
    diagnose = sub.add_parser("doctor", help="检查依赖、配置、挂载和 Registry")
    diagnose.add_argument("--config", type=Path, default=default_config)
    diagnose.add_argument("--registry", type=Path)
    diagnose.add_argument("--source-root", type=Path)
    diagnose.add_argument("--output-root", type=Path)
    smoke = sub.add_parser("self-test", help="生成合成数据并运行完整管线")
    smoke.add_argument("--config", type=Path, default=default_config)
    smoke.add_argument("--workdir", type=Path)
    dashboard = sub.add_parser("dashboard", help="生成全局 Registry 质量看板")
    dashboard.add_argument("--registry", type=Path, required=True)
    dashboard.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def database_url(value: Optional[str]) -> str:
        result = value or os.environ.get("DATABASE_URL")
        if not result:
            parser.error("需要 --database-url 或 DATABASE_URL")
        return result

    if args.command == "scan":
        config = load_config(args.config)
        if args.video_check:
            config.setdefault("video_check", {})["mode"] = args.video_check
        summary = run(
            args.dataset.expanduser(),
            args.output.expanduser(),
            config,
            args.hash_mode,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"报告: {(args.output / 'report.html').resolve()}")
    elif args.command == "extract-samples":
        mano_renderer = None
        if args.mano_data_root and not args.hawor_root:
            parser.error("--mano-data-root 必须与 --hawor-root 一起使用")
        if args.hawor_root:
            mano_renderer = ManoOverlayRenderer(
                HaworManoBackend(
                    args.hawor_root,
                    args.mano_data_root,
                ),
                alpha=args.mano_alpha,
            )
        summary = extract_samples(
            args.dataset.expanduser(),
            args.plan.expanduser(),
            args.output.expanduser(),
            args.video_key,
            mano_renderer=mano_renderer,
            annotated_root=args.annotated_root.expanduser() if args.annotated_root else None,
            seek=not args.no_seek,
            seek_margin_s=args.seek_margin_s,
            seek_min_frame=args.seek_min_frame,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "render-annotated-video":
        renderer = ManoOverlayRenderer(
            HaworManoBackend(args.hawor_root, args.mano_data_root),
            alpha=args.mano_alpha,
        )
        summary = render_annotated_episode(
            args.dataset,
            args.episode,
            args.output,
            renderer,
            video_key=args.video_key,
            batch_size=args.batch_size,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            review_labels=args.review_labels,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "inspect-episode":
        summary = inspect_episode(
            args.dataset,
            args.episode,
            load_config(args.config.expanduser()),
        )
        if args.output:
            write_json(args.output.expanduser(), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "temporal-plot":
        summary = write_temporal_plot(
            args.dataset,
            args.episode,
            args.output,
            load_config(args.config.expanduser()),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "repair-preview":
        if args.mano_data_root and not args.hawor_root:
            parser.error("--mano-data-root 必须与 --hawor-root 一起使用")
        config = load_config(args.config.expanduser())
        repair_config = config.setdefault("repair", {})
        position = repair_config.setdefault("position", {})
        rotation = repair_config.setdefault("rotation", {})
        if args.position_min_cutoff is not None:
            position["min_cutoff"] = args.position_min_cutoff
        if args.position_beta is not None:
            position["beta"] = args.position_beta
        if args.rotation_min_cutoff is not None:
            rotation["min_cutoff"] = args.rotation_min_cutoff
        if args.rotation_beta is not None:
            rotation["beta"] = args.rotation_beta
        if args.derivative_cutoff is not None:
            position["derivative_cutoff"] = args.derivative_cutoff
            rotation["derivative_cutoff"] = args.derivative_cutoff
        renderer = None
        if args.hawor_root:
            renderer = ManoOverlayRenderer(
                HaworManoBackend(args.hawor_root, args.mano_data_root),
                alpha=args.mano_alpha,
            )
        summary = write_repair_preview(
            args.dataset,
            args.episode,
            args.output,
            config,
            mano_renderer=renderer,
            video_key=args.video_key,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "inspect-adapter":
        adapter_config = load_config(args.config.expanduser())
        summary = inspect_adapter(
            args.dataset,
            args.episode,
            confidence_threshold=args.confidence_threshold,
            video_check=args.video_check,
            video_options=adapter_config.get("video_check", {}),
        )
        if args.output:
            write_json(args.output.expanduser(), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-generic-ego-views":
        generic_config = load_config(args.config.expanduser())
        summary = build_generic_ego_views(
            args.dataset,
            args.output,
            source_dataset=args.source_dataset,
            source_class=args.source_class,
            license_id=args.license_id,
            workers=args.workers,
            maximum_depth=args.maximum_depth,
            limit=args.limit,
            video_check=args.video_check,
            video_options=generic_config.get("video_check", {}),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "plan-completion":
        summary = plan_public_completion(
            args.dataset,
            load_config(args.config.expanduser()),
            args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-completion-overlay":
        summary = build_completion_overlay(
            args.dataset,
            args.plan,
            args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "screen-rekadaily-hands":
        summary = screen_rekadaily_hands(
            args.dataset.expanduser(),
            args.video_id,
            args.output.expanduser(),
            args.weights.expanduser(),
            sample_fps=args.sample_fps,
            confidence=args.confidence,
            batch_size=args.batch_size,
            device=args.device,
            resume=not args.no_resume,
            workers=args.workers,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-rekadaily-views":
        summary = build_rekadaily_training_views(
            args.dataset,
            args.output,
            materialized_only=args.materialized_only,
            hand_screen_root=args.hand_screen_root,
            mano_root=args.mano_root,
            alignment_root=args.alignment_root,
            minimum_duration_s=args.minimum_duration_s,
            projects=args.projects,
            limit=args.limit,
            license_id=args.license_id,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "smoke-vla-loader":
        summary = smoke_vla_loader(
            args.manifest,
            args.output,
            split=args.split,
            batch_size=args.batch_size,
            allow_technical_candidates=args.allow_technical_candidates,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "smoke-vla-train":
        summary = smoke_vla_train(
            args.manifest,
            args.output,
            split=args.split,
            batch_size=args.batch_size,
            steps=args.steps,
            learning_rate=args.learning_rate,
            allow_technical_candidates=args.allow_technical_candidates,
            device=args.device,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-qc-distillation":
        summary = build_distillation_manifest(
            args.records,
            args.task_config,
            args.output,
            hand_screen_root=args.hand_screen_root,
            teacher_root=args.teacher_root,
            gold_labels=args.gold_labels,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-teacher-training-pool":
        summary = build_teacher_training_pool(
            args.queue,
            args.task_config,
            args.output,
            minimum_overall_confidence=args.minimum_overall_confidence,
            minimum_task_confidence=args.minimum_task_confidence,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-synthetic-qc-training":
        summary = build_synthetic_qc_training(
            args.manifest,
            args.output,
            maximum_per_task=args.maximum_per_task,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "plan-qc-interventions":
        summary = plan_qc_interventions(
            args.dataset,
            args.intervention_config,
            args.output,
            episode_indices=args.episodes,
            families=args.families,
            maximum_episodes=args.maximum_episodes,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "run-qc-interventions":
        summary = run_qc_interventions(
            args.dataset,
            args.manifest,
            args.config,
            args.output,
            maximum_interventions=args.maximum_interventions,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "smoke-qc-student":
        summary = smoke_train_qc_student(
            args.manifest,
            args.output,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
            image_size=args.image_size,
            temporal_stride=args.temporal_stride,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "audit-qc-training":
        summary = audit_qc_training_data(
            args.manifest,
            args.task_config,
            args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "evaluate-qc-student":
        summary = evaluate_qc_predictions(
            args.predictions,
            args.gold_labels,
            args.task_config,
            args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "evaluate-qc-research":
        summary = evaluate_qc_research_protocol(
            args.validation_predictions,
            args.validation_gold,
            args.test_predictions,
            args.test_gold,
            args.task_config,
            args.output,
            group_fields=args.group_fields
            or ("supplier_id", "camera_id", "source_dataset"),
            bootstrap_replicates=args.bootstrap_replicates,
            minimum_group_samples=args.minimum_group_samples,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "plan-qc-clips":
        summary = plan_qc_clips(
            args.dataset,
            args.quality_root,
            args.output,
            args.task_config,
            video_key=args.video_key,
            minimum_s=args.minimum_s,
            maximum_s=args.maximum_s,
            context_s=args.context_s,
            merge_gap_s=args.merge_gap_s,
            control_ratio=args.control_ratio,
            minimum_control_clips=args.minimum_control_clips,
            maximum_clips=args.maximum_clips,
            seed=args.seed,
            source_class=args.source_class,
            source_dataset=args.source_dataset,
            supplier_id=args.supplier_id,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "run-teacher-api":
        summary = run_teacher_api(
            args.queue,
            args.output,
            provider=args.provider,
            region=args.region,
            workspace_id=args.workspace_id,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            response_format=not args.no_response_format,
            concurrency=args.concurrency,
            cost_profile=args.cost_profile,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            max_edge=args.max_edge,
            jpeg_quality=args.jpeg_quality,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            max_requests=args.max_requests,
            allow_external_supplier_data=args.allow_external_supplier_data,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "merge-teacher-queues":
        summary = merge_teacher_queues(
            args.queues,
            args.output,
            maximum_requests=args.maximum_requests,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "normalize-teacher-queue":
        summary = normalize_teacher_queue_provenance(
            args.queue,
            args.output,
            source_class=args.source_class,
            source_dataset=args.source_dataset,
            supplier_id=args.supplier_id,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "freeze-training-partition":
        summary = freeze_training_partition(
            args.teacher_queue,
            args.gold_events,
            args.output,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-overlay-teacher-queue":
        summary = build_overlay_teacher_queue(
            args.queue,
            args.review_events,
            args.output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-queue-gold-review":
        summary = build_queue_gold_review(
            args.queue,
            args.output,
            maximum_events=args.maximum_events,
            seed=args.seed,
            materialize_media=args.materialize_media,
            workers=args.workers,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "render-queue-gold-overlays":
        summary = render_queue_gold_overlays(
            args.events,
            args.output,
            args.hawor_root,
            mano_data_root=args.mano_data_root,
            workers=args.workers,
            alpha=args.mano_alpha,
            maximum_events=args.maximum_events,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "plan-adapter-clips":
        summary = plan_adapter_clips(
            args.dataset,
            args.episode,
            args.output,
            args.task_config,
            window_s=args.window_s,
            maximum_clips=args.maximum_clips,
            confidence_threshold=args.confidence_threshold,
            visual_source=args.visual_source,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-egodex-candidates":
        summary = build_egodex_training_candidates(
            args.dataset,
            args.output,
            episodes_per_task=args.episodes_per_task,
            seed=args.seed,
            workers=args.workers,
            confidence_threshold=args.confidence_threshold,
            minimum_duration_s=args.minimum_duration_s,
            minimum_fps=args.minimum_fps,
            minimum_width=args.minimum_width,
            minimum_height=args.minimum_height,
            maximum_absence_s=args.maximum_absence_s,
            clean_quantile=args.clean_quantile,
            hard_negative_quantile=args.hard_negative_quantile,
            resume=args.resume,
            fast_profile=not args.full_profile,
            checkpoint_every=args.checkpoint_every,
            inventory_cache=args.inventory_cache,
            refresh_inventory=args.refresh_inventory,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-egodex-review-batch":
        summary = build_egodex_review_batch(
            args.profiles,
            args.task_config,
            args.output,
            dataset=args.dataset,
            window_s=args.window_s,
            maximum_clean=args.maximum_clean,
            maximum_hard_negative=args.maximum_hard_negative,
            maximum_review=args.maximum_review,
            maximum_hand_absence=args.maximum_hand_absence,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "discover-lerobot-roots":
        summary = discover_lerobot_roots(
            args.source_root,
            args.output,
            maximum_per_task=args.maximum_per_task,
            maximum_tasks=args.maximum_tasks,
            workers=args.workers,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "classify-tasks":
        summary = classify_task_records(
            args.input,
            args.taxonomy,
            args.output,
            task_field=args.task_field,
            source_id=args.source_id,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "plan-vitra-undistortion":
        summary = plan_vitra_undistortion(
            args.dataset_kind,
            args.video_root,
            args.intrinsics_root,
            args.save_root,
            args.output,
            selection_list=args.selection_list,
            aria_name_map=args.aria_name_map,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "verify-vitra-undistortion":
        summary = verify_vitra_undistortion(args.manifest, args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "run-vitra-undistortion":
        summary = run_vitra_undistortion(
            args.manifest, args.vitra_root, args.output,
            shard_index=args.shard_index, shard_count=args.shard_count,
            max_tasks=args.max_tasks, batch_size=args.batch_size,
            crf=args.crf, overwrite=args.overwrite,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "render-egodex-overlay":
        summary = render_egodex_overlay(
            args.dataset, args.episode, args.output,
            start_frame=args.start_frame, max_frames=args.max_frames, stride=args.stride,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-phase-a-review-events":
        summary = build_phase_a_review_events(
            args.dataset,
            args.baseline_evidence,
            args.output,
            annotated_root=args.annotated_root,
            video_key=args.video_key,
            materialize_media=not args.no_materialize_media,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "serve-review":
        serve_review(
            args.evidence_root,
            args.quality_root,
            args.registry,
            args.host,
            args.port,
        )
    elif args.command == "init-review-db":
        ReviewStore(database_url(args.database_url)).init_schema()
        print(json.dumps({"ok": True, "schema": "review-v1"}, ensure_ascii=False))
    elif args.command == "import-review-events":
        store = ReviewStore(database_url(args.database_url))
        store.init_schema()
        detector = load_config(args.detector_json.expanduser()) if args.detector_json else None
        summary = store.import_events(
            load_event_file(args.events.expanduser()),
            args.dataset_name,
            args.run_name,
            args.source_root,
            detector,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "serve-postgres-review":
        serve_postgres_review(
            database_url(args.database_url),
            args.evidence_root,
            args.host,
            args.port,
        )
    elif args.command == "inspect-urdf":
        summary = inspect_robot_pair(args.left, args.right, args.expected_dof)
        if args.output:
            write_json(args.output.expanduser(), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["ok"]:
            raise SystemExit(2)
    elif args.command == "render-robot20":
        q = None
        if args.q:
            q = [float(value.strip()) for value in args.q.split(",")]
        summary = render_robot(
            args.urdf,
            args.output,
            q=q,
            preset=args.pose,
            width=args.width,
            height=args.height,
            max_triangles=args.max_triangles,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "register":
        datasets = [path.expanduser() for path in args.datasets]
        if args.dataset_list:
            datasets.extend(
                Path(line.strip()).expanduser()
                for line in args.dataset_list.expanduser().read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        if not datasets:
            parser.error("register 至少需要一个数据集路径或 --dataset-list")
        summary = register_datasets(
            args.registry.expanduser(),
            datasets,
            args.source,
            args.video_key,
            args.source_root.expanduser() if args.source_root else None,
            args.require_marker,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "plan":
        config = load_config(args.config.expanduser())
        summary = create_manifest(
            args.registry.expanduser(),
            args.manifest.expanduser(),
            args.output_root.expanduser(),
            config,
            args.dataset_ids,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "run-manifest":
        def progress_line(event: Dict[str, Any]) -> None:
            print(json.dumps({"type": "progress", **event}, ensure_ascii=False), file=sys.stderr, flush=True)

        summary = run_manifest(
            args.registry.expanduser(),
            args.manifest.expanduser(),
            load_config(args.config.expanduser()),
            args.hash_mode,
            args.results.expanduser() if args.results else None,
            args.continue_on_error,
            args.cache_root.expanduser() if args.cache_root else None,
            args.worker_id,
            args.lease_seconds,
            args.workers,
            progress_line if args.progress else None,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "estimate":
        summary = estimate_manifest(
            args.registry.expanduser(),
            args.manifest.expanduser(),
            load_config(args.config.expanduser()),
            args.workers,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "tune":
        summary = write_tuner(
            [path.expanduser() for path in args.quality_root],
            load_config(args.config.expanduser()),
            args.output.expanduser(),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "build-experiment-evidence":
        summary = build_experiment_evidence(
            args.events,
            args.output,
            experiment_id=args.experiment_id,
            code_root=args.code_root,
            config_path=args.config,
            run_results_path=args.run_results,
            maximum_rule_positive=args.maximum_rule_positive,
            maximum_clean_control=args.maximum_clean_control,
            maximum_low_event_control=args.maximum_low_event_control,
            workers=args.workers,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "plan-retry":
        summary = create_retry_plan(
            [path.expanduser() for path in args.quality_root],
            args.output.expanduser(),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "status":
        summary = registry_status(args.registry.expanduser())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "doctor":
        summary = doctor(
            load_config(args.config.expanduser()),
            args.registry.expanduser() if args.registry else None,
            args.source_root.expanduser() if args.source_root else None,
            args.output_root.expanduser() if args.output_root else None,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["ok"]:
            raise SystemExit(2)
    elif args.command == "self-test":
        config = load_config(args.config.expanduser())
        if args.workdir:
            summary = self_test(args.workdir.expanduser(), config)
        else:
            with tempfile.TemporaryDirectory(prefix="egoqc-self-test-") as temporary:
                summary = self_test(Path(temporary), config)
                summary["workdir"] = "temporary directory removed"
                summary["result_root"] = "temporary directory removed"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["ok"]:
            raise SystemExit(2)
    elif args.command == "dashboard":
        summary = write_registry_dashboard(
            args.registry.expanduser(),
            args.output.expanduser(),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
