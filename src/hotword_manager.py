"""
热词管理器（默认词表 + 预编码缓存 + 运行时热更新）

职责：
1. 加载服务端默认词表（models/asr/hotwords.txt）
2. 校验链：格式 / 数量 / 可编码 / 试编码
3. 路径判定：默认词表恒走路径 B（Faiss 保守纠错，防通用识别误触发）；
   路径 A（SeACo）只保留给客户端主动传的 hotwords（见 main._encode_hotwords）
4. 运行时热更新（多 worker 安全）：
   - flock 跨进程互斥
   - expected_version 乐观并发（CAS）
   - 原子写 temp → fsync → os.rename，commit point = version 文件
   - 各 worker 后台轮询 version 文件，发现变更重建本地缓存（最终一致）
   - 原子切换引用（GIL 保证），在途请求零中断
   - 保留上一版引用，支持回滚

部署形态：单机单容器多 worker（WORKERS=N），所有 worker 共享容器本地
hotwords.txt + hotwords.version 文件，无需挂载/NFS/K8s。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np

from src.config import settings
from src.logger import logger

# Windows 无 fcntl，跨进程锁降级为进程内线程锁（本地开发用）
try:
    import fcntl  # type: ignore
    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore
    _HAS_FCNTL = False


# ============================================================
# 缓存对象（只读，原子替换的单位）
# ============================================================
@dataclass(frozen=True)
class HotwordCache:
    """默认词表的一份不可变快照。原子切换的单位。"""
    version: int
    md5: str
    words: tuple[str, ...]          # 有效词条（去重 + 剔 OOV 后）
    route: str                       # "A"（SeACo 预编码）或 "B"（Faiss）
    bias_embed: np.ndarray | None    # route=A 时的预编码 bias_embed (1, N+1, 512)，否则 None
    loaded_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def count(self) -> int:
        return len(self.words)


class ValidationError(Exception):
    """词表校验失败。"""


class VersionConflict(Exception):
    """乐观并发版本冲突。"""


# ============================================================
# 热词管理器
# ============================================================
class HotwordManager:
    """默认词表加载 / 校验 / 预编码缓存 / 热更新 / 回滚。"""

    def __init__(self):
        self._path = settings.DEFAULT_HOTWORD_PATH
        self._version_path = self._path + ".version"  # hotwords.txt.version
        self._lock_path = os.path.join(os.path.dirname(self._path) or ".", ".hotwords.lock")

        # 当前缓存与上一版（原子切换 + 回滚）
        self._cache: HotwordCache | None = None
        self._prev_cache: HotwordCache | None = None

        # reload/构建过程的进程内互斥（与 flock 跨进程互斥配合）
        self._build_lock = threading.Lock()

        # 编码回调：注入 tokenizer.encode 和 asr_engine.encode_hotwords
        # 避免循环依赖，由 main 在启动时设置
        self._encode_fn: Callable[[str], list[int]] | None = None
        self._bias_encode_fn: Callable[[np.ndarray], np.ndarray | None] | None = None

        # 轮询线程控制
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None

    # --------------------------------------------------------
    # 依赖注入（启动时由 main 设置）
    # --------------------------------------------------------
    def set_encoders(
        self,
        encode_fn: Callable[[str], list[int]],
        bias_encode_fn: Callable[[np.ndarray], np.ndarray | None],
    ):
        self._encode_fn = encode_fn
        self._bias_encode_fn = bias_encode_fn

    # --------------------------------------------------------
    # 当前缓存访问（推理路由用）
    # --------------------------------------------------------
    @property
    def cache(self) -> HotwordCache | None:
        """当前默认词表缓存（原子读，无锁）。"""
        return self._cache

    def status(self) -> dict:
        """词表状态（用于 /hotwords/status 巡检）。"""
        c = self._cache
        if c is None:
            return {"version": 0, "md5": "", "count": 0, "route": None,
                    "loaded_at": None, "cache_ready": False}
        return {
            "version": c.version,
            "md5": c.md5,
            "count": c.count,
            "route": c.route,
            "loaded_at": c.loaded_at,
            "cache_ready": True,
        }

    # --------------------------------------------------------
    # 启动加载
    # --------------------------------------------------------
    def load_initial(self):
        """启动时加载默认词表并构建缓存。失败不阻塞服务启动。"""
        try:
            words = self._read_words_file()
            version = self._read_version()
            if version == 0:
                # 无 version 文件，初始化为 1 并写入
                version = 1
            cache = self._build_cache(words, version)
            self._cache = cache
            logger.info(
                f"默认词表加载成功: version={cache.version}, count={cache.count}, "
                f"route={cache.route}, md5={cache.md5[:8]}"
            )
            # 确保 version 文件存在
            if self._read_version() == 0:
                self._write_version_file(cache)
        except FileNotFoundError:
            logger.info(f"默认词表不存在（{self._path}），跳过默认热词加载")
            self._cache = None
        except Exception as e:
            logger.warning(f"默认词表加载失败: {e}，服务继续启动（无默认热词）")
            self._cache = None

    # --------------------------------------------------------
    # 校验链
    # --------------------------------------------------------
    def _validate_and_clean(self, raw_words: list[str]) -> tuple[list[str], list[str]]:
        """
        格式 + 可编码校验。返回 (有效词, 被剔除的 OOV 词)。

        - 去空白、去重、非空、UTF-8（list[str] 已是 str，编码隐含）
        - 每词过 tokenizer.encode，编码为空（全 OOV）的剔除
        """
        # 去空白 + 非空 + 去重（保序）
        seen = set()
        cleaned: list[str] = []
        for w in raw_words:
            w = (w or "").strip()
            if not w or w in seen:
                continue
            seen.add(w)
            cleaned.append(w)

        if not cleaned:
            raise ValidationError("词表为空（去空白去重后无有效词条）")

        # 可编码校验
        if self._encode_fn is None:
            # tokenizer 未注入（理论不应发生），跳过编码校验
            return cleaned, []

        valid: list[str] = []
        dropped: list[str] = []
        for w in cleaned:
            ids = self._encode_fn(w)
            if ids:
                valid.append(w)
            else:
                dropped.append(w)

        if dropped:
            logger.warning(f"词表剔除 {len(dropped)} 个无法编码(OOV)词条: {dropped[:10]}")

        if not valid:
            raise ValidationError("词表有效词条为 0（全部无法编码）")

        return valid, dropped

    def _determine_route(self, count: int) -> str:
        """
        默认词表路径判定：固定走 B（Faiss 保守后处理纠错）。

        设计依据（通用识别场景防误触发）：
            SeACo（路径 A）是"只要声学接近热词就强增强"，适合垂直场景
            （音频确定含热词）；但通用识别里绝大多数音频不含默认热词，
            SeACo 会把声学相似的普通词误纠成热词（如"神棚"→"沈鹏"）。
            默认词表统一走 Faiss，靠三重判定（拼音+编辑距离+区分度阈值）
            仅在高度吻合时才替换，大幅降低误触发。
        SeACo（路径 A）只保留给"客户端主动传 hotwords"（见 main._encode_hotwords）
            —— 客户端传热词 = 明确知道该音频含这些词，才需要激进增强。
        """
        return "B"

    def _build_cache(self, raw_words: list[str], version: int) -> HotwordCache:
        """构建缓存对象（含校验 + 路径判定 + 预编码）。"""
        valid, _dropped = self._validate_and_clean(raw_words)
        route = self._determine_route(len(valid))
        md5 = hashlib.md5("\n".join(valid).encode("utf-8")).hexdigest()

        bias_embed = None
        if route == "A":
            bias_embed = self._pre_encode(valid)
        elif settings.ENABLE_FAISS_CORRECTION:
            # route=B：构建 Faiss 纠错索引（懒导入，失败不阻塞）
            # ENABLE_FAISS_CORRECTION=false 时跳过构建，省内存与启动耗时
            try:
                from src.hotword_faiss import faiss_corrector
                faiss_corrector.build(list(valid), version)
            except Exception as e:
                logger.warning(f"Faiss 索引构建跳过: {e}")
        else:
            logger.info("ENABLE_FAISS_CORRECTION=false，跳过默认词表 Faiss 索引构建")

        return HotwordCache(
            version=version,
            md5=md5,
            words=tuple(valid),
            route=route,
            bias_embed=bias_embed,
        )

    def _pre_encode(self, words: list[str]) -> np.ndarray | None:
        """
        预编码默认词表为 bias_embed（route=A）。

        流程同 main._encode_hotwords：encode → 追加 [sos] 哨兵 → padding → bias_encoder。
        试编码校验：验 bias_embed 无 nan/inf。
        """
        if self._encode_fn is None or self._bias_encode_fn is None:
            logger.warning("编码器未注入，默认词表无法预编码 bias_embed")
            return None

        encoded = [self._encode_fn(w) for w in words]
        encoded = [ids for ids in encoded if ids]
        if not encoded:
            raise ValidationError("预编码失败：无有效 token 序列")

        # 追加 [sos]=[1] 哨兵（SeACo NO_BIAS 占位）
        encoded.append([1])

        max_len = max(len(ids) for ids in encoded)
        padded = np.zeros((len(encoded), max_len), dtype=np.int64)
        for i, ids in enumerate(encoded):
            padded[i, : len(ids)] = ids

        bias_embed = self._bias_encode_fn(padded)
        if bias_embed is None:
            raise ValidationError("预编码失败：bias_encoder 返回 None（热词模型不可用）")

        # 试编码校验
        if not np.isfinite(bias_embed).all():
            raise ValidationError("预编码失败：bias_embed 含 nan/inf")

        logger.info(f"默认词表预编码完成: bias_embed shape={bias_embed.shape}")
        return bias_embed

    # --------------------------------------------------------
    # 文件 IO（原子写 + 版本）
    # --------------------------------------------------------
    def _read_words_file(self) -> list[str]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(self._path)
        with open(self._path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _read_version(self) -> int:
        """读 version 文件的 version 字段，无则返回 0。"""
        if not os.path.exists(self._version_path):
            return 0
        try:
            with open(self._version_path, "r", encoding="utf-8") as f:
                return int(json.load(f).get("version", 0))
        except Exception:
            return 0

    def _atomic_write(self, path: str, content: str):
        """原子写：temp → fsync → os.rename（同目录 rename 原子）。"""
        d = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_hw_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # 原子替换
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _write_version_file(self, cache: HotwordCache):
        meta = {
            "version": cache.version,
            "md5": cache.md5,
            "count": cache.count,
            "route": cache.route,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._atomic_write(self._version_path, json.dumps(meta, ensure_ascii=False, indent=2))

    class _FileLock:
        """跨进程文件锁（flock），Windows 降级为 no-op（仅靠进程内线程锁）。"""
        def __init__(self, lock_path: str):
            self._lock_path = lock_path
            self._fp = None

        def __enter__(self):
            if not _HAS_FCNTL:
                return self
            self._fp = open(self._lock_path, "w")
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, *exc):
            if self._fp is not None:
                try:
                    fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
                finally:
                    self._fp.close()
                    self._fp = None

    # --------------------------------------------------------
    # 热更新（reload / rollback）
    # --------------------------------------------------------
    def reload(
        self,
        words: list[str] | None = None,
        reload_from_file: bool = False,
        expected_version: int | None = None,
    ) -> HotwordCache:
        """
        重载词表（本 worker 立即生效，其他 worker 轮询收敛）。

        参数：
            words: 新词表内容（与 reload_from_file 二选一）
            reload_from_file: 重读磁盘文件
            expected_version: 乐观并发，与当前磁盘 version 不符则抛 VersionConflict

        返回：新的 HotwordCache。校验失败抛 ValidationError，保留旧缓存。
        """
        with self._build_lock:
            with self._FileLock(self._lock_path):
                disk_version = self._read_version()

                # 乐观并发 CAS
                if expected_version is not None and expected_version != disk_version:
                    raise VersionConflict(
                        f"期望 version={expected_version}，当前磁盘 version={disk_version}"
                    )

                # 取新词表内容
                if reload_from_file:
                    new_words = self._read_words_file()
                elif words is not None:
                    new_words = words
                else:
                    raise ValidationError("reload 需提供 words 或 reload_from_file=true")

                new_version = disk_version + 1

                # 构建 + 校验（失败抛异常，旧缓存不动）
                new_cache = self._build_cache(new_words, new_version)

                # 持久化：先写词表，再写 version（version 是 commit point）
                self._atomic_write(self._path, "\n".join(new_cache.words) + "\n")
                self._write_version_file(new_cache)

                # 原子切换引用（GIL 保证）
                self._prev_cache = self._cache
                self._cache = new_cache

        logger.info(
            f"词表热更新成功: version={new_cache.version}, count={new_cache.count}, "
            f"route={new_cache.route}"
        )
        return new_cache

    def rollback(self) -> HotwordCache:
        """回滚到上一版（写回内容触发 version+1，各 worker 轮询收敛）。"""
        if self._prev_cache is None:
            raise ValidationError("无上一版本可回滚")
        prev = self._prev_cache
        # 用上一版内容重新 reload（version 会 +1，保证 version 单调递增触发收敛）
        return self.reload(words=list(prev.words))

    # --------------------------------------------------------
    # 后台轮询收敛（多 worker 最终一致）
    # --------------------------------------------------------
    def start_polling(self):
        """启动后台轮询线程，监听 version 文件变更并重建本地缓存。"""
        if not settings.HOTWORD_RELOAD_ENABLED:
            return
        if self._poll_thread is not None:
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="hotword_poller", daemon=True
        )
        self._poll_thread.start()
        logger.info(f"热词轮询启动: interval={settings.HOTWORD_POLL_INTERVAL}s")

    def stop_polling(self):
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2)
            self._poll_thread = None

    def _poll_loop(self):
        interval = settings.HOTWORD_POLL_INTERVAL
        while not self._poll_stop.wait(interval):
            try:
                disk_version = self._read_version()
                local_version = self._cache.version if self._cache else 0
                if disk_version > local_version:
                    # 磁盘有更新版本（其他 worker 写入），重建本地缓存
                    logger.info(
                        f"检测到词表更新: 磁盘 version={disk_version} > 本地 {local_version}，重建缓存"
                    )
                    # 持 _build_lock，避免与本 worker 的 reload 竞争
                    with self._build_lock:
                        # 双检：可能在等锁期间本 worker 已 reload 收敛
                        if disk_version <= (self._cache.version if self._cache else 0):
                            continue
                        words = self._read_words_file()
                        new_cache = self._build_cache(words, disk_version)
                        self._prev_cache = self._cache
                        self._cache = new_cache
                    logger.info(f"本地缓存已收敛至 version={disk_version}")
            except Exception as e:
                logger.warning(f"热词轮询重建失败: {e}")


# 全局单例
hotword_manager = HotwordManager()
