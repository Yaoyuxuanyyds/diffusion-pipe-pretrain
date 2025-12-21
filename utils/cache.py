import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import torch


@dataclass
class ShardInfo:
    path: Path
    length: int


class ShardCacheWriter:
    """
    A lightweight, append-only sharded cache writer.

    Key goals compared to the previous Cache:
    - No SQLite or extra processes.
    - Fully deterministic shards, each owned by the main process.
    - Simple manifest that can be memory-mapped and shared by dataloader workers.
    """

    def __init__(
        self,
        cache_dir: Path,
        fingerprint: str,
        shard_size: int = 512,
        existing_shards: Optional[List[ShardInfo]] = None,
        start_total: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.fingerprint = fingerprint
        self.shard_size = shard_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / "manifest.json"
        self.items: List[Any] = []
        self.shards: List[ShardInfo] = existing_shards or []
        self.total = start_total

    def add(self, item: Any):
        self.items.append(item)
        if len(self.items) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.items:
            return
        shard_id = len(self.shards)
        shard_path = self.cache_dir / f"shard_{shard_id:06d}.pt"
        torch.save(self.items, shard_path)
        self.shards.append(ShardInfo(path=shard_path, length=len(self.items)))
        self.total += len(self.items)
        self.items = []

    def finalize(self):
        self._flush()
        manifest = {
            "fingerprint": self.fingerprint,
            "total": self.total,
            "shard_size": self.shard_size,
            "shards": [
                {"path": shard.path.name, "length": shard.length}
                for shard in self.shards
            ],
        }
        with open(self.manifest_file, "w") as f:
            json.dump(manifest, f)


class ShardCache:
    """
    Reader for sharded caches created by ShardCacheWriter.
    Keeps a small LRU of loaded shards to minimize disk reads while keeping memory bounded.
    """

    def __init__(self, cache_dir: Path, expected_fingerprint: str, max_shards_in_memory: int = 2):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "manifest.json") as f:
            manifest = json.load(f)

        if manifest["fingerprint"] != expected_fingerprint:
            raise ValueError(
                f"Cache fingerprint mismatch: expected {expected_fingerprint}, got {manifest['fingerprint']}"
            )

        self.total = manifest["total"]
        self.shards: List[ShardInfo] = []
        for shard in manifest["shards"]:
            self.shards.append(
                ShardInfo(
                    path=self.cache_dir / shard["path"],
                    length=shard["length"],
                )
            )
        self.max_shards_in_memory = max_shards_in_memory
        self._shard_cache: OrderedDict[int, List[Any]] = OrderedDict()

    def __len__(self):
        return self.total

    def _load_shard(self, shard_id: int):
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            return self._shard_cache[shard_id]
        shard = torch.load(self.shards[shard_id].path, map_location="cpu")
        self._shard_cache[shard_id] = shard
        if len(self._shard_cache) > self.max_shards_in_memory:
            self._shard_cache.popitem(last=False)
        return shard

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        shard_id = 0
        offset = idx
        for i, shard in enumerate(self.shards):
            if offset < shard.length:
                shard_id = i
                break
            offset -= shard.length
        shard_data = self._load_shard(shard_id)
        return shard_data[offset]


def streaming_write(cache_dir: Path, fingerprint: str, data_iter: Iterable[Any], shard_size: int = 512):
    writer = ShardCacheWriter(cache_dir, fingerprint, shard_size=shard_size)
    for item in data_iter:
        writer.add(item)
    writer.finalize()


class Cache:
    """
    Backwards-compatible facade used by legacy dataset paths.
    Internally delegates to the new ShardCacheWriter/ShardCache.
    """

    def __init__(self, path: str, fingerprint: str, shard_size_gb: float = 1):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.shard_size = int(max(1, shard_size_gb * 1024))  # roughly align with old behavior
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                data = json.load(f)
            if data.get("fingerprint") != fingerprint:
                self.clear()
                self._init_empty()
            else:
                self.total = data["total"]
                self.shards = [
                    ShardInfo(path=self.path / shard["path"], length=shard["length"])
                    for shard in data["shards"]
                ]
                self.reader = ShardCache(self.path, fingerprint)
                self.writer = ShardCacheWriter(
                    self.path,
                    fingerprint,
                    shard_size=self.shard_size,
                    existing_shards=self.shards,
                    start_total=self.total,
                )
        else:
            self._init_empty()

    def _init_empty(self):
        self.total = 0
        self.shards: List[ShardInfo] = []
        self.reader: Optional[ShardCache] = None
        self.writer = ShardCacheWriter(self.path, self.fingerprint, shard_size=self.shard_size)

    def __len__(self):
        return self.total + len(self.writer.items)

    def __getitem__(self, idx):
        if self.reader is None:
            self.writer.finalize()
            self.reader = ShardCache(self.path, self.fingerprint)
        return self.reader[idx]

    def add(self, item: Any):
        self.writer.add(item)

    def finalize_current_shard(self):
        self.writer.finalize()
        self.total = self.writer.total
        self.shards = self.writer.shards
        self.reader = ShardCache(self.path, self.fingerprint)

    def clear(self):
        if self.path.exists():
            for f in self.path.glob("*"):
                if f.is_file():
                    f.unlink()
        self._init_empty()
