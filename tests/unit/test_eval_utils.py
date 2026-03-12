# ABOUTME: Tests for shared evaluation utilities
# ABOUTME: Covers checkpoint save/load, SLURM detection, sharding, completion markers

import os
import pickle
import sys
import pytest
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


class TestModelShortName:
    def test_8b_model(self):
        from eval_utils import model_short_name
        assert model_short_name("meta-llama/Llama-3.1-8B-Instruct") == "llama_3.1_8b_instruct"

    def test_70b_model(self):
        from eval_utils import model_short_name
        assert model_short_name("meta-llama/Llama-3.1-70B-Instruct") == "llama_3.1_70b_instruct"


class TestShardSamples:
    def test_even_split(self):
        from eval_utils import shard_samples
        items = list(range(8))
        shard0 = shard_samples(items, 0, 4)
        shard1 = shard_samples(items, 1, 4)
        shard2 = shard_samples(items, 2, 4)
        shard3 = shard_samples(items, 3, 4)
        assert shard0 == [0, 4]
        assert shard1 == [1, 5]
        assert shard2 == [2, 6]
        assert shard3 == [3, 7]

    def test_no_overlap(self):
        from eval_utils import shard_samples
        items = list(range(10))
        all_items = []
        for i in range(4):
            all_items.extend(shard_samples(items, i, 4))
        assert sorted(all_items) == items

    def test_single_gpu(self):
        from eval_utils import shard_samples
        items = list(range(5))
        assert shard_samples(items, 0, 1) == items


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        from eval_utils import save_checkpoint, load_checkpoint
        path = str(tmp_path / "test.pkl")
        results = [{"context_id": 1, "data": "a"}, {"context_id": 2, "data": "b"}]
        completed = {1, 2}
        save_checkpoint(path, results, completed)
        loaded_results, loaded_completed = load_checkpoint(path)
        assert loaded_results == results
        assert loaded_completed == completed

    def test_load_nonexistent(self, tmp_path):
        from eval_utils import load_checkpoint
        path = str(tmp_path / "nonexistent.pkl")
        results, completed = load_checkpoint(path)
        assert results == []
        assert completed == set()

    def test_load_corrupt(self, tmp_path):
        from eval_utils import load_checkpoint
        path = str(tmp_path / "corrupt.pkl")
        with open(path, "wb") as f:
            f.write(b"not a pickle")
        results, completed = load_checkpoint(path)
        assert results == []
        assert completed == set()

    def test_atomic_save(self, tmp_path):
        """Checkpoint file should exist after save (no partial writes)."""
        from eval_utils import save_checkpoint
        path = str(tmp_path / "atomic.pkl")
        save_checkpoint(path, [{"a": 1}], {1})
        assert os.path.exists(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert data["results"] == [{"a": 1}]


class TestCompletionMarker:
    def test_mark_complete_creates_file(self, tmp_path):
        from eval_utils import mark_complete
        mark_complete(str(tmp_path), "sanity_check", "llama_3.1_8b_instruct", 0, 4)
        expected = tmp_path / "sanity_check_eval_llama_3.1_8b_instruct_gpu0_of_4_COMPLETE"
        assert expected.exists()

    def test_marker_contains_timestamp(self, tmp_path):
        from eval_utils import mark_complete
        mark_complete(str(tmp_path), "main", "llama_3.1_8b_instruct", 2, 4)
        marker = tmp_path / "main_eval_llama_3.1_8b_instruct_gpu2_of_4_COMPLETE"
        content = marker.read_text()
        assert float(content) > 0


class TestPathGeneration:
    def test_result_path(self):
        from eval_utils import result_path
        p = result_path("/results", "sanity_check", "llama_3.1_8b_instruct", 0, 4)
        assert p == "/results/sanity_check_eval_llama_3.1_8b_instruct_gpu0_of_4.json"

    def test_checkpoint_path(self):
        from eval_utils import checkpoint_path
        p = checkpoint_path("/ckpt", "main", "llama_3.1_70b_instruct", 3, 4)
        assert p == "/ckpt/main_eval_llama_3.1_70b_instruct_gpu3_of_4_checkpoint.pkl"


class TestDetectSlurm:
    def test_no_slurm(self, monkeypatch):
        from eval_utils import detect_slurm
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)
        gpu_id, total = detect_slurm()
        assert gpu_id is None
        assert total is None

    def test_with_slurm(self, monkeypatch):
        from eval_utils import detect_slurm
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")
        gpu_id, total = detect_slurm()
        assert gpu_id == 2
        assert total == 4


class TestEdgeCases:
    def test_shard_empty_list(self):
        """Sharding an empty list returns empty list."""
        from eval_utils import shard_samples
        assert shard_samples([], 0, 4) == []

    def test_shard_more_gpus_than_items(self):
        """When more GPUs than items, some shards are empty."""
        from eval_utils import shard_samples
        items = [0, 1]
        assert shard_samples(items, 0, 4) == [0]
        assert shard_samples(items, 1, 4) == [1]
        assert shard_samples(items, 2, 4) == []
        assert shard_samples(items, 3, 4) == []

    def test_save_checkpoint_creates_parent_dirs(self, tmp_path):
        """save_checkpoint creates parent directories if needed."""
        from eval_utils import save_checkpoint
        path = str(tmp_path / "nested" / "dir" / "ckpt.pkl")
        save_checkpoint(path, [{"a": 1}], {1})
        assert os.path.exists(path)
