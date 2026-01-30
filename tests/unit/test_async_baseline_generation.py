# ABOUTME: Tests for async parallel baseline generation
# ABOUTME: Verifies concurrency control, correct results, and checkpoint behavior

import asyncio
import json
import sys
import time
import pytest

sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/src')


class TestAsyncBaselineGeneration:
    """Tests for async parallel baseline generation with concurrency control."""

    def test_generate_all_baselines_async_exists(self):
        """generate_all_baselines_async function must exist and be async."""
        from generate_baselines import generate_all_baselines_async
        assert asyncio.iscoroutinefunction(generate_all_baselines_async)

    def test_generate_baseline_async_exists(self):
        """generate_baseline_async function must exist and be async."""
        from generate_baselines import generate_baseline_async
        assert asyncio.iscoroutinefunction(generate_baseline_async)

    def test_concurrency_limited_to_max_concurrent(self, tmp_path):
        """Concurrent requests must not exceed max_concurrent parameter."""
        # Track peak concurrency during execution
        current_concurrent = 0
        peak_concurrent = 0
        lock = asyncio.Lock()

        async def mock_api_call(sample_id, ptype):
            nonlocal current_concurrent, peak_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > peak_concurrent:
                    peak_concurrent = current_concurrent

            # Simulate API latency
            await asyncio.sleep(0.05)

            async with lock:
                current_concurrent -= 1

            return {
                'paraphrase': f'paraphrase for {sample_id}_{ptype}',
                'target_pct': 10.0,
                'actual_pct': 10.1,
                'deviation': 0.1,
                'retries_used': 0,
                'skipped': False
            }

        from generate_baselines import generate_all_baselines_async

        # Mock tokenizer that returns different token counts for original vs perturbed
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                if 'Perturbed' in text:
                    return ['tok'] * 12  # 12 tokens for perturbed
                return ['tok'] * 10  # 10 tokens for original (20% change)

        # Create test data with enough samples to test concurrency
        import pandas as pd
        df = pd.DataFrame({
            'Index': list(range(50)),
            'clinical_context': [f'Patient {i} context' for i in range(50)],
            'gender_swap_text': [f'Perturbed {i}' for i in range(50)],
        })

        output_path = tmp_path / 'baselines.json'
        max_concurrent = 10

        # Run with mock
        async def run_test():
            return await generate_all_baselines_async(
                df=df,
                tokenizer=MockTokenizer(),
                output_path=str(output_path),
                max_concurrent=max_concurrent,
                _api_call_fn=mock_api_call  # Inject mock for testing
            )

        asyncio.run(run_test())

        assert peak_concurrent <= max_concurrent, (
            f"Peak concurrency {peak_concurrent} exceeded max {max_concurrent}"
        )
        # Should have used concurrency (not sequential)
        assert peak_concurrent > 1, "Should use multiple concurrent requests"

    def test_all_samples_processed(self, tmp_path):
        """All samples must be processed and included in output."""
        call_log = []

        async def mock_api_call(sample_id, ptype):
            call_log.append((sample_id, ptype))
            return {
                'paraphrase': f'paraphrase for {sample_id}_{ptype}',
                'target_pct': 10.0,
                'actual_pct': 10.1,
                'deviation': 0.1,
                'retries_used': 0,
                'skipped': False
            }

        from generate_baselines import generate_all_baselines_async

        # Mock tokenizer that returns different token counts for original vs perturbed
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                if 'pert' in text:
                    return ['tok'] * 12  # 12 tokens for perturbed
                return ['tok'] * 10  # 10 tokens for original (20% change)

        import pandas as pd
        df = pd.DataFrame({
            'Index': [10, 20, 30],
            'clinical_context': ['ctx A', 'ctx B', 'ctx C'],
            'gender_swap_text': ['pert A', 'pert B', 'pert C'],
        })

        output_path = tmp_path / 'baselines.json'

        async def run_test():
            return await generate_all_baselines_async(
                df=df,
                tokenizer=MockTokenizer(),
                output_path=str(output_path),
                max_concurrent=10,
                _api_call_fn=mock_api_call
            )

        result = asyncio.run(run_test())

        # Check all samples processed
        assert '10' in result
        assert '20' in result
        assert '30' in result
        assert result['10']['gender_swap']['paraphrase'] == 'paraphrase for 10_gender_swap'
        assert result['20']['gender_swap']['paraphrase'] == 'paraphrase for 20_gender_swap'
        assert result['30']['gender_swap']['paraphrase'] == 'paraphrase for 30_gender_swap'

    def test_checkpoint_saves_periodically(self, tmp_path):
        """Checkpoints must be saved periodically during processing."""
        checkpoint_saves = []

        async def mock_api_call(sample_id, ptype):
            return {
                'paraphrase': f'paraphrase for {sample_id}_{ptype}',
                'target_pct': 10.0,
                'actual_pct': 10.1,
                'deviation': 0.1,
                'retries_used': 0,
                'skipped': False
            }

        from generate_baselines import generate_all_baselines_async

        # Mock tokenizer that returns different token counts
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                if 'pert' in text:
                    return ['tok'] * 12
                return ['tok'] * 10

        import pandas as pd
        # 25 samples should trigger at least 2 checkpoint saves (at 10 and 20)
        df = pd.DataFrame({
            'Index': list(range(25)),
            'clinical_context': [f'ctx {i}' for i in range(25)],
            'gender_swap_text': [f'pert {i}' for i in range(25)],
        })

        output_path = tmp_path / 'baselines.json'

        # Track checkpoint saves by checking file modification
        original_json_dump = json.dump

        def tracking_json_dump(obj, f, **kwargs):
            checkpoint_saves.append(len(obj))
            return original_json_dump(obj, f, **kwargs)

        import generate_baselines
        original_dump = json.dump

        async def run_test():
            # Monkey-patch json.dump to track saves
            json.dump = tracking_json_dump
            try:
                return await generate_all_baselines_async(
                    df=df,
                    tokenizer=MockTokenizer(),
                    output_path=str(output_path),
                    max_concurrent=5,
                    checkpoint_freq=10,
                    _api_call_fn=mock_api_call
                )
            finally:
                json.dump = original_dump

        asyncio.run(run_test())

        # Should have saved checkpoints multiple times
        assert len(checkpoint_saves) >= 2, (
            f"Expected at least 2 checkpoint saves, got {len(checkpoint_saves)}"
        )

    def test_skips_small_perturbations(self, tmp_path):
        """Perturbations with <0.5% token change should be skipped."""
        api_calls = []

        async def mock_api_call(sample_id, ptype):
            api_calls.append((sample_id, ptype))
            return {
                'paraphrase': 'should not be called',
                'target_pct': 0.1,
                'actual_pct': 0.1,
                'deviation': 0.0,
                'retries_used': 0,
                'skipped': False
            }

        from generate_baselines import generate_all_baselines_async, compute_token_change_percent

        import pandas as pd
        df = pd.DataFrame({
            'Index': [1],
            'clinical_context': ['Patient presents with symptoms'],
            # Identical text = 0% change, should skip
            'gender_swap_text': ['Patient presents with symptoms'],
        })

        output_path = tmp_path / 'baselines.json'

        # Use word-level tokenizer mock that returns low change %
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        async def run_test():
            return await generate_all_baselines_async(
                df=df,
                tokenizer=MockTokenizer(),
                output_path=str(output_path),
                max_concurrent=10,
                _api_call_fn=mock_api_call
            )

        result = asyncio.run(run_test())

        # API should not have been called (perturbation too small)
        assert len(api_calls) == 0, "API should not be called for small perturbations"
        # Result should have skipped=True
        assert result['1']['gender_swap']['skipped'] == True

    def test_handles_api_errors_gracefully(self, tmp_path):
        """API errors should be caught and logged, not crash the entire run."""
        call_count = 0

        async def mock_api_call_with_errors(sample_id, ptype):
            nonlocal call_count
            call_count += 1
            if sample_id == '1':
                raise Exception("Simulated API error")
            return {
                'paraphrase': f'paraphrase for {sample_id}_{ptype}',
                'target_pct': 10.0,
                'actual_pct': 10.1,
                'deviation': 0.1,
                'retries_used': 0,
                'skipped': False
            }

        from generate_baselines import generate_all_baselines_async

        import pandas as pd
        df = pd.DataFrame({
            'Index': [1, 2, 3],
            'clinical_context': ['ctx A', 'ctx B', 'ctx C'],
            'gender_swap_text': ['pert A', 'pert B', 'pert C'],
        })

        output_path = tmp_path / 'baselines.json'

        # Mock tokenizer that returns different token counts to trigger >0.5% change
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                if 'pert' in text:
                    return ['tok'] * 12  # 12 tokens for perturbed
                return ['tok'] * 10  # 10 tokens for original (20% change)

        async def run_test():
            return await generate_all_baselines_async(
                df=df,
                tokenizer=MockTokenizer(),
                output_path=str(output_path),
                max_concurrent=10,
                _api_call_fn=mock_api_call_with_errors
            )

        # Should not raise, even though sample 1 errors
        result = asyncio.run(run_test())

        # Sample 1 should have error marker
        assert 'error' in result['1']['gender_swap']
        # Samples 2 and 3 should succeed
        assert result['2']['gender_swap']['paraphrase'] == 'paraphrase for 2_gender_swap'
        assert result['3']['gender_swap']['paraphrase'] == 'paraphrase for 3_gender_swap'
