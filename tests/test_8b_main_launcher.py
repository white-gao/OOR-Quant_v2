"""Exercise launch commands and safe resume without loading weights or CUDA."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/fake_quant/run_8b_ad_product_w8a8_main_cuda67.sh'


@pytest.fixture
def launch_env(tmp_path):
    model = tmp_path / '8B'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({
        'num_hidden_layers': 36, 'hidden_size': 4096,
    }))
    data = tmp_path / 'benchmark_data'
    data.mkdir()
    return {
        **os.environ,
        'PYTHON_BIN': sys.executable,
        'MODEL_PATH': str(model),
        'DATA_DIR': str(data),
        'RUN_ROOT': str(tmp_path / 'results'),
        'W4A8_RUN_ROOT': str(tmp_path / 'results_w4a8'),
        'PRECISIONS': 'w8a8',
        'DRY_RUN': '1',
        'OVERWRITE': '0',
        'EVAL_OVERWRITE': '0',
        'CLEAN_EVAL_ARTIFACTS': '1',
        'GPUS': '6,7',
        'TASKS': 'ad product',
        'METHODS': 'bf16 rtn smoothquant omniquant flatquant recquant',
        'EVAL_SAMPLE_SIZE': 'full',
    }


def run_launcher(env, **overrides):
    return subprocess.run(
        ['bash', str(SCRIPT)], cwd=ROOT, env={**env, **overrides},
        capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize('gpu', ['6', '3'])
def test_single_gpu_product_training_wrapper(launch_env, gpu):
    script = ROOT / 'scripts/flat_quant/run_8b_product_w4a8_flatquant_recquant_train_cuda6.sh'
    result = subprocess.run(
        ['bash', str(script)], cwd=ROOT,
        env={**launch_env, 'GPU': gpu, 'RUN_EVAL': '1', 'OVERWRITE': '1'},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(launch_env['W4A8_RUN_ROOT']).exists()
    commands = runner_commands(result.stdout)
    assert len(commands) == 3  # Prefix, final MSE, final task alignment.
    for module, args in commands:
        assert module == 'flat_quant.run_m1_onerec_ad'
        assert '--calibration_only' in args
        assert '--eval_num_shards' not in args and '--overwrite' not in args
        assert value(args, '--task') == 'product'
        assert value(args, '--model_path') == launch_env['MODEL_PATH']
        assert value(args, '--weight_quant_format') == 'fp4_e2m1'
        assert value(args, '--activation_quant_format') == 'fp8_e4m3fn'
        assert value(args, '--flat_epochs') == '15'
    command_lines = [line for line in result.stdout.splitlines() if ' -m ' in line]
    assert all(f'CUDA_VISIBLE_DEVICES={gpu}' in line for line in command_lines)
    prefix, final, rec = [args for _, args in commands]
    assert value(prefix, '--layers') == '0-34'
    assert value(prefix, '--calib_sample_size') == '128'
    assert value(final, '--flat_train_sample_size') == '512'
    assert value(final, '--flat_validation_sample_size') == '512'
    assert value(rec, '--flat_train_sample_size') == '1024'
    assert value(rec, '--omni_lfq_boundary_loss_weight') == '0.3'
    assert '--flat_learn_transform' in rec and '--flat_learn_lac' in rec
    task_root = Path(launch_env['W4A8_RUN_ROOT']) / 'product'
    assert value(final, '--output_dir') == str(task_root / 'flatquant/final_train512_heldout512')
    assert value(rec, '--flat_finetune_checkpoint_dir') == str(
        task_root / 'flatquant/final_train512_heldout512/8B/product/flatquant_calibration')
    assert value(rec, '--output_dir') == str(task_root / 'recquant/abc_boundary0.3_joint')


def runner_commands(output):
    commands = []
    for line in output.splitlines():
        if ' -m ' not in line or '.run_m1_onerec_ad ' not in line:
            continue
        words = shlex.split(line)
        index = words.index('-m')
        commands.append((words[index + 1], words[index + 2:]))
    return commands


def value(args, name):
    return args[args.index(name) + 1]


def write_completed_eval(env, task, method, precision='w8a8'):
    root = Path(env['W4A8_RUN_ROOT'] if precision == 'w4a8' and method != 'bf16' else env['RUN_ROOT'])
    output = root / task / (f'flatquant/{task}_full_eval' if method == 'flatquant' else f'{method}_full_eval')
    output.mkdir(parents=True, exist_ok=True)
    metrics = {key: 0.1 for key in ('pass@16', 'pass@32', 'recall@16', 'recall@32')}
    metrics['total_samples'] = {'ad': 27677, 'product': 27910}[task]
    (output / 'eval_results.json').write_text(json.dumps({'8B': {task: {'test': metrics}}}))
    mode = {'bf16': 'full_precision', 'rtn': 'baseline_qdq', 'smoothquant': 'smoothquant_w8a8',
            'omniquant': 'omniquant', 'flatquant': 'flatquant_core', 'recquant': 'flatquant_core'}[method]
    config = {'method': mode, 'task': task, 'layers': list(range(36)), 'dtype': 'bfloat16',
              'eval_sample_size': 'full', 'eval_offset': 0, 'eval_merged': True,
              'split': 'test', 'num_beams': 32, 'num_return_sequences': 32,
              'max_new_tokens': 3, 'seed': 42, 'calibration_only': False,
              'eval_data_dir': env['DATA_DIR']}
    if method != 'bf16':
        config.update(weight_quant_format='fp4_e2m1' if precision == 'w4a8' else 'fp8_e4m3fn',
                      activation_quant_format='fp8_e4m3fn', weight_group_size=0,
                      weight_quant_scheme='symmetric', act_quant='per_token', act_quant_mode='shared_input')
    config_path = output / '8B' / task / f'{mode}_config.json'
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    return output, config_path


def test_twelve_experiments_match_8b_protocol_and_cli(launch_env, monkeypatch):
    result = run_launcher(launch_env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('[experiment ') == 12
    assert not Path(launch_env['RUN_ROOT']).exists()  # Dry runs do not create outputs.
    commands = runner_commands(result.stdout)
    assert len(commands) == 34  # 10 calibration processes + 12 x 2 evaluation shards.
    train = [args for _, args in commands if '--calibration_only' in args]
    evaluate = [args for _, args in commands if '--eval_num_shards' in args]
    assert len(train) == 10
    assert len(evaluate) == 24
    assert sum(value(args, '--layers') == '0-34' for args in train) == 4

    for module, args in commands:
        assert value(args, '--model_path') == launch_env['MODEL_PATH']
        assert value(args, '--weight_quant_format') == 'fp8_e4m3fn'
        assert value(args, '--activation_quant_format') == 'fp8_e4m3fn'
        assert value(args, '--weight_group_size') == '0'
        assert '--act_quant_mode' not in args
        assert '/1.7B/' not in ' '.join(args)
        # Actual argparse entrypoints, not a copied list of accepted flags.
        monkeypatch.setattr(sys, 'argv', [module, *args])
        parsed = importlib.import_module(module).parse_args()
        assert parsed.seed == 42
        assert parsed.num_beams == parsed.num_return_sequences == 32
        assert parsed.max_new_tokens == 3

    for task in ('ad', 'product'):
        scoped = [args for args in train if value(args, '--task') == task]
        omni = [args for args in scoped if value(args, '--mode') == 'omniquant']
        assert [value(args, '--calib_sample_size') for args in omni] == ['128', '1024']
        assert all(value(args, '--omni_epochs') == '20' for args in omni)
        assert all(value(args, '--omni_let_mode') == 'none' for args in omni)
        flat = [args for args in scoped if value(args, '--mode') == 'flatquant_core']
        assert len(flat) == 3
        baseline_final = next(args for args in flat if '--flat_prefix_checkpoint_dir' in args)
        assert value(baseline_final, '--flat_train_sample_size') == '512'
        assert value(baseline_final, '--flat_validation_sample_size') == '512'
        rec = next(args for args in flat if '--flat_finetune_checkpoint_dir' in args)
        assert value(rec, '--flat_train_sample_size') == '1024'
        assert value(rec, '--flat_validation_sample_size') == '0'
        assert value(rec, '--flat_epochs') == '15'
        assert value(rec, '--omni_lfq_boundary_loss_weight') == '0.3'
        assert '--flat_learn_transform' in rec
        assert '--flat_learn_lac' in rec
        expected_source = Path(value(baseline_final, '--output_dir')) / '8B' / task / 'flatquant_calibration'
        assert value(rec, '--flat_finetune_checkpoint_dir') == str(expected_source)

    for args in evaluate:
        assert value(args, '--eval_sample_size') == 'full'
        assert '--omni_final_objective' not in args or value(args, '--omni_final_objective') == 'mse'


def test_eval_overwrite_does_not_retrain_or_change_gpu_allocation(launch_env):
    result = run_launcher(launch_env, EVAL_OVERWRITE='1', GPUS='2,3')
    assert result.returncode == 0, result.stderr
    assert 'physical_gpu=6' not in result.stdout
    assert 'physical_gpu=2' in result.stdout
    assert 'physical_gpu=3' in result.stdout
    for _, args in runner_commands(result.stdout):
        assert ('--overwrite' in args) == ('--eval_num_shards' in args)


def test_completed_36_layer_checkpoints_are_reused(launch_env):
    task_root = Path(launch_env['RUN_ROOT']) / 'ad'
    stages = [
        ('omniquant/prefix_calib128/8B/ad/omniquant_calibration', 35),
        ('omniquant/mse_lwc_final1024_calibration/8B/ad/omniquant_calibration', 36),
        ('flatquant/prefix_calib128/8B/ad/flatquant_calibration', 35),
        ('flatquant/final_train512_heldout512/8B/ad/flatquant_calibration', 36),
        ('recquant/abc_boundary0.3_joint/8B/ad/flatquant_calibration', 36),
    ]
    for relative, count in stages:
        directory = task_root / relative
        directory.mkdir(parents=True)
        for layer in range(count):
            (directory / f'layer_{layer:02d}.pt').write_bytes(b'placeholder')
    result = run_launcher(
        launch_env, TASKS='ad', METHODS='omniquant flatquant recquant',
        EVAL_OVERWRITE='1',
    )
    assert result.returncode == 0, result.stderr
    commands = runner_commands(result.stdout)
    assert len(commands) == 6
    assert all('--calibration_only' not in args for _, args in commands)
    assert all('--overwrite' in args for _, args in commands)


@pytest.mark.parametrize('overrides', [
    {'GPUS': '6,6'}, {'GPUS': '6'}, {'GPUS': '6,7,'},
    {'TASKS': 'video'}, {'METHODS': 'rtn rtn'},
    {'OVERWRITE': '1', 'EVAL_OVERWRITE': '0'},
    {'PRECISIONS': 'w4a8 w4a8'}, {'PRECISIONS': 'w4a4'}, {'PRECISIONS': ' '},
])
def test_invalid_settings_fail_before_work(launch_env, overrides):
    result = run_launcher(launch_env, **overrides)
    assert result.returncode == 2
    assert not Path(launch_env['RUN_ROOT']).exists()


def test_wrong_model_rejected(launch_env):
    config = Path(launch_env['MODEL_PATH']) / 'config.json'
    config.write_text(json.dumps({'num_hidden_layers': 28, 'hidden_size': 2048}))
    result = run_launcher(launch_env)
    assert result.returncode != 0
    assert 'Expected OneRec-8B' in result.stderr


def test_complete_results_skip_without_modifying_existing_files(launch_env):
    output, _ = write_completed_eval(launch_env, 'ad', 'bf16')
    metrics = output / 'eval_results.json'
    shard = output.with_name(output.name + '.shards') / 'shard_000_of_002'
    shard.mkdir(parents=True)
    for directory in (output, shard):
        (directory / 'test_generated.json').write_text('large generated data')
        (directory / 'run.log').write_text('retain log')
        (directory / 'config.json').write_text('{}')
        (directory / 'layer_35.pt').write_bytes(b'retain checkpoint')
    result = run_launcher(launch_env, DRY_RUN='0', TASKS='ad', METHODS='bf16')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '[skip]' in result.stdout
    assert metrics.exists()
    for directory in (output, shard):
        assert (directory / 'test_generated.json').exists()
        for name in ('run.log', 'config.json', 'layer_35.pt'):
            assert (directory / name).exists()


def test_partial_eval_stops_without_starting_generation(launch_env):
    output = Path(launch_env['RUN_ROOT']) / 'ad/bf16_full_eval'
    output.mkdir(parents=True)
    result = run_launcher(launch_env, DRY_RUN='0', TASKS='ad', METHODS='bf16')
    assert result.returncode == 3
    assert 'EVAL_OVERWRITE=1' in result.stdout
    assert not runner_commands(result.stdout)


def test_recquant_without_baseline_fails_before_training(launch_env):
    result = run_launcher(launch_env, DRY_RUN='0', TASKS='ad', METHODS='recquant')
    assert result.returncode == 3
    assert 'Incomplete source MSE checkpoint' in result.stdout
    assert not runner_commands(result.stdout)


def test_default_both_precisions_share_bf16_and_separate_checkpoints(launch_env, monkeypatch):
    env = dict(launch_env)
    env.pop('PRECISIONS')
    result = run_launcher(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('[experiment ') == 22
    commands = runner_commands(result.stdout)
    assert len(commands) == 64  # 20 calibration processes + 22 x 2 eval shards.
    bf16 = [args for _, args in commands if value(args, '--mode') == 'full_precision']
    assert len(bf16) == 4
    w4 = [(module, args) for module, args in commands if value(args, '--weight_quant_format') == 'fp4_e2m1']
    assert len(w4) == 30  # Ten training processes and twenty evaluation shards.
    assert not Path(env['RUN_ROOT']).exists()
    assert not Path(env['W4A8_RUN_ROOT']).exists()
    for module, args in w4:
        assert value(args, '--weight_group_size') == '0'
        assert value(args, '--activation_quant_format') == 'fp8_e4m3fn'
        assert Path(value(args, '--output_dir')).is_relative_to(env['W4A8_RUN_ROOT'])
        for flag in ('--omni_prefix_checkpoint_dir', '--omni_load_checkpoint_dir',
                     '--flat_prefix_checkpoint_dir', '--flat_finetune_checkpoint_dir', '--flat_load_checkpoint_dir'):
            if flag in args:
                assert Path(value(args, flag)).is_relative_to(env['W4A8_RUN_ROOT'])
        monkeypatch.setattr(sys, 'argv', [module, *args])
        parsed = importlib.import_module(module).parse_args()
        assert parsed.seed == 42
        assert parsed.num_beams == parsed.num_return_sequences == 32
    for task in ('ad', 'product'):
        train = [args for _, args in w4 if '--calibration_only' in args and value(args, '--task') == task]
        assert len(train) == 5
        assert sum(value(args, '--layers') == '0-34' for args in train) == 2
        rec = next(args for args in train if '--flat_finetune_checkpoint_dir' in args)
        baseline = next(args for args in train if '--flat_prefix_checkpoint_dir' in args)
        assert value(rec, '--flat_train_sample_size') == '1024'
        assert value(rec, '--flat_epochs') == '15'
        assert value(rec, '--omni_lfq_boundary_loss_weight') == '0.3'
        assert '--flat_learn_transform' in rec and '--flat_learn_lac' in rec
        assert value(rec, '--flat_finetune_checkpoint_dir') == str(Path(value(baseline, '--output_dir')) / '8B' / task / 'flatquant_calibration')


def test_completed_w8a8_default_run_only_launches_ten_w4a8_experiments(launch_env):
    for task in ('ad', 'product'):
        for method in launch_env['METHODS'].split():
            write_completed_eval(launch_env, task, method)
    original = {p: p.read_bytes() for p in Path(launch_env['RUN_ROOT']).rglob('*') if p.is_file()}
    env = dict(launch_env)
    env.pop('PRECISIONS')
    result = run_launcher(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('[skip]') == 12
    commands = runner_commands(result.stdout)
    assert len(commands) == 30
    assert all(value(args, '--weight_quant_format') == 'fp4_e2m1' for _, args in commands)
    assert all(value(args, '--mode') != 'full_precision' for _, args in commands)
    assert {p: p.read_bytes() for p in Path(launch_env['RUN_ROOT']).rglob('*') if p.is_file()} == original
    assert not Path(launch_env['W4A8_RUN_ROOT']).exists()


def test_w4a8_only_reuses_shared_bf16(launch_env):
    for task in ('ad', 'product'):
        write_completed_eval(launch_env, task, 'bf16')
    result = run_launcher(launch_env, PRECISIONS='w4a8')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('[skip]') == 2
    assert len(runner_commands(result.stdout)) == 30
    assert all(value(args, '--weight_quant_format') == 'fp4_e2m1' for _, args in runner_commands(result.stdout))


@pytest.mark.parametrize('wrong', ['fp8_e4m3fn', 'int4'])
def test_w4a8_rejects_cached_result_with_wrong_precision(launch_env, wrong):
    _, config_path = write_completed_eval(launch_env, 'ad', 'rtn', 'w4a8')
    cfg = json.loads(config_path.read_text())
    cfg['weight_quant_format'] = wrong
    config_path.write_text(json.dumps(cfg))
    result = run_launcher(launch_env, PRECISIONS='w4a8', TASKS='ad', METHODS='rtn')
    assert result.returncode != 0
    assert 'Config mismatch' in result.stderr
    assert not runner_commands(result.stdout)


@pytest.mark.parametrize('suffix', ['', '/nested'])
def test_precision_output_roots_cannot_overlap(launch_env, suffix):
    result = run_launcher(launch_env, W4A8_RUN_ROOT=launch_env['RUN_ROOT']+suffix)
    assert result.returncode == 2
    assert 'non-nested' in result.stderr


def test_w4a8_does_not_use_existing_w8a8_flatquant_checkpoint(launch_env):
    checkpoint = Path(launch_env['RUN_ROOT']) / 'ad/flatquant/final_train512_heldout512/8B/ad/flatquant_calibration'
    checkpoint.mkdir(parents=True)
    for layer in range(36):
        (checkpoint / f'layer_{layer:02d}.pt').write_bytes(b'W8A8 fixture')
    result = run_launcher(launch_env, DRY_RUN='0', PRECISIONS='w4a8', TASKS='ad', METHODS='recquant')
    assert result.returncode == 3
    assert 'Incomplete source MSE checkpoint' in result.stdout
    assert launch_env['W4A8_RUN_ROOT'] in result.stdout
    assert not runner_commands(result.stdout)
