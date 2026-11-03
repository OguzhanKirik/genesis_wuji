"""The actor: adam-rl's exported ONNX MLP, run on the CPU with onnxruntime.

Input `obs` float32 [1, N], output `action` float32 [1, 8] in [-1, 1], the
observation normaliser folded in by the export. Batch size 1 at 30 Hz is
~100k multiply-adds, where a GPU would only add launch latency and CUDA to
the deploy image (see README).
"""
import numpy as np

from .level import N_ACTION


class ActorError(RuntimeError):
    pass


class OnnxActor:
    def __init__(self, path, obs_dim=None, input_name='obs', output_name='action'):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - the image installs it
            raise ActorError('onnxruntime is not installed (pip install onnxruntime)') from e
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        inputs = {i.name: i for i in self.session.get_inputs()}
        outputs = {o.name: o for o in self.session.get_outputs()}
        if input_name not in inputs:
            raise ActorError(f'{path}: no input {input_name!r}; inputs are {list(inputs)}')
        if output_name not in outputs:
            raise ActorError(f'{path}: no output {output_name!r}; outputs are {list(outputs)}')
        self.input_name, self.output_name = input_name, output_name
        shape = list(inputs[input_name].shape)
        if len(shape) != 2 or not isinstance(shape[1], int):
            raise ActorError(f'{path}: input {input_name!r} must be [batch, N], is {shape}')
        self.obs_dim = int(shape[1])
        if obs_dim is not None and self.obs_dim != obs_dim:
            raise ActorError(f'{path}: actor takes {self.obs_dim} observations, level.yaml lays out {obs_dim}')
        out_shape = list(outputs[output_name].shape)
        if len(out_shape) != 2 or out_shape[1] != N_ACTION:
            raise ActorError(f'{path}: output {output_name!r} must be [batch, {N_ACTION}], is {out_shape}')
        self.path = str(path)

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.obs_dim:
            raise ActorError(f'observation has {x.shape[1]} values, the actor takes {self.obs_dim}')
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        return np.asarray(y, dtype=float).reshape(N_ACTION)


class ZeroActor:
    """Every action is 0: no wrist step, and a closure command of 0.5, which
    the runtime's smoothing drives both synergies to half way between open
    and closed. The stub policy; it exercises the whole path and never grasps."""

    def __init__(self, obs_dim):
        self.obs_dim = int(obs_dim)

    def __call__(self, obs):
        x = np.asarray(obs).reshape(-1)
        if x.shape[0] != self.obs_dim:
            raise ActorError(f'observation has {x.shape[0]} values, the actor takes {self.obs_dim}')
        return np.zeros(N_ACTION)


class ScriptedActor:
    """A callable per step index, for tests: fn(step, obs) -> action."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, obs):
        a = np.asarray(self.fn(self.calls, obs), dtype=float).reshape(N_ACTION)
        self.calls += 1
        return a
