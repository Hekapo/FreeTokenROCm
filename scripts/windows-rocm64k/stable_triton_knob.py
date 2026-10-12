"""Pin the effective AMD buffer-ops flag once in each Triton 3.8 process."""

def pin_stable_amd_knob(triton):
    if triton.__version__ != '3.8.0':
        raise RuntimeError('Stable AMD knob requires the verified Triton 3.8.0 runtime')
    value = triton.knobs.amd.use_buffer_ops
    if type(value) is not bool:
        raise RuntimeError('Unexpected AMD buffer ops setting')
    triton.knobs.amd.use_buffer_ops = value
    if triton.knobs.amd.use_buffer_ops is not value:
        raise RuntimeError('AMD buffer ops setting changed during initialization')
    return {'triton_version':triton.__version__, 'use_buffer_ops':value, 'cached_at_process_start':True}
