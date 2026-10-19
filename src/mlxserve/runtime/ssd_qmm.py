"""Decode affine Q4 directly from cached buffers, using MLX's native matvec.

The MIT-licensed helper is read from the installed MLX headers at runtime. No
Janas source is incorporated. Unsupported MLX layouts use native gather_qmm.
"""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def native_header():
    for root in Path(mx.__file__).resolve().parents:
        path = root / 'include/mlx/backend/metal/kernels/quantized.h'
        if path.is_file():
            source = path.read_text()
            boundary = 'template <typename T, int group_size, int bits>\nMETAL_FUNC void qmv_impl'
            if boundary in source and 'void qmv_fast_impl(' in source:
                return '#ifndef METAL_FUNC\n#define METAL_FUNC inline\n#endif\n' + source.split(boundary, 1)[0]
            break
    return None


@lru_cache(maxsize=128)
def _kernel(experts, input_dims, output_dims, group_size, scale_offset, bias_offset, input_stride):
    header = native_header()
    if header is None:
        return None
    header += f'\nconstant int ssd_in_dims = {input_dims};\nconstant int ssd_out_dims = {output_dims};\n'
    buffers = [f'p{i}' for i in range(experts)]
    choose = buffers[-1]
    for i in reversed(range(experts - 1)):
        choose = f'(e == {i} ? p{i} : {choose})'
    source = f'''
        uint t = threadgroup_position_in_grid.y;
        int e = indices[t];
        const device uint8_t* packed = {choose};
        const device uint* weights = reinterpret_cast<const device uint*>(packed);
        const device T* scales = reinterpret_cast<const device T*>(packed + {scale_offset});
        const device T* biases = reinterpret_cast<const device T*>(packed + {bias_offset});
        qmv_fast_impl<T, {group_size}, 4>(weights, scales, biases,
            x + t * {input_stride}, out + t * {output_dims}, ssd_in_dims, ssd_out_dims,
            uint3(0, threadgroup_position_in_grid.x, 0),
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    '''
    return mx.fast.metal_kernel(name='ssd_native_q4_matvec', input_names=['x', 'indices', *buffers],
                               output_names=['out'], source=source, header=header,
                               compile_options={'math_mode': 'fast'})


def packed_q4_matvec(x, indices, packed, *, input_dims, output_dims, group_size,
                      scale_type, scale_offset, bias_offset):
    kernel = _kernel(len(packed), input_dims, output_dims, group_size, scale_offset, bias_offset,
                     0 if x.shape[0] == 1 else input_dims)
    if kernel is None:
        return None
    return kernel(inputs=[x, indices, *packed], template=[('T', x.dtype)],
                  grid=(64 * ((output_dims + 7) // 8), indices.size, 1), threadgroup=(64, 1, 1),
                  output_shapes=[(indices.size, 1, output_dims)], output_dtypes=[x.dtype])[0]


@lru_cache(maxsize=128)
def _swiglu_kernel(experts, input_dims, output_dims, group_size, scale_offset, bias_offset, input_stride):
    header = native_header()
    if header is None or experts > 12 or output_dims % 8:
        return None
    start = header.find('template <typename T, int group_size, int bits, bool partial_rows = false>\nMETAL_FUNC void qmv_fast_impl(')
    if start < 0:
        return None
    helper = header[start:]
    required = ('    device T* y,', '  y += tid.x * out_vec_size + out_row;')
    if any(helper.count(part) != 1 for part in required):
        return None
    helper = helper.replace('qmv_fast_impl(', 'ssd_qmv_register(').replace(required[0], '    thread T* y,').replace(required[1], '')
    header += helper
    header += f'\nconstant int ssd_glu_in = {input_dims};\nconstant int ssd_glu_out = {output_dims};\n'
    def choose(prefix):
        result = f'{prefix}{experts - 1}'
        for i in reversed(range(experts - 1)):
            result = f'(e == {i} ? {prefix}{i} : {result})'
        return result
    source = f'''
        uint t = threadgroup_position_in_grid.y;
        int e = indices[t];
        const device uint8_t* up = {choose('u')};
        const device uint8_t* gate = {choose('g')};
        thread T uv[4];
        thread T gv[4];
        uint3 tile(0, threadgroup_position_in_grid.x, 0);
        ssd_qmv_register<T, {group_size}, 4>(reinterpret_cast<const device uint*>(up),
            reinterpret_cast<const device T*>(up + {scale_offset}),
            reinterpret_cast<const device T*>(up + {bias_offset}), x + t * {input_stride}, uv,
            ssd_glu_in, ssd_glu_out, tile, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        ssd_qmv_register<T, {group_size}, 4>(reinterpret_cast<const device uint*>(gate),
            reinterpret_cast<const device T*>(gate + {scale_offset}),
            reinterpret_cast<const device T*>(gate + {bias_offset}), x + t * {input_stride}, gv,
            ssd_glu_in, ssd_glu_out, tile, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        if (thread_index_in_simdgroup == 0) {{
            uint row = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup * 4;
            for (uint i = 0; i < 4; ++i) {{
                float g = float(gv[i]);
                out[t * {output_dims} + row + i] = T((g / (1.0f + exp(-g))) * float(uv[i]));
            }}
        }}
    '''
    return mx.fast.metal_kernel(name='ssd_native_q4_swiglu',
        input_names=['x', 'indices', *[f'u{i}' for i in range(experts)], *[f'g{i}' for i in range(experts)]],
        output_names=['out'], source=source, header=header, compile_options={'math_mode': 'fast'})


def packed_q4_swiglu(x, indices, up, gate, *, input_dims, output_dims, group_size, scale_offset, bias_offset):
    """Fuse two native Q4 matvec reductions and SwiGLU, retaining rounded T projections."""
    kernel = _swiglu_kernel(len(up), input_dims, output_dims, group_size, scale_offset, bias_offset,
                            0 if x.shape[0] == 1 else input_dims)
    if kernel is None:
        return None
    return kernel(inputs=[x, indices, *up, *gate], template=[('T', x.dtype)],
        grid=(64 * (output_dims // 8), indices.size, 1), threadgroup=(64, 1, 1),
        output_shapes=[(indices.size, 1, output_dims)], output_dtypes=[x.dtype])[0]
