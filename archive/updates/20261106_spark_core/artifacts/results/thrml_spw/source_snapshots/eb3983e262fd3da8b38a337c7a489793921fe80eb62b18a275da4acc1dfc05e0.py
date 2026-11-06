"""Pure NumPy fixed-range learned-command adapter, without Torch dependencies."""
import numpy as np


def encoding(exported):
    result = []
    for prefix in ['conv1', 'conv2', 'head']:
        codes = exported[prefix+'_commands']
        assert codes.dtype == np.uint8
        fields = (-4+codes.astype(float)*8/255).astype(np.float32)
        np.testing.assert_array_equal(fields, exported[prefix+'_fields'])
        result.append({'center': exported[prefix+'_center'], 'radius': exported[prefix+'_radius'],
                       'fields': fields.T.copy(), 'commands': codes.copy(),
                       'bias': exported[prefix+'_bias'].astype(float)})
    return result
