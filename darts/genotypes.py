

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'Mc_sepConv_3x3',
    'Mc_sepConv_5x5',
    'Mc_sepConv_7x7',
]
HSI=Genotype(normal=[('Mc_sepConv_3x3', 1), ('Mc_sepConv_7x7', 0), ('Mc_sepConv_7x7', 0), ('Mc_sepConv_5x5', 2), ('Mc_sepConv_7x7', 1), ('Mc_sepConv_3x3', 3), ('Mc_sepConv_3x3', 2), ('Mc_sepConv_3x3', 3)], normal_concat=range(2, 6), reduce=[('Mc_sepConv_7x7', 1), ('Mc_sepConv_3x3', 0), ('Mc_sepConv_7x7', 1), ('Mc_sepConv_7x7', 2), ('Mc_sepConv_7x7', 3), ('skip_connect', 1), ('Mc_sepConv_3x3', 4), ('skip_connect', 1)], reduce_concat=range(2, 6))
 
