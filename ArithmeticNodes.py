import pyiron_workflow as pwf
from DataClassNode import DataInputs, WorkflowState
from typing import Union, Optional, Any

@pwf.as_function_node
def compute_gamma(I_Periodic:Union[DataInputs.dataclass,WorkflowState.dataclass],
                  I_Surfaces:Union[DataInputs.dataclass,WorkflowState.dataclass],
                  verbose=False):
    import numpy as np
    
    E0 = I_Periodic.Energy #_Minimization
    E1 = I_Surfaces.Energy #_Minimization
    a,b,c = I_Surfaces.Cell #_Minimization
    area_per_surface = np.linalg.norm(np.cross(a,b))
    area = 2*area_per_surface
    gamma = (E1 - E0)/(area)
    if verbose:
        print("###################################")
        print(f'gamma = {round(gamma,5)} eV/A^2')
        print("###################################")
    return gamma, area
