import pyiron_workflow as pwf
from Helper_functions import _get_structure
from DataClassNode import DataInputs, WorkflowState
from typing import Union, Optional, Any

@pwf.as_function_node
def Struct_gen(I):
    """
        Generate a face-centered cubic (FCC) crystal structure with specified Miller indices.

        This function creates an FCC structure using ASE and converts it to pyiron format.
        The structure orientation is determined by the Miller indices, with support for
        common surface orientations (001, 110, 111).

        Parameters
        ----------
        I : dataclass
            Input dataclass containing:
            - Element : str
                Chemical symbol of the element (e.g., 'Ni', 'Al')
            - MillerIndices : tuple of int
                Miller indices for surface orientation (e.g., (0,0,1), (1,1,0), (1,1,1))
            - SuperCellDimensions : list of int
                Dimensions of the supercell in [x, y, z] directions
            - InputStructure : optional
                Placeholder for the generated structure (will be set by function)

        Returns
        -------
        I : dataclass
            Updated input dataclass with InputStructure field populated with the
            generated FCC crystal structure in pyiron format.

        Notes
        -----
        - Miller indices (0,0,1), (1,0,0), (0,1,0) are treated as equivalent '001' orientation
        - Miller indices (0,1,1), (1,1,0), (1,0,1) are treated as equivalent '110' orientation
        - Structures are created with periodic boundary conditions (pbc=(1,1,1))
    """

    from ase.lattice.cubic import FaceCenteredCubic
    from pyiron import ase_to_pyiron

    if I.MillerIndices in [(0,0,1), (1,0,0), (0,1,0)] or not I.MillerIndices:
        key = '001'
    elif I.MillerIndices in [(0,1,1), (1,1,0), (1,0,1)]:
        key = '110'
    else:
        key = ''.join(str(i) for i in I.MillerIndices)

    dir_dict = {'001':[[1,0,0],[0,1,0],[0,0,1]],
               '110':[[0,0,1],[-1,1,0],[1,1,0]],
               '111':[[1,-1,0],[1,1,-2],[1,1,1]]}
    
    Structure = ase_to_pyiron(FaceCenteredCubic(directions=dir_dict.get(key),
                                                                    size=tuple(I.SuperCellDimensions), 
                                                                    symbol=I.Element,
                                                                    latticeconstant=I.acell,
                                                                    pbc=(1,1,1)))
    I.InputStructure = Structure
    
    return I

@pwf.as_function_node
def add_vacuum(I:Union[DataInputs.dataclass,WorkflowState.dataclass], Structure_variable_name: str, override_vacuum:bool=False):
    import ase.build as abuild
    import numpy as np
    
    # Get the structure using the variable name string
    Structnew = getattr(I, Structure_variable_name).copy()
    
    if isinstance(I.acell_relaxed, (list, np.ndarray)):
        alat = I.acell_relaxed[2]
    elif isinstance(I.acell_relaxed, float):
        alat = I.acell_relaxed
        
    vacuum = I.vacuum*alat
    
    if I.DeformationMode.upper()=='ALIAS' or override_vacuum:
        print("Adding vacuum of ", vacuum, " Angstroms to the structure")
        abuild.add_vacuum(atoms=Structnew, vacuum=vacuum)
    else:
        print("Affine shear does not need vacuum")
    I.Structure_w_vacuum = Structnew
    return I

@pwf.as_function_node
def apply_strain(I: Union[DataInputs.dataclass,WorkflowState.dataclass],
                 strain: float = 0.0,
                 deformation_axis: int = 0,
                 fix_atoms: bool = False):
    import numpy as np
    
    # deformation_axis convention: 0->xx, 1->yy, 2->zz, 3->xy, 4->xz, 5->yz
    
    print(f"Straining given structure by {strain} along axis {deformation_axis}")
    
    Strainmatrix = np.eye(3)
    
    if deformation_axis < 3:
        # Normal strain
        Strainmatrix[deformation_axis, deformation_axis] += strain
    elif deformation_axis == 3:  # xy shear
        Strainmatrix[0, 1] += strain
    elif deformation_axis == 4:  # xz shear
        Strainmatrix[0, 2] += strain
    elif deformation_axis == 5:  # yz shear
        Strainmatrix[1, 2] += strain
    else:
        raise ValueError(f"Invalid deformation_axis: {deformation_axis}. Must be 0-5.")
    
    I_out = I.copy()
    Structure = I.RelaxedStructure
    newcell = Structure.copy()
    cellpar = newcell.cell
    newcellpar = np.matmul(Strainmatrix, cellpar)
    newcell.set_cell(newcellpar, scale_atoms=True)
    
    I_out.StrainedStructure = newcell
    
    return I_out

def apply_shear(Structure,
                strain: float = 0.0):
    """Apply affine shear deformation to structure."""
    import numpy as np
    
    print(f"Applying affine shear with strain {strain}")
    Strainmatrix = np.eye(3)
    Strainmatrix[2, 1] += strain

    newcell = Structure.copy()
    cellpar = newcell.cell
    newcellpar = np.matmul(Strainmatrix, cellpar)
    newcell.set_cell(newcellpar, scale_atoms=True)
    strained_cell = newcell.copy()      
        
    return strained_cell


# def apply_slip(Structure,
#                acell: float,
#                fraction: float = 1.0):
#     """Apply slip (atom shuffling) to structure."""
#     import numpy as np
    
#     print(f"Applying slip with fraction {fraction}")
#     Slipped_structure = Structure.copy()
#     pos = Structure.get_positions()
#     max_z = max(pos[:, 2])
    
#     slip = fraction * acell / np.sqrt(6)
    
#     coords = pos.copy()
    
#     for i, position in enumerate(pos):
#         if position[2] > 0.5 * max_z:
#             newpos = position + np.array([0, slip, 0])
#             coords[i] = newpos
    
#     Slipped_structure.set_positions(coords)
    
#     return Slipped_structure

@pwf.as_function_node
def set_selective_dynamics(I:Union[DataInputs.dataclass,WorkflowState.dataclass], 
                           Structure_variable_name: str):
                           #SelDyn:Union[list[bool],None]=None):
    import numpy as np
    I_out = I.copy()
    AlteredStructureName = Structure_variable_name
    
    if I.calctype.upper()=='RELAX' and I.DeformationMode.upper()=='ALIAS':
        Structure = getattr(I_out, Structure_variable_name).copy()
        print('Applying selective dynamics 2 ')
        #Structure.set_array("selective_dynamics",np.repeat([False,False,True],len(Structure),axis=0))
        Structure.set_array("selective_dynamics", np.tile([True, False, False], (len(Structure), 1)))
        #I.DeformedStructure_seldyn = Structure
        AlteredStructureName = Structure_variable_name+'_seldyn'
        setattr(I_out,AlteredStructureName,Structure)

    return I_out, AlteredStructureName

def apply_slip(Structure,
               acell: float,
               vacuum: float,
               fraction: float = 1.0,
              verbose:bool=False):
    """Apply slip (atom shuffling) to structure."""
    import numpy as np
    
    print(f"Applying slip with fraction {fraction}")
    Slipped_structure = Structure.copy()
    pos = Structure.get_positions()
    
    # Find the actual center of the slab based on atom positions
    min_z = np.min(pos[:, 2])
    max_z = np.max(pos[:, 2])
    zsplit = (min_z + max_z) / 2
    
    if verbose:
        print(f"Split plane at z = {zsplit:.3f}")
        print(f"Atoms below: {np.sum(pos[:, 2] <= zsplit)}, Atoms above: {np.sum(pos[:, 2] > zsplit)}")
    
    # Ensure acell is a scalar
    if isinstance(acell, (list, np.ndarray)):
        acell = float(acell[0])
    else:
        acell = float(acell)
    
    slip = fraction * acell / np.sqrt(6)
    slip = float(slip)
    
    if verbose:
        print(f"Applying slip of {slip:.4f} Å")
    
    coords = pos.copy()
    
    # Apply slip only to atoms in upper half
    mask = pos[:, 2] > zsplit
    coords[mask, 1] += slip  # Add slip in y-direction
    
    Slipped_structure.set_positions(coords)
    
    return Slipped_structure


# def apply_slip(Structure,
#                acell: float,
#                vacuum: float,
#                fraction: float = 1.0):
#     """Apply slip (atom shuffling) to structure."""
#     import numpy as np
    
#     print(f"Applying slip with fraction {fraction}")
#     Slipped_structure = Structure.copy()
#     pos = Structure.get_positions()

#     zsplit = (Structure.cell.cellpar()[2] - vacuum) / 2

    
#     #max_z = np.max(pos[:, 2])
    
#     # Ensure acell is a scalar
#     if isinstance(acell, (list, np.ndarray)):
#         acell = float(acell[0])  # Or whichever component you need
#     else:
#         acell = float(acell)
    
#     slip = fraction * acell / np.sqrt(6)
#     slip = float(slip)  # Ensure slip is a scalar
    
#     coords = pos.copy()
    
#     # Apply slip only to atoms in upper half
#     mask = pos[:, 2] > zsplit
#     coords[mask, 1] += slip  # Add slip in y-direction
    
#     Slipped_structure.set_positions(coords)
    
#     return Slipped_structure

@pwf.as_function_node
def DeformStructure(I: Union[DataInputs.dataclass, WorkflowState.dataclass],
                    Mode: str,
                    fraction: float = 1.0,
                    structure_attribute: str = 'Structure_Minimized'):
    """
    Apply deformation to structure from dataclass.
    
    Parameters
    ----------
    I : DataInputs or WorkflowState
        Input dataclass containing the structure to deform
    Mode : str
        Deformation mode: 'AFFINE' for shear strain, 'ALIAS' for slip
    fraction : float
        Deformation magnitude (strain for AFFINE, slip fraction for ALIAS)
    structure_attribute : str
        Name of structure attribute to deform (default: 'RelaxedStructure')
    
    Returns
    -------
    I_out : Same type as input
        Updated dataclass with DeformedStructure attribute added
    """
    import numpy as np
    
    assert Mode.upper() in ['AFFINE', 'ALIAS'], "!! Unknown Mode of deformation !!"
    
    # Get the structure to deform
    Structure = getattr(I, structure_attribute)
    
    if Mode.upper() == 'AFFINE':
        DeformedStructure = apply_shear(
            Structure=Structure,
            strain=fraction
        )
    
    elif Mode.upper() == 'ALIAS':
        # For ALIAS mode, we need acell_relaxed from the dataclass
        if hasattr(I, 'acell_relaxed') and I.acell_relaxed is not None:
            if isinstance(I.acell_relaxed, (list, np.ndarray)):
                acell = I.acell_relaxed[0]  # Or whichever component you need
            else:
                acell = I.acell_relaxed
        else:
            raise ValueError('!! Could not obtain acell_relaxed from dataclass for ALIAS mode !!')
        
        DeformedStructure = apply_slip(
            Structure=Structure,
            vacuum = acell * I.vacuum,
            fraction=fraction,
            acell=acell
        )
    
    # Copy dataclass and add deformed structure
    I_out = I.copy()
    I_out.DeformedStructure = DeformedStructure
    
    return I_out
