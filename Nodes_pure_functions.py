import pyiron_workflow as pwf
from Helper_functions import _get_structure

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
               '110':[[1,1,0],[-1,1,0],[0,0,1]],
               '111':[[1,1,1],[1,-1,0],[1,1,-2]]}

    Structure = ase_to_pyiron(FaceCenteredCubic(directions=dir_dict.get(key),
                                                                    size=tuple(I.SuperCellDimensions), 
                                                                    symbol=I.Element, 
                                                                    pbc=(1,1,1)))
    I.InputStructure = Structure
    
    return I


@pwf.as_function_node
def GeometryOptimization(I,
                        return_as_dataclass:bool = True):
    """
        Perform complete geometry optimization (cell and atomic positions) using LAMMPS or VASP.

        This function relaxes both the atomic positions and the simulation cell to minimize
        the total energy and achieve target pressure/force convergence criteria.

        Parameters
        ----------
        I : dataclass
            Input dataclass containing:
            - calcengine : str
                Calculation engine, either 'LAMMPS' or 'VASP'
            - InputStructure : Structure
                Initial atomic structure to optimize
            - InteratomicPotential : str or DataFrame
                Path to potential file (LAMMPS) or potential DataFrame
            - ProjectName : str
                Name of the project directory for output files
            - MinimizerForCompleteRelaxation : str
                Minimization algorithm for LAMMPS (e.g., 'cg', 'sd')
            - etol : float
                Energy tolerance for convergence
            - ftol : float
                Force tolerance for convergence
            - Encut : float
                Plane wave energy cutoff for VASP (eV)
            - Kmesh : list of int
                K-point mesh for VASP [kx, ky, kz]
            - MillerIndices : tuple of int
                Miller indices for lattice constant calculation
            - SuperCellDimensions : list of int
                Supercell dimensions [nx, ny, nz]
        return_as_dataclass : bool, optional
            If True, returns updated dataclass; if False, returns dictionary (default: True)

        Returns
        -------
        Output : dataclass or dict
            If return_as_dataclass=True:
                Updated input dataclass with additional fields:
                - Energy_RelaxedStructure : float
                    Total energy of relaxed structure (eV)
                - Pressures_RelaxedStructure : array
                    Pressure tensor components (GPa)
                - ForceMax_RelaxedStructure : float
                    Maximum force magnitude on any atom (eV/Å)
                - Cell_RelaxedStructure : ndarray
                    Relaxed simulation cell vectors (Å)
                - RelaxedStructure : Structure
                    Relaxed atomic structure
                - acell_relaxed : ndarray
                    Relaxed lattice constants (Å)
            If return_as_dataclass=False:
                Dictionary with keys: 'FinalEnergy', 'FinalPressures', 'ForceMax',
                'FinalStructure', 'LatticeConstants'

        Raises
        ------
        AssertionError
            If LAMMPS minimizer is incompatible with box relaxation
        ValueError
            If calcengine is not 'LAMMPS' or 'VASP'

        Notes
        -----
        - For LAMMPS: Uses fix box/relax for cell optimization
        - For VASP: Uses ISIF=3 type relaxation (implicit)
        - Pressure target is set to 0.0 GPa (isotropic)
        - Output files are written to: ProjectName/CompleteGeometryOptimization/
    """
    import numpy as np
    import pandas as pd
    import os
    
    name = 'CompleteGeometryOptimization'

    if I.calcengine.upper()=='LAMMPS':
        assert I.MinimizerForCompleteRelaxation not in ['quickmin', 'fire', 'hftn','cg/kk'], "The quickmin, fire, hftn, and cg/kk styles do not yet support the use of the fix box/relax command or minimizations involving the electron radius in eFF models."

        
        from pyiron_atomistics.lammps.lammps import lammps_function
        
        if isinstance(I.InteratomicPotential, pd.DataFrame): 
            executable_version = 'conda'
            PotentialName = I.InteratomicPotential['Name']
        else:
            executable_version = None
            
        print("###############################################################################")
        print(f"Using the Interatomic potential defined in: {I.InteratomicPotential}")
        print("###############################################################################")
        _, parsed_output, job_crashed = lammps_function(working_directory=os.path.abspath(os.path.join(I.ProjectName, name)), 
                                                        structure=I.InputStructure,
                                                        potential=I.InteratomicPotential, 
                                                        calc_mode="minimize", 
                                                        calc_kwargs={"style":I.MinimizerForCompleteRelaxation,
                                                                     "pressure": [0.0,0.0,0.0,0.0,0.0,0.0],
                                                                     "ionic_energy_tolerance":I.etol,
                                                                     "ionic_force_tolerance":I.ftol,
                                                                     "n_print": 100000}, 
                                                        cutoff_radius=None, 
                                                        units="metal", 
                                                        bonds_kwargs={}, 
                                                        enable_h5md=False,
                                                        executable_version=executable_version
                                                    )
        parsed_output_red = parsed_output.get('generic')
        RelaxedStructure = _get_structure(structure=I.InputStructure, 
                                            cell = parsed_output_red.get('cells')[-1], 
                                            indices = parsed_output_red.get('indices')[-1],
                                            positions = parsed_output_red.get('positions')[-1],
                                            unwrapped_positions= parsed_output_red.get('unwrapped_positions')[-1]
                                           )
        
    elif I.calcengine.upper()=='VASP':
     
        from pyiron_atomistics.vasp.vasp import vasp_function
        print("###############################################################################")
        print(f"Setting Encut to {I.Encut} and using a Kmesh of {I.Kmesh}")
        print("###############################################################################")

        _, parsed_output, job_crashed = vasp_function(working_directory=os.path.abspath(os.path.join(I.ProjectName, name)),
                                                         structure=I.InputStructure,
                                                         plane_wave_cutoff=I.Encut,
                                                         kpoints_kwargs={"mesh":I.Kmesh},
                                                         calc_mode="minimize",
                                                         calc_kwargs={"pressure":0.0,
                                                                     "ionic_steps":100,
                                                                     "electronic_steps":60,
                                                                     "ionic_energy_tolerance":I.etol,
                                                                     "ionic_force_tolerance":I.ftol
                                                                     },
                                                         occupancy_smearing_kwargs={"smearing":'MP',
                                                                                    "width":0.6},
                                                         convergence_precision_kwargs={"electronic_energy":I.etol},
                                                         #algorithm_kwargs={"algorithm":"Normal"}
                                                        )
        parsed_output_red = parsed_output.get('generic')
        RelaxedStructure = _get_structure(structure=I.InputStructure, 
                                                cell = parsed_output.get('generic').get('cells')[-1], 
                                                indices = parsed_output.get('structure').get('indices'),
                                                positions = parsed_output.get('generic').get('positions')[-1],
                                                unwrapped_positions = parsed_output.get('generic').get('positions')[-1]
                                                )
    else:
        raise ValueError(f"Unknown calcengine: {I.calcengine}. Must be 'LAMMPS' or 'VASP'")
        
        
    
    Cell_RelaxedStructure = parsed_output_red.get('cells')[-1]
    

    if I.MillerIndices in [(0,0,1), (1,0,0), (0,1,0)] or not I.MillerIndices:
        key = '001'
    elif I.MillerIndices in [(0,1,1), (1,1,0), (1,0,1)]:
        key = '110'
    else:
        key = ''.join(str(i) for i in I.MillerIndices)

    dir_dict = {'001':[[1,0,0],[0,1,0],[0,0,1]],
               '110':[[1,1,0],[-1,1,0],[0,0,1]],
               '111':[[1,1,1],[1,-1,0],[1,1,-2]]}

    
    millermod = [np.linalg.norm(ele) for ele in dir_dict.get(key)]
    
    div_matrix = np.eye(3) * I.SuperCellDimensions * millermod

    
    Energy_RelaxedStructure = parsed_output_red.get('energy_tot')[-1]
    Pressures_RelaxedStructure = parsed_output_red.get('pressures')[-1]
    ForceMax_RelaxedStructure = max([np.linalg.norm(ele) for ele in parsed_output_red.get('forces')[-1]])
    acell_relaxed = Cell_RelaxedStructure @ np.linalg.inv(div_matrix)

    if return_as_dataclass:
        Output = I.copy()
        Output.Energy_RelaxedStructure = Energy_RelaxedStructure
        Output.Pressures_RelaxedStructure = Pressures_RelaxedStructure 
        Output.ForceMax_RelaxedStructure = ForceMax_RelaxedStructure 
        
        Output.Cell_RelaxedStructure = Cell_RelaxedStructure
        Output.RelaxedStructure = RelaxedStructure
        Output.acell_relaxed = acell_relaxed
        
    else:
        Output = {
                    'FinalEnergy':Energy_RelaxedStructure,
                    'FinalPressures':Pressures_RelaxedStructure,
                    'ForceMax':ForceMax_RelaxedStructure,
                    'FinalStructure':RelaxedStructure,
                    'LatticeConstants':acell_relaxed
                 }

    return Output
    # parsed_output, RelaxedStructure, Energy_RelaxedStructure, Pressures_RelaxedStructure, ForceMax_RelaxedStructure, acell

import pyiron_workflow as pwf
from Helper_functions import _get_structure

@pwf.as_function_node
def ForceMinimization(I, 
                      return_as_dataclass:bool=True):
    """
        Perform force minimization or static calculation on a structure.

        This function either minimizes forces while keeping the cell fixed, or performs
        a static single-point energy calculation, depending on the calctype parameter.

        Parameters
        ----------
        I : dataclass
            Input dataclass containing:
            - calcengine : str
                Calculation engine, either 'LAMMPS' or 'VASP'
            - calctype : str
                Type of calculation: 'RELAX'/'RELAXED' for minimization,
                'STATIC'/'UNRELAXED' for single-point energy
            - RelaxedStructure : Structure
                Input atomic structure (typically from GeometryOptimization)
            - InteratomicPotential : str or DataFrame
                Path to potential file (LAMMPS) or potential DataFrame
            - ProjectName : str
                Name of the project directory for output files
            - MinimizerForSurfaces : str
                Minimization algorithm for LAMMPS (used if calctype='RELAX')
            - etol : float
                Energy tolerance for convergence
            - ftol : float
                Force tolerance for convergence
            - Encut : float
                Plane wave energy cutoff for VASP (eV)
            - Kmesh : list of int
                K-point mesh for VASP [kx, ky, kz]
        return_as_dataclass : bool, optional
            If True, returns updated dataclass; if False, returns dictionary (default: True)

        Returns
        -------
        Output : dataclass or dict
            If return_as_dataclass=True:
                Updated input dataclass with additional fields:
                - Energy_Minimization : float
                    Total energy after minimization/static calc (eV)
                - Pressures_Minimization : array
                    Pressure tensor components (GPa)
                - ForceMax_Minimization : float
                    Maximum force magnitude on any atom (eV/Å)
                - Cell_Minimization : ndarray
                    Final simulation cell vectors (Å)
                - Structure_Minimized : Structure
                    Final atomic structure
            If return_as_dataclass=False:
                Dictionary with keys: 'FinalEnergy', 'FinalPressures', 'ForceMax',
                'FinalStructure'

        Raises
        ------
        ValueError
            If calcengine is not 'LAMMPS' or 'VASP'

        Notes
        -----
        - For RELAX calculations: Only atomic positions are optimized, cell is fixed
        - For STATIC calculations: No optimization, just energy/force evaluation
        - Output files are written to: ProjectName/Minimization/
        - Commonly used for surface calculations where cell should remain fixed
    """
    import pandas as pd
    import numpy as np
    import os
    
    name = 'Minimization'

    if I.calcengine.upper()=='LAMMPS':

        ###############################################################################
                                    # LAMMPS Stuff #
        ###############################################################################

        from pyiron_atomistics.lammps.lammps import lammps_function
        
        if isinstance(I.InteratomicPotential, pd.DataFrame): 
            executable_version = 'conda'
            PotentialName = I.InteratomicPotential['Name']
        else:
            executable_version = None
        
        if I.calctype.upper() in ['RELAX','RELAXED']:
            _, parsed_output, job_crashed = lammps_function(working_directory=os.path.abspath(os.path.join(I.ProjectName, name)), 
                                                            structure=I.RelaxedStructure,
                                                            potential=I.InteratomicPotential, 
                                                            calc_mode="minimize", 
                                                            calc_kwargs={"style":I.MinimizerForSurfaces,
                                                                         "ionic_energy_tolerance":I.etol,
                                                                         "ionic_force_tolerance":I.ftol,
                                                                         "n_print": 100000}, 
                                                            cutoff_radius=None, 
                                                            units="metal", 
                                                            bonds_kwargs={}, 
                                                            enable_h5md=False,
                                                            executable_version=executable_version
                                                        )
        elif I.calctype.upper() in ['STATIC','UNRELAXED']:
            _, parsed_output, job_crashed = lammps_function(working_directory=os.path.abspath(os.path.join(I.ProjectName, name)), 
                                                            structure=I.RelaxedStructure,
                                                            potential=I.InteratomicPotential,
                                                            calc_mode="static",
                                                            calc_kwargs = {},
                                                            cutoff_radius=None, 
                                                            units="metal", 
                                                            bonds_kwargs={}, 
                                                            enable_h5md=False,
                                                            executable_version=executable_version
                                                        )
            
            
        parsed_output_red = parsed_output.get('generic')
        Structure_Minimized = _get_structure(structure=I.RelaxedStructure, 
                                            cell = parsed_output_red.get('cells')[-1], 
                                            indices = parsed_output_red.get('indices')[-1],
                                            positions = parsed_output_red.get('positions')[-1],
                                            unwrapped_positions= parsed_output_red.get('unwrapped_positions')[-1]
                                           )
        
    elif I.calcengine.upper()=='VASP':

        from pyiron_atomistics.vasp.vasp import vasp_function

        ###############################################################################
                                        # VASP Stuff #
        ###############################################################################

        if I.calctype.upper() in ['RELAX','RELAXED']:
            _, parsed_output, job_crashed = vasp_function(working_directory=os.path.abspath(os.path.join(I.ProjectName, name)),
                                                          structure=I.RelaxedStructure,
                                                          plane_wave_cutoff=I.Encut,
                                                          kpoints_kwargs={"mesh":I.Kmesh},
                                                          calc_mode="minimize",
                                                          calc_kwargs={"ionic_steps":100,
                                                                       "electronic_steps":60,
                                                                       "ionic_energy_tolerance":I.etol,
                                                                       "ionic_force_tolerance":I.ftol
                                                                       },
                                                          occupancy_smearing_kwargs={"smearing":'MP',
                                                                                     "width":0.6},
                                                          convergence_precision_kwargs={"electronic_energy":I.etol},
                                                          #algorithm_kwargs={"algorithm":"Normal"}
                                                          )
        elif I.calctype.upper() in ['STATIC','UNRELAXED']:
            _, parsed_output, job_crashed = vasp_function(working_directory=os.path.abspath(os.path.join(I.ProjectName, name)),
                                                          structure=I.RelaxedStructure,
                                                          plane_wave_cutoff=I.Encut,
                                                          kpoints_kwargs={"mesh":I.Kmesh},
                                                          calc_mode="static",
                                                          calc_kwargs={},
                                                          occupancy_smearing_kwargs={"smearing":'MP',
                                                                                     "width":0.6},
                                                          convergence_precision_kwargs={"electronic_energy":I.etol},
                                                          #algorithm_kwargs={"algorithm":"Normal"}
                                                          )

        parsed_output_red = parsed_output.get('generic')
        Structure_Minimized = _get_structure(structure=I.RelaxedStructure,
                                          cell = parsed_output_red.get('cells')[-1],
                                          indices = parsed_output.get('structure').get('indices'),
                                          positions = parsed_output_red.get('positions')[-1],
                                          unwrapped_positions = parsed_output_red.get('positions')[-1]
                                          )
    else:
        raise ValueError(f"Unknown calcengine: {I.calcengine}. Must be 'LAMMPS' or 'VASP'")

    Energy_Minimization = parsed_output_red.get('energy_tot')[-1]
    Pressures_Minimization = parsed_output_red.get('pressures')[-1]
    ForceMax_Minimization = max([np.linalg.norm(ele) for ele in parsed_output_red.get('forces')[-1]])
    Cell_Minimization = parsed_output_red.get('cells')[-1]
    

    if return_as_dataclass:
        Output = I.copy()
        Output.Energy_Minimization = Energy_Minimization 
        Output.Pressures_Minimization = Pressures_Minimization 
        Output.ForceMax_Minimization = ForceMax_Minimization 
        Output.Cell_Minimization = Cell_Minimization 
        Output.Structure_Minimized = Structure_Minimized
    else:
        Output = {
                    'FinalEnergy':Energy_Minimization,
                    'FinalPressures':Pressures_Minimization,
                    'ForceMax':ForceMax_Minimization,
                    'FinalStructure':Structure_Minimized
                 }
    return Output
# This is a change

        




