import pyiron_workflow as pwf
from Helper_functions import _get_structure
from DataClassNode import WorkflowState, DataInputs
from typing import Union, Optional, Any

@pwf.as_function_node
def GeometryOptimization(I,
                        return_as_dataclass:bool = True):
    """
        Perform complete geometry optimization (cell and atomic positions) using LAMMPS or VASP.

        [... keep your existing docstring ...]

        Returns
        -------
        Output : WorkflowState or dict
            If return_as_dataclass=True:
                WorkflowState with fields:
                - RelaxedStructure : Structure
                    Relaxed atomic structure
                - acell_relaxed : ndarray
                    Relaxed lattice constants (Å)
                - Cell_RelaxedStructure : ndarray
                    Relaxed simulation cell vectors (Å)
                - Energy : float
                    Total energy of relaxed structure (eV)
                - Pressures : ndarray
                    Pressure tensor components (GPa)
                - ForceMax : float
                    Maximum force magnitude on any atom (eV/Å)
            If return_as_dataclass=False:
                Dictionary with keys: 'FinalEnergy', 'FinalPressures', 'ForceMax',
                'FinalStructure', 'LatticeConstants'
    """
    import numpy as np
    import pandas as pd
    import os
    
    name = 'CompleteGeometryOptimization'

    if I.calcengine.upper()=='LAMMPS':
        assert I.MinimizerForCompleteRelaxation not in ['quickmin', 'fire', 'hftn','cg/kk'], \
            "The quickmin, fire, hftn, and cg/kk styles do not yet support the use of the fix box/relax command or minimizations involving the electron radius in eFF models."

        from pyiron_atomistics.lammps.lammps import lammps_function
        
        if isinstance(I.InteratomicPotential, pd.DataFrame): 
            executable_version = 'conda'
            PotentialName = I.InteratomicPotential['Name']
        else:
            executable_version = None
            
        print("###############################################################################")
        print(f"Using the Interatomic potential defined in: {I.InteratomicPotential}")
        print("###############################################################################")
        
        _, parsed_output, job_crashed = lammps_function(
            working_directory=os.path.abspath(os.path.join(I.ProjectName, name)), 
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
        RelaxedStructure = _get_structure(
            structure=I.InputStructure, 
            cell=parsed_output_red.get('cells')[-1], 
            indices=parsed_output_red.get('indices')[-1],
            positions=parsed_output_red.get('positions')[-1],
            unwrapped_positions=parsed_output_red.get('unwrapped_positions')[-1]
        )
        
    elif I.calcengine.upper()=='VASP':
        from pyiron_atomistics.vasp.vasp import vasp_function
        
        print("###############################################################################")
        print(f"Setting Encut to {I.Encut} and using a Kmesh of {I.Kmesh}")
        print("###############################################################################")

        _, parsed_output, job_crashed = vasp_function(
            working_directory=os.path.abspath(os.path.join(I.ProjectName, name)),
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
        )
        
        parsed_output_red = parsed_output.get('generic')
        RelaxedStructure = _get_structure(
            structure=I.InputStructure, 
            cell=parsed_output.get('generic').get('cells')[-1], 
            indices=parsed_output.get('structure').get('indices'),
            positions=parsed_output.get('generic').get('positions')[-1],
            unwrapped_positions=parsed_output.get('generic').get('positions')[-1]
        )
    else:
        raise ValueError(f"Unknown calcengine: {I.calcengine}. Must be 'LAMMPS' or 'VASP'")
    
    # Extract results
    Cell_RelaxedStructure = parsed_output_red.get('cells')[-1]
    Energy_RelaxedStructure = parsed_output_red.get('energy_tot')[-1]
    Pressures_RelaxedStructure = parsed_output_red.get('pressures')[-1]
    ForceMax_RelaxedStructure = max([np.linalg.norm(ele) for ele in parsed_output_red.get('forces')[-1]])
    
    # Compute relaxed lattice constants
    if I.MillerIndices in [(0,0,1), (1,0,0), (0,1,0)] or not I.MillerIndices:
        key = '001'
    elif I.MillerIndices in [(0,1,1), (1,1,0), (1,0,1)]:
        key = '110'
    else:
        key = ''.join(str(i) for i in I.MillerIndices)

    dir_dict = {'001':[[1,0,0],[0,1,0],[0,0,1]],
               '110':[[0,0,1],[-1,1,0],[1,1,0]],
               '111':[[1,-1,0],[1,1,-2],[1,1,1]]}
    
    millermod = [np.linalg.norm(ele) for ele in dir_dict.get(key)]
    div_matrix = np.eye(3) * I.SuperCellDimensions * millermod
    acell_relaxed = Cell_RelaxedStructure @ np.linalg.inv(div_matrix)

    
    volume_per_atom = RelaxedStructure.get_volume() / len(RelaxedStructure)
    acell_calculated = (4 * volume_per_atom)**(1/3)

    # Return appropriate format
    if return_as_dataclass:
        Output = WorkflowState.dataclass(
        ProjectName=I.ProjectName,
        calcengine=I.calcengine,
        etol=I.etol,
        ftol=I.ftol,
        verbose=I.verbose,
        vacuum=I.vacuum,
        calctype=I.calctype,
        InteratomicPotential=I.InteratomicPotential,
        MinimizerForSurfaces=I.MinimizerForSurfaces,
        Encut=I.Encut,
        Kmesh=I.Kmesh,
        RelaxedStructure=RelaxedStructure,
        acell_relaxed=acell_calculated,
        Cell_RelaxedStructure=Cell_RelaxedStructure,
        Energy=Energy_RelaxedStructure,
        Pressures=Pressures_RelaxedStructure,
        ForceMax=ForceMax_RelaxedStructure,
        Cell=Cell_RelaxedStructure
    )
        if hasattr(I, 'DeformationMode'):
            Output.DeformationMode = I.DeformationMode
        
        # Output = create_workflow_state_from_geometry_optimization(
        #     I=I,
        #     RelaxedStructure=RelaxedStructure,
        #     acell_relaxed=acell_relaxed,
        #     Cell_RelaxedStructure=Cell_RelaxedStructure,
        #     Energy=Energy_RelaxedStructure,
        #     Pressures=Pressures_RelaxedStructure,
        #     ForceMax=ForceMax_RelaxedStructure
        # )
    else:
        Output = {
            'FinalEnergy': Energy_RelaxedStructure,
            'FinalPressures': Pressures_RelaxedStructure,
            'ForceMax': ForceMax_RelaxedStructure,
            'FinalStructure': RelaxedStructure,
            'LatticeConstants': acell_relaxed
        }

    return Output
    # parsed_output, RelaxedStructure, Energy_RelaxedStructure, Pressures_RelaxedStructure, ForceMax_RelaxedStructure, acell

@pwf.as_function_node
def ForceMinimization(I: Union[DataInputs.dataclass, WorkflowState.dataclass],
                      Structure_variable_name: str,
                      comment: str = '0',
                      poisson_relaxation: bool = False,
                      return_as_dataclass: bool = True):
    """
        Perform force minimization or static calculation on a structure.

        [... keep existing docstring but update Parameters section ...]

        Parameters
        ----------
        I : DataInputs or WorkflowState
            Input dataclass containing calculation parameters and structure
        
        [... rest of parameters ...]

        Returns
        -------
        Output : WorkflowState or dict
            If return_as_dataclass=True:
                WorkflowState with updated fields
            If return_as_dataclass=False:
                Dictionary with keys: 'FinalEnergy', 'FinalPressures', 'ForceMax',
                'FinalStructure'
    """
    import pandas as pd
    import numpy as np
    import os
    import copy
    
    name = f'Minimization_{Structure_variable_name}_{comment}' 
    Structure = getattr(I, Structure_variable_name)
    Structure = copy.deepcopy(Structure)

    print(f'Performing Calculation: {name}')

    if I.calcengine.upper() == 'LAMMPS':

        ###############################################################################
                                    # LAMMPS Stuff #
        ###############################################################################

        from pyiron_atomistics.lammps.lammps import lammps_function
        
        if isinstance(I.InteratomicPotential, pd.DataFrame): 
            executable_version = 'conda'
            PotentialName = I.InteratomicPotential['Name']
        else:
            executable_version = None
        
        if I.calctype.upper() in ['RELAX', 'RELAXED']:
            if poisson_relaxation:
                assert I.MinimizerForSurfaces not in ['quickmin', 'fire', 'hftn','cg/kk'], \
            "The quickmin, fire, hftn, and cg/kk styles do not yet support the use of the fix box/relax command or minimizations involving the electron radius in eFF models."
                calc_kwargs = {
                    "style": I.MinimizerForSurfaces,
                    "pressure": [None, 0.0, 0.0, None, None, None],
                    "ionic_energy_tolerance": I.etol,
                    "ionic_force_tolerance": I.ftol,
                    "n_print": 100000
                }
            else:
                calc_kwargs = {
                    "style": I.MinimizerForSurfaces,
                    "ionic_energy_tolerance": I.etol,
                    "ionic_force_tolerance": I.ftol,
                    "n_print": 100000
                }
            _, parsed_output, job_crashed = lammps_function(
                working_directory=os.path.abspath(os.path.join(I.ProjectName, name)), 
                structure=Structure,
                potential=I.InteratomicPotential, 
                calc_mode="minimize", 
                calc_kwargs=calc_kwargs, 
                cutoff_radius=None, 
                units="metal", 
                bonds_kwargs={}, 
                enable_h5md=False,
                executable_version=executable_version
            )
        elif I.calctype.upper() in ['STATIC', 'UNRELAXED']:
            _, parsed_output, job_crashed = lammps_function(
                working_directory=os.path.abspath(os.path.join(I.ProjectName, name)), 
                structure=Structure,
                potential=I.InteratomicPotential,
                calc_mode="static",
                calc_kwargs={},
                cutoff_radius=None, 
                units="metal", 
                bonds_kwargs={}, 
                enable_h5md=False,
                executable_version=executable_version
            )
            
        parsed_output_red = parsed_output.get('generic')

        Structure_Minimized = _get_structure(
            structure=Structure, 
            cell=parsed_output_red.get('cells')[-1], 
            indices=parsed_output_red.get('indices')[-1],
            positions=parsed_output_red.get('positions')[-1],
            unwrapped_positions=parsed_output_red.get('unwrapped_positions')[-1]
        )
        
    elif I.calcengine.upper() == 'VASP':

        from pyiron_atomistics.vasp.vasp import vasp_function

        ###############################################################################
                                        # VASP Stuff #
        ###############################################################################

        if I.calctype.upper() in ['RELAX', 'RELAXED']:
            _, parsed_output, job_crashed = vasp_function(
                working_directory=os.path.abspath(os.path.join(I.ProjectName, name)),
                structure=Structure,
                plane_wave_cutoff=I.Encut,
                kpoints_kwargs={"mesh": I.Kmesh},
                calc_mode="minimize",
                calc_kwargs={
                    "ionic_steps": 100,
                    "electronic_steps": 60,
                    "ionic_energy_tolerance": I.etol,
                    "ionic_force_tolerance": I.ftol
                },
                occupancy_smearing_kwargs={"smearing": 'MP', "width": 0.6},
                convergence_precision_kwargs={"electronic_energy": I.etol},
            )
        elif I.calctype.upper() in ['STATIC', 'UNRELAXED']:
            _, parsed_output, job_crashed = vasp_function(
                working_directory=os.path.abspath(os.path.join(I.ProjectName, name)),
                structure=Structure,
                plane_wave_cutoff=I.Encut,
                kpoints_kwargs={"mesh": I.Kmesh},
                calc_mode="static",
                calc_kwargs={},
                occupancy_smearing_kwargs={"smearing": 'MP', "width": 0.6},
                convergence_precision_kwargs={"electronic_energy": I.etol},
            )

        parsed_output_red = parsed_output.get('generic')
        Structure_Minimized = _get_structure(
            structure=Structure,
            cell=parsed_output_red.get('cells')[-1],
            indices=parsed_output.get('structure').get('indices'),
            positions=parsed_output_red.get('positions')[-1],
            unwrapped_positions=parsed_output_red.get('positions')[-1]
        )
    else:
        raise ValueError(f"Unknown calcengine: {I.calcengine}. Must be 'LAMMPS' or 'VASP'")

    Energy_Minimization = parsed_output_red.get('energy_tot')[-1]
    Pressures_Minimization = parsed_output_red.get('pressures')[-1]
    ForceMax_Minimization = max([np.linalg.norm(ele) for ele in parsed_output_red.get('forces')[-1]])
    Cell_Minimization = parsed_output_red.get('cells')[-1]
    
    if return_as_dataclass:
        # Check if input is WorkflowState or DataInputs
        if isinstance(I, WorkflowState.dataclass):
            # If WorkflowState, just copy and update
            Output = I.copy()
            Output.Structure_Minimized = Structure_Minimized
            Output.Energy = Energy_Minimization 
            Output.Pressures = Pressures_Minimization 
            Output.ForceMax = ForceMax_Minimization 
            Output.Cell = Cell_Minimization
        else:
            
            # If DataInputs, create a new WorkflowState
            Output = WorkflowState.dataclass(
                ProjectName=I.ProjectName,
                calcengine=I.calcengine,
                etol=I.etol,
                ftol=I.ftol,
                verbose=I.verbose,
                vacuum=I.vacuum,
                calctype=I.calctype,
                InteratomicPotential=I.InteratomicPotential,
                MinimizerForSurfaces=I.MinimizerForSurfaces,
                Encut=I.Encut,
                Kmesh=I.Kmesh,
                Structure_Minimized=Structure_Minimized,
                Energy=Energy_Minimization,
                Pressures=Pressures_Minimization,
                ForceMax=ForceMax_Minimization,
                Cell=Cell_Minimization
            )
            # After creating your output dataclass instance
        if hasattr(I, 'DeformationMode'):
            Output.DeformationMode = I.DeformationMode
    else:
        Output = {
            'FinalEnergy': Energy_Minimization,
            'FinalPressures': Pressures_Minimization,
            'ForceMax': ForceMax_Minimization,
            'FinalStructure': Structure_Minimized
        }
    
    return Output
# I am making this change now and now

        




