import pyiron_workflow as pwf
import pandas as pd
from DataClassNode import DataInputs, WorkflowState, GammaSurfaceInputs
from typing import Union, Optional, Any


@pwf.as_function_node
def WriteSurfaceStressArchive(
        WrappedDict: dict,
        I: DataInputs.dataclass,
        GeomOptOutput: Union[DataInputs.dataclass, WorkflowState.dataclass],
        workflow_type: str,
        inspect: bool = True,
        Overwrite: bool = False,
        debug: bool = False
) -> dict:
    """
    Archive surface stress results using structured format.

    Args:
        WrappedDict: Dictionary containing strain point calculations
        I: DataInputs dataclass with calculation parameters
        GeomOptOutput: Geometry optimization output
        workflow_type: Type of workflow (e.g., 'SurfaceStress')
        inspect: Whether to create plots
        Overwrite: Whether to overwrite existing entries
        debug: Debug mode flag
    """
    import numpy as np
    from datetime import datetime
    from AnalyseAndArchiveFunctions import compute_surface_stress_from_fit, compute_surface_stress_from_pressures

    # Prepare data from inputs
    RelaxedCell = np.array(GeomOptOutput.RelaxedStructure.cell.tolist())
    CellwSurfaces = WrappedDict.get(0.0, {}).get('FinalCell')
    Pressure_tensor = WrappedDict.get(0.0, {}).get('PressureTensor')
    SurfaceArea = WrappedDict.get(0.0, {}).get('Area')

    # 1. Compute surface stress from fitting
    fit_results = compute_surface_stress_from_fit(
        WrappedDict=WrappedDict,
        SurfaceArea=SurfaceArea,
        inspect=inspect,
        ProjectName=I.ProjectName
    )

    # 2. Compute surface stress from pressures
    tau_from_pressures = compute_surface_stress_from_pressures(
        Pressure_tensor=Pressure_tensor,
        RelaxedCell=RelaxedCell,
        CellwSurfaces=CellwSurfaces,
        SurfaceArea=SurfaceArea
    )

    # 3. Calculate method agreement
    tau_difference = np.abs(fit_results['tau'] - tau_from_pressures)
    methods_agree = np.isclose(fit_results['tau'], tau_from_pressures, atol=1e-3)

    if methods_agree:
        WorkflowComment = f'Fit and pressure methods agree within 1e-3 eV/Å²'
    else:
        WorkflowComment = f'Methods differ by {tau_difference:.6f} eV/Å² - check convergence'

    # Prepare structured entry
    entry = {
        'StructuralInputs': {
            'Element': I.Element,
            'MillerIndices': I.MillerIndices,
            'SuperCellDimensions': I.SuperCellDimensions,
        },

        'CalculationInputs': {
            'InteratomicPotential': I.InteratomicPotential,
            'calcengine': I.calcengine,
            'calctype': I.calctype,
            'Encut': I.Encut,
            'Kmesh': I.Kmesh,
            'minimizer_surfaces': I.MinimizerForSurfaces,
            'minimizer_box': I.MinimizerForCompleteRelaxation,
            'etol': I.etol,
            'ftol': I.ftol,
            'vacuum': I.vacuum,
        },

        'LoopOutputs': WrappedDict,

        'BenchmarkOutputs': {
            'acell_relaxed': GeomOptOutput.acell_relaxed,
            'ReferenceCell': RelaxedCell,
            'Volume': RelaxedCell[2, 2] * SurfaceArea,
            'Lx': RelaxedCell[0, 0],
            'Ly': RelaxedCell[1, 1],
            'Lz': RelaxedCell[2, 2],
            'Lx_surfaces': CellwSurfaces[0, 0],
            'Ly_surfaces': CellwSurfaces[1, 1],
            'Lz_surfaces': CellwSurfaces[2, 2],
            'fmax_GeomOpt': GeomOptOutput.ForceMax,
            'fmax_OpenSurface': WrappedDict.get(0.0, {}).get('ForceMax'),
            'ConvergenceStatus': GeomOptOutput.converged if hasattr(GeomOptOutput, 'converged') else None,
            'StrainPointsComputed': len(WrappedDict),
            'FitQuality_R2': fit_results.get('r_squared', None),
            'MethodAgreement': methods_agree,
            'MethodAgreement_difference': tau_difference,
        },

        'ExtractedOutputs': {
            # Primary results from fitting method
            'tau': fit_results['tau'],
            'gamma': fit_results['gamma'],
            'gamma_surface': WrappedDict.get(0.0, {}).get('gamma'),

            # Alternative method result
            'tau_from_pressures': tau_from_pressures,

            # Fit errors
            'tau_err': fit_results['tau_err'],
            'gamma_err': fit_results['gamma_err'],

            # Pressure tensor
            'Pressure_Tensor': Pressure_tensor,
            'SurfaceArea': SurfaceArea,

            # Method agreement
            'MethodsAgree': methods_agree,
            'MethodDifference': tau_difference,
        },

        'ProjectName': I.ProjectName,
        'UserComment': I.UserComment,
        'WorkflowComment': WorkflowComment,
        'WorkflowType': 'SurfaceStress',
        'timestamp': datetime.now(),
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Surface Stress Results for {I.Element} {I.MillerIndices}:")
    print(f"{'=' * 60}")
    print(f"  Surface stress (τ from fit): {fit_results['tau']:.6f} eV/Å²")
    print(f"  Surface energy (γ): {fit_results['gamma']:.6f} eV/Å²")
    print(f"  Surface stress (τ from pressures): {tau_from_pressures:.6f} eV/Å²")
    print(f"  Method agreement: {'Yes' if methods_agree else 'No'}")
    if not methods_agree:
        print(f"  Difference: {tau_difference:.6f} eV/Å²")
    print(f"{'=' * 60}\n")

    # Archive the results using unified function
    return archive_results(
        entry_data=entry,
        I=I,
        archive_path=I.StorageDataFrame,
        workflow_type=workflow_type,
        Overwrite=Overwrite
    )


@pwf.as_function_node
def MakeFits(WrappedDict:dict, 
             FullyRelaxOutputDataClass:Union[DataInputs.dataclass, WorkflowState.dataclass],
            deformation_axis:int=0):
    
    from AnalyseAndArchiveFunctions import extract_elastic_moduli
    import numpy as np
    
    diction = WrappedDict.copy()
    a = FullyRelaxOutputDataClass.Cell_RelaxedStructure
    V0 = np.dot(a[0],np.cross(a[1],a[2]))

    strains = sorted(diction.keys())
    PressureTensors = np.array([diction[k]["FinalPressures"] for k in sorted(diction.keys())])
    FinalCells = np.array([ np.array(diction[k]["FinalStructure"].cell) for k in sorted(diction.keys())])
    volume = []
    for ele in FinalCells:
        volume.append(np.dot(ele[0],np.cross(ele[1],ele[2])))
    Energies = np.array([diction[k]["FinalEnergy"] for k in sorted(diction.keys())])
    RelaxedCell = np.array(FullyRelaxOutputDataClass.RelaxedStructure.cell)

    result_dict = extract_elastic_moduli(energies=Energies,
                          pressure_tensors=PressureTensors,
                          final_cells=FinalCells,
                          reference_cell=RelaxedCell,
                          deformation_axis=deformation_axis,
                          V0 = V0,
                          degree = 4,
                          slice_min=None,
                          slice_max=None)
    return result_dict


@pwf.as_function_node
def WriteElasticModuliArchive(
    moduli_results: dict,
    I: DataInputs.dataclass,
    GeomOptOutput: Union[DataInputs.dataclass, WorkflowState.dataclass],
    workflow_type:str,
    LoopOutputs: dict,
    inspect: bool = True,
    Overwrite: bool = False,
    debug: bool = False
) -> dict:
    """
    Archive elastic moduli results using structured format.
    
    Args:
        moduli_results: Output from extract_elastic_moduli function
        I: ElasticModuliInputs dataclass
        GeomOptOutput: Geometry optimization output
        LoopOutputs: Dictionary containing all strain point calculations
        inspect: Whether plots were created
        Overwrite: Whether to overwrite existing entries
        debug: Debug mode flag
    """
    import numpy as np
    from datetime import datetime
    #from AnalyseAndArchiveFunctions import archive_results
    
    # Prepare data from inputs
    RelaxedCell = np.array(GeomOptOutput.RelaxedStructure.cell.tolist())
    
    # Get deformation axis (with fallback)
    if hasattr(I, 'DeformationAxis'):
        DeformationAxis = I.DeformationAxis
    else:
        DeformationAxis = 0
    
    # Generate workflow comment comparing methods
    energy_c2 = moduli_results['energy']['constants']['C2']
    stress_c2 = moduli_results['stress']['constants']['C2']
    diff_percent_E = 100 * np.abs(energy_c2 - stress_c2) / np.mean([energy_c2, stress_c2])

    energy_c3 = moduli_results['energy']['constants']['C3']
    stress_c3 = moduli_results['stress']['constants']['C3']
    diff_percent_D = 100 * np.abs(energy_c3 - stress_c3) / np.mean([energy_c3, stress_c3])
    
    if diff_percent_D < 2.0 and diff_percent_E<0.5:
        WorkflowComment = f'Energy and stress methods agree within {diff_percent_D:.3f}% and E agrees with {diff_percent_E:.3f}%'
    else:
        WorkflowComment = f'Methods differ by {diff_percent:.3f}% - check fit quality'
    
    # Prepare structured entry
    entry = {
        'StructuralInputs': {
            'Element': I.Element,
            'MillerIndices': I.MillerIndices,
            'SuperCellDimensions': I.SuperCellDimensions,
        },
        
        'CalculationInputs': {
            'InteratomicPotential': I.InteratomicPotential,
            'calcengine': I.calcengine,
            'calctype': I.calctype,
            'Encut': I.Encut,
            'Kmesh': I.Kmesh,
            'minimizer': I.MinimizerForSurfaces,
            'etol': I.etol,
            'ftol': I.ftol,
            'DeformationAxis': DeformationAxis,
            'StrainRange': (I.StrainMinimum, I.StrainMaximum),
            # Add these if they exist:
            # 'DeformationType': I.DeformationType if hasattr(I, 'DeformationType') else None,
            # 'PolynomialDegree': I.PolynomialDegree if hasattr(I, 'PolynomialDegree') else None,
        },
        
        'LoopOutputs': LoopOutputs,
        
        'BenchmarkOutputs': {
            'acell_relaxed': GeomOptOutput.acell_relaxed,
            'ReferenceCell': RelaxedCell,
            'Volume': np.linalg.det(RelaxedCell),
            'Lx': RelaxedCell[0, 0],
            'Ly': RelaxedCell[1, 1],
            'Lz': RelaxedCell[2, 2],
            'fmax_GeomOpt': GeomOptOutput.ForceMax,
            'ConvergenceStatus': GeomOptOutput.converged if hasattr(GeomOptOutput, 'converged') else None,
            'StrainPointsComputed': len(LoopOutputs),
            'FitQuality_Energy_R2': moduli_results['energy']['r_squared'],
            'FitQuality_Stress_R2': moduli_results['stress']['r_squared'],
            'MethodAgreement_percent_E': diff_percent_E,
            'MethodAgreement_percent_D': diff_percent_D,
        },
        
        'ExtractedOutputs': {
            # Full moduli results from both methods
            'Moduli': moduli_results,
            
            # Primary results (most commonly queried) - from stress method
            'C2': moduli_results['stress']['constants']['C2'],
            'C3': moduli_results['stress']['constants']['C3'],
            'C4': moduli_results['stress']['constants']['C4'],
            
            # Also store energy method for comparison
            'C2_energy': moduli_results['energy']['constants']['C2'],
            'C3_energy': moduli_results['energy']['constants']['C3'],
            'C4_energy': moduli_results['energy']['constants']['C4'],
            
            # Errors
            'C2_err': moduli_results['stress']['errors']['C2_err'],
            'C3_err': moduli_results['stress']['errors']['C3_err'],
            'C4_err': moduli_results['stress']['errors']['C4_err'],
            
            # Method agreement
            'MethodsAgree': diff_percent_D < 1.0,
        },
        
        'ProjectName': I.ProjectName,
        'UserComment': I.UserComment,
        'WorkflowComment': WorkflowComment,
        'WorkflowType': 'NonlinearModuli',
        'timestamp': datetime.now(),
    }
    
    # Archive the results using unified function
    return archive_results(
        entry_data=entry,
        I=I,
        archive_path=I.StorageDataFrame,
        workflow_type=workflow_type, #'NonlinearModuli',
        Overwrite=Overwrite
    )


@pwf.as_function_node
def ProcessFaultEnergies(
        LoopOutputs: pd.DataFrame,
        a_lattice: float,
        save_structures: bool = True,
        structure_filename: str = 'GSFE_Structures.xyz',
        #    element: str = None,
        #    miller_indices: list = None
) -> dict:
    """
    Process LoopOutputs from fault energy calculations to extract key values.

    Args:
        LoopOutputs: Dict containing 'fraction', 'gamma', and 'DeformedStructure_final'
        a_lattice: Lattice parameter for displacement calculation
        save_structures: Whether to save structures to file
        structure_filename: Custom filename for structures (optional)
        element: Element name for filename generation (optional)
        miller_indices: Miller indices for filename generation (optional)

    Returns:
        Dictionary with SSFE, USFE, SSFE_displacement, USFE_displacement
    """
    import numpy as np
    import ase.io as aio
    import pandas as pd

    # Convert to arrays for easier processing
    fractions = np.array(LoopOutputs['fraction'])
    gamma = np.array(LoopOutputs['gamma'])
    structures = LoopOutputs['DeformedStructure_final']

    # 1. Find USFE (maximum gamma value)
    usfe_idx = np.argmax(gamma)
    USFE = gamma[usfe_idx]
    usfe_fraction = fractions[usfe_idx]
    USFE_displacement = usfe_fraction * a_lattice / np.sqrt(6)

    # 2. Find SSFE (minimum gamma after USFE point)
    # Only consider points after USFE
    post_usfe_gamma = gamma[usfe_idx:]
    post_usfe_fractions = fractions[usfe_idx:]

    if len(post_usfe_gamma) > 1:
        # Find minimum in the post-USFE region
        ssfe_relative_idx = np.argmin(post_usfe_gamma)
        SSFE = post_usfe_gamma[ssfe_relative_idx]
        ssfe_fraction = post_usfe_fractions[ssfe_relative_idx]
    else:
        # If USFE is the last point, SSFE equals USFE
        print("Warning: USFE is at the last point, cannot find SSFE after it")
        SSFE = USFE
        ssfe_fraction = usfe_fraction

    SSFE_displacement = ssfe_fraction * a_lattice / np.sqrt(6)

    # 3. Save structures if requested
    if save_structures:
        if structure_filename is None:
            # Generate filename from element and miller indices if provided
            # if element and miller_indices:
            #     miller_str = ''.join(map(str, miller_indices))
            #     structure_filename = f'GSFE_Structures_{element}_{miller_str}.xyz'
            # else:
            structure_filename = 'GSFE_Structures.xyz'

        # Save all structures to file
        aio.write(structure_filename, structures, format='extxyz')
        print(f"Saved {len(structures)} structures to {structure_filename}")

    # Create results dictionary
    fault_results = {
        'SSFE': SSFE,
        'USFE': USFE,
        'SSFE_displacement': SSFE_displacement,
        'USFE_displacement': USFE_displacement,
        # Optional: include additional info
        'SSFE_fraction': ssfe_fraction,
        'USFE_fraction': usfe_fraction  # ,
        # 'SFE_barrier': USFE - SSFE,
    }

    # Print summary
    print(f"\nFault Energy Results:")
    print(f"  USFE: {USFE:.4f} at fraction {usfe_fraction:.3f} eV/A^2 (displacement: {USFE_displacement:.4f} Å)")
    print(f"  SSFE: {SSFE:.4f} at fraction {ssfe_fraction:.3f} eV/A^2 (displacement: {SSFE_displacement:.4f} Å)")
    # print(f"  Barrier: {USFE - SSFE:.4f}")

    return fault_results

@pwf.as_function_node
def WriteGammaSurfaceArchive(
    LoopOutputs: pd.DataFrame,
    I: GammaSurfaceInputs.dataclass,
    GeomOptOutput: Union[DataInputs.dataclass, WorkflowState.dataclass],
    workflow_type: str,
    save_structures: bool = True,
    inspect: bool = True,
    Overwrite: bool = False,
    debug: bool = False
) -> dict:
    """
    Archive stacking fault energy results using structured format.
    
    Args:
        LoopOutputs: Dictionary containing fraction, gamma, and DeformedStructure_final
        I: DataInputs dataclass with calculation parameters
        GeomOptOutput: Geometry optimization output
        workflow_type: Type of workflow (e.g., 'StackingFault')
        save_structures: Whether to save structure files
        inspect: Whether plots were created
        Overwrite: Whether to overwrite existing entries
        debug: Debug mode flag
    """
    import numpy as np
    from datetime import datetime
    import ase.io as aio
    import os
    
    # Process the loop outputs to extract fault energies
    # Extract arrays from LoopOutputs
    fractions = np.array(LoopOutputs['fraction'])
    gamma = np.array(LoopOutputs['gamma'])
    structures = LoopOutputs['DeformedStructure_final']
    
    # Get lattice parameter for displacement calculation
    a_lattice = GeomOptOutput.acell_relaxed
    
    # 1. Find USFE (maximum gamma value)
    usfe_idx = np.argmax(gamma)
    USFE = gamma[usfe_idx]
    usfe_fraction = fractions[usfe_idx]
    USFE_displacement = usfe_fraction * a_lattice / np.sqrt(6)
    
    # 2. Find SSFE (minimum gamma after USFE point)
    post_usfe_gamma = gamma[usfe_idx:]
    post_usfe_fractions = fractions[usfe_idx:]
    
    if len(post_usfe_gamma) > 1:
        ssfe_relative_idx = np.argmin(post_usfe_gamma)
        SSFE = post_usfe_gamma[ssfe_relative_idx]
        ssfe_fraction = post_usfe_fractions[ssfe_relative_idx]
    else:
        print("Warning: USFE is at the last point, cannot find SSFE after it")
        SSFE = USFE
        ssfe_fraction = usfe_fraction
    
    SSFE_displacement = ssfe_fraction * a_lattice / np.sqrt(6)
    
    # 3. Save structures if requested
    if save_structures:
        miller_str = ''.join(map(str, I.MillerIndices))
        structure_filename = f'GSFE_Structures_{I.Element}_{miller_str}_{I.UserComment}.xyz'
        structure_filepath = os.path.join(I.ProjectName, structure_filename)
        aio.write(structure_filepath, structures, format='extxyz')
        print(f"Saved {len(structures)} structures to {structure_filepath}")
    
    # Calculate derived quantities
    SFE_barrier = USFE - SSFE
    
    # Generate workflow comment
    if SFE_barrier > 0:
        WorkflowComment = f'GSFE calculation: USFE={USFE:.4f}, SSFE={SSFE:.4f}, Barrier={SFE_barrier:.4f}'
    else:
        WorkflowComment = f'Warning: Negative barrier detected. USFE={USFE:.4f}, SSFE={SSFE:.4f}'
    
    # Prepare data from inputs
    RelaxedCell = np.array(GeomOptOutput.RelaxedStructure.cell.tolist())
    
    # Prepare structured entry
    entry = {
        'StructuralInputs': {
            'Element': I.Element,
            'MillerIndices': I.MillerIndices,
            'SuperCellDimensions': I.SuperCellDimensions,
        },
        
        'CalculationInputs': {
            'InteratomicPotential': I.InteratomicPotential,
            'calcengine': I.calcengine,
            'calctype': I.calctype,
            'Encut': I.Encut,
            'Kmesh': I.Kmesh,
            'minimizer': I.MinimizerForSurfaces,
            'etol': I.etol,
            'ftol': I.ftol,
            'FractionMinimum': I.FractionMinimum,
            'FractionMaximum': I.FractionMaximum,
            'FractionStateCount': I.FractionStateCount,
        },
        
        'LoopOutputs': LoopOutputs,
        
        'BenchmarkOutputs': {
            'acell_relaxed': GeomOptOutput.acell_relaxed,
            'ReferenceCell': RelaxedCell,
            'Volume': np.linalg.det(RelaxedCell),
            'Lx': RelaxedCell[0, 0],
            'Ly': RelaxedCell[1, 1],
            'Lz': RelaxedCell[2, 2],
            'fmax_GeomOpt': GeomOptOutput.ForceMax,
            'ConvergenceStatus': GeomOptOutput.converged if hasattr(GeomOptOutput, 'converged') else None,
            'FractionPointsComputed': len(fractions),
            'MinimumEnergyFraction': ssfe_fraction,
            'MaximumEnergyFraction': usfe_fraction,
            'StructuresSaved': save_structures,
        },
        
        'ExtractedOutputs': {
            # Primary results
            'SSFE': SSFE,
            'USFE': USFE,
            'SSFE_displacement': SSFE_displacement,
            'USFE_displacement': USFE_displacement,
            
            # Additional useful quantities
            'SSFE_fraction': ssfe_fraction,
            'USFE_fraction': usfe_fraction,
            'SFE_barrier': SFE_barrier,
            'SFE_ratio': SSFE / USFE if USFE != 0 else None,
            
            # Data quality indicators
            'gamma_min': np.min(gamma),
            'gamma_max': np.max(gamma),
            'gamma_range': np.max(gamma) - np.min(gamma),
        },
        
        'ProjectName': I.ProjectName,
        'UserComment': I.UserComment,
        'WorkflowComment': WorkflowComment,
        'WorkflowType': 'StackingFault',
        'timestamp': datetime.now(),
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Fault Energy Results for {I.Element} {I.MillerIndices}:")
    print(f"{'='*60}")
    print(f"  USFE: {USFE:.4f} at fraction {usfe_fraction:.3f} (displacement: {USFE_displacement:.4f} Å)")
    print(f"  SSFE: {SSFE:.4f} at fraction {ssfe_fraction:.3f} (displacement: {SSFE_displacement:.4f} Å)")
    print(f"  Barrier: {SFE_barrier:.4f}")
    print(f"  Ratio (SSFE/USFE): {SSFE/USFE:.3f}" if USFE != 0 else "  Ratio: N/A (USFE=0)")
    print(f"{'='*60}\n")
    
    # Archive the results using unified function
    return archive_results(
        entry_data=entry,
        I=I,
        archive_path=I.StorageDataFrame,
        workflow_type=workflow_type,
        Overwrite=Overwrite
    )

def archive_results(
    entry_data: dict,
    I: DataInputs.dataclass,
    archive_path: str,
    workflow_type: str,
    Overwrite: bool = False
) -> dict:
    """
    Unified archiving function for all workflow types.
    
    Args:
        entry_data: Structured dictionary with StructuralInputs, CalculationInputs, etc.
        I: DataInputs dataclass containing input parameters
        archive_path: Path to pickle file for storage
        workflow_type: 'SurfaceStress', 'NonlinearModuli', 'StackingFault', etc.
        Overwrite: If True, replace duplicate entries
        
    Returns:
        entry_data: The entry that was added/updated
    """
    import pandas as pd
    import os
    
    # Validate entry structure
    required_keys = ['StructuralInputs', 'CalculationInputs', 'LoopOutputs', 
                     'BenchmarkOutputs', 'ExtractedOutputs', 'WorkflowType']
    
    missing_keys = [key for key in required_keys if key not in entry_data]
    if missing_keys:
        raise ValueError(f"Entry missing required keys: {missing_keys}")
    
    # Load or create DataFrame
    if os.path.isfile(archive_path):
        df = pd.read_pickle(archive_path)
        print(f"Loaded existing archive with {len(df)} entries")
    else:
        df = pd.DataFrame()
        print("Creating new archive")
    
    # Check for duplicate entries
    is_duplicate, duplicate_indices = _check_for_duplicates_structured(df, entry_data)
    
    # Handle adding/updating entries
    if not is_duplicate:
        # No match found, add new entry
        df = pd.concat([df, pd.DataFrame([entry_data])], ignore_index=True)
        print(f"✓ New {workflow_type} entry added to archive")
        
    elif Overwrite:
        # Match found and Overwrite is True
        print(f"⚠ Duplicate {workflow_type} entry found at indices: {duplicate_indices}")
        print("  Overwriting existing entry...")
        df = df.drop(duplicate_indices)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, pd.DataFrame([entry_data])], ignore_index=True)
        print("✓ Entry overwritten successfully")
        
    else:
        # Match found but Overwrite is False
        print(f"\n{'='*60}")
        print(f"⚠ {workflow_type} entry already exists with matching parameters:")
        print(f"{'='*60}")
        _print_duplicate_info(entry_data, duplicate_indices)
        print(f"\nSet Overwrite=True to replace the existing entry.")
        print(f"{'='*60}\n")
    
    # Save the dataframe
    df.to_pickle(archive_path)
    
    # Also save local copy in project directory
    local_path = os.path.join(I.ProjectName, f'Local_{workflow_type}_archive.pckl')
    df.to_pickle(local_path)
    print(f"✓ Archive saved to: {archive_path}")
    print(f"✓ Local copy saved to: {local_path}")
    
    return entry_data


def _check_for_duplicates_structured(df: pd.DataFrame, new_entry: dict) -> tuple:
    """
    Check if an entry with matching StructuralInputs and CalculationInputs exists.
    
    Returns:
        (is_duplicate, list_of_matching_indices)
    """
    import numpy as np
    
    if df.empty:
        return False, []
    
    # Extract what we're comparing
    new_struct = new_entry['StructuralInputs']
    new_calc = new_entry['CalculationInputs']
    
    matching_indices = []
    
    for idx, row in df.iterrows():
        # Compare structural inputs
        struct_match = _compare_dicts(row['StructuralInputs'], new_struct)
        
        # Compare calculation inputs
        calc_match = _compare_dicts(row['CalculationInputs'], new_calc)
        
        if struct_match and calc_match:
            matching_indices.append(idx)
    
    is_duplicate = len(matching_indices) > 0
    
    if is_duplicate:
        print(f"\n{'='*60}")
        print(f"Found {len(matching_indices)} matching entry(ies)")
        print(f"{'='*60}")
    
    return is_duplicate, matching_indices


def _compare_dicts(dict1: dict, dict2: dict, tolerance: float = 1e-6) -> bool:
    """
    Compare two dictionaries with tolerance for numerical values.
    """
    import numpy as np
    
    if dict1.keys() != dict2.keys():
        return False
    
    for key in dict1.keys():
        val1, val2 = dict1[key], dict2[key]
        
        # Handle None values
        if val1 is None and val2 is None:
            continue
        if val1 is None or val2 is None:
            return False
        
        # Handle numerical scalars
        if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
            if not np.isclose(val1, val2, rtol=tolerance):
                return False
        
        # Handle numpy arrays
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.allclose(val1, val2, rtol=tolerance):
                return False
        
        # Handle lists/tuples (including nested ones)
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if not _compare_sequences(val1, val2, tolerance):
                return False
        
        # Handle everything else with equality
        else:
            if val1 != val2:
                return False
    
    return True


def _compare_sequences(seq1, seq2, tolerance: float = 1e-6) -> bool:
    """Helper function to compare sequences (lists/tuples) recursively."""
    import numpy as np
    
    if len(seq1) != len(seq2):
        return False
    
    for v1, v2 in zip(seq1, seq2):
        if isinstance(v1, (int, float, np.number)) and isinstance(v2, (int, float, np.number)):
            if not np.isclose(v1, v2, rtol=tolerance):
                return False
        elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            if not _compare_sequences(v1, v2, tolerance):
                return False
        elif v1 != v2:
            return False
    
    return True


def _print_duplicate_info(entry: dict, indices: list):
    """Print information about duplicate entries."""
    struct = entry['StructuralInputs']
    calc = entry['CalculationInputs']
    
    print("\nStructural Parameters:")
    for key, val in struct.items():
        print(f"  {key}: {val}")
    
    print("\nCalculation Parameters:")
    for key, val in calc.items():
        print(f"  {key}: {val}")
    
    print(f"\nMatching entry indices: {indices}")