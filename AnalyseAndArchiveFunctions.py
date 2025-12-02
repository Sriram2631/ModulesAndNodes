import numpy as np
from DataClassNode import DataInputs, WorkflowState, GammaSurfaceInputs
import pyiron_workflow as pwf
from dataclasses import field
from typing import Union, Optional, Any
import pandas as pd
import numpy as np
import os
import ase

def create_fault_energy_archive_entry(
    I: DataInputs.dataclass,
    GeomOptOutput: Union[DataInputs.dataclass, WorkflowState.dataclass],
    LoopOutputs: dict,
    fault_results: dict,
    workflow_comment: str = None
) -> dict:
    """
    Create archive entry for stacking fault energy calculations.
    
    Args:
        I: DataInputs containing calculation parameters
        GeomOptOutput: Geometry optimization results
        LoopOutputs: Raw data from fault energy calculations
        fault_results: Dictionary containing SSFE, USFE, and displacement values
        workflow_comment: Optional workflow-specific comment
        
    Returns:
        Structured entry dictionary for archiving
    """
    
    entry = {
        'StructuralInputs': {
            'Element': I.Element,
            'MillerIndices': I.MillerIndices,  # Fault plane
            'SuperCellDimensions': I.SuperCellDimensions,
            # Add any fault-specific structural parameters if needed
            # 'FaultVector': I.FaultVector,  # if you have this
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
            # Fault-specific calculation parameters
            'FractionMinimum': I.FractionMinimum,
            'FractionMaximum': I.FractionMaximum,
            'FractionStateCount': I.FractionStateCount,
        },
        
        'LoopOutputs': LoopOutputs,  # Raw fraction vs energy data
        
        'BenchmarkOutputs': {
            # Geometry optimization quality
            'fmax_GeomOpt': GeomOptOutput.ForceMax,
            'acell_relaxed': GeomOptOutput.acell_relaxed,
            'cell_relaxed': GeomOptOutput.RelaxedCell,
            'volume_relaxed': np.linalg.det(GeomOptOutput.RelaxedCell),
            'ConvergenceStatus': GeomOptOutput.ForceMax < I.ftol,
            # Could add fault-specific benchmarks
            'MinimumEnergyDisplacement': fault_results.get('SSFE_displacement'),
        },
        
        'ExtractedOutputs': {
            'SSFE': fault_results['SSFE'],  # Stable stacking fault energy
            'USFE': fault_results['USFE'],  # Unstable stacking fault energy
            'SSFE_displacement': fault_results['SSFE_displacement'],  # Displacement at SSFE
            'USFE_displacement': fault_results['USFE_displacement'],  # Displacement at USFE
            # Optional: could add more derived quantities
            # 'SFE_barrier': fault_results['USFE'] - fault_results['SSFE'],
            # 'SFE_ratio': fault_results['SSFE'] / fault_results['USFE'] if fault_results['USFE'] != 0 else None,
        },
        
        'ProjectName': I.ProjectName,
        'UserComment': I.UserComment,
        'WorkflowComment': workflow_comment or f"Stacking fault energy for {I.Element} {I.MillerIndices}",
        'WorkflowType': 'StackingFault',
        'timestamp': datetime.now().isoformat(),
    }
    
    return entry

def archive_fault_energy_results(
    fault_results: dict,
    I: DataInputs.dataclass,
    GeomOptOutput: Union[DataInputs.dataclass, WorkflowState.dataclass],
    LoopOutputs: dict,
    Overwrite: bool = False
) -> dict:
    """
    Archive stacking fault energy calculation results.
    
    Args:
        fault_results: Dict with SSFE, USFE, SSFE_displacement, USFE_displacement
        I: Input parameters dataclass
        GeomOptOutput: Geometry optimization results
        LoopOutputs: Raw calculation data
        Overwrite: Whether to overwrite existing entries
        
    Returns:
        The archived entry
    """
    
    # Validate fault_results has required keys
    required_keys = ['SSFE', 'USFE', 'SSFE_displacement', 'USFE_displacement']
    missing = [k for k in required_keys if k not in fault_results]
    if missing:
        raise ValueError(f"fault_results missing required keys: {missing}")
    
    # Create structured entry
    entry = create_fault_energy_archive_entry(
        I=I,
        GeomOptOutput=GeomOptOutput,
        LoopOutputs=LoopOutputs,
        fault_results=fault_results
    )
    
    # Use the unified archiving function
    return archive_results(
        entry_data=entry,
        I=I,
        archive_path=I.StorageDataFrame,
        workflow_type='StackingFault',
        Overwrite=Overwrite
    )

def archive_elastic_moduli_results(
    entry_data: dict,
    I: DataInputs.dataclass,
    Overwrite: bool = False
) -> dict:
    """
    Archive elastic moduli calculation results to a DataFrame stored as pickle.
    """
    import numpy as np
    import pandas as pd
    import os
    from AnalyseAndArchiveFunctions import _check_for_duplicates
    
    COLUMNS = [
        # Material properties
        'Element', 'MillerIndices', 'acell', 'Lx', 'Ly', 'Lz', 'Volume',
        
        # Results (stored as nested dict)
        'Moduli',
        
        # Input parameters
        'DeformationAxis', 'StrainRange', 'PolynomialDegree',
        'Method', 'Engine', 'Potential_ID', 'Encut', 'Kmesh',
        'ProjectName', 'UserComment', 'SuperCellDimensions',
        'minimizer', 'etol', 'ftol', 'fmax_GeomOpt',
        
        # Storage
        'ReferenceCell', 'LoopOutputs', 'WorkflowComment'
    ]
    
    # Load or create DataFrame
    if os.path.isfile(I.StorageDataFrame):
        df = pd.read_pickle(I.StorageDataFrame)
        
        # Fix duplicate columns if they exist
        if df.columns.duplicated().any():
            print("Warning: Duplicate columns detected. Removing duplicates...")
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Ensure backward compatibility - add missing columns
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
                print(f"Added missing column: {col}")
    else:
        df = pd.DataFrame(columns=COLUMNS)
    
    # Check for duplicate entries
    mask = _check_for_duplicates(df, I)
    
    # Handle adding/updating entries
    if not mask.any():
        df.loc[len(df)] = entry_data
        print("New entry added to the archive.")
    elif Overwrite:
        print("Entry already recorded. Overwriting...")
        matching_indices = df.index[mask].tolist()
        print(f"Removing existing entries at indices: {matching_indices}")
        df = df.drop(matching_indices)
        df.reset_index(drop=True, inplace=True)
        df.loc[len(df)] = entry_data
        print("Entry overwritten successfully.")
    else:
        print("---> Entry already recorded with the same parameters:")
        print("     Element, MillerIndices, SuperCellDimensions, DeformationAxis,")
        print("     StrainRange, InteratomicPotential, Encut, Kmesh,")
        print("     and UserComment match existing entry.")
        print("     Set Overwrite=True to delete previous entry and archive the present values.")
    
    # Save the dataframe
    df.to_pickle(I.StorageDataFrame)
    df.to_pickle(os.path.join(I.ProjectName, 'Local_storage.pckl'))
    
    return entry_data

def compute_surface_stress_from_fit(
    WrappedDict: dict,
    SurfaceArea: float,
    inspect: bool = True,
    ProjectName: str = None
) -> dict:
    """
    Compute surface stress (tau) and surface energy (gamma) from polynomial fitting
    of strain-energy data.
    
    Returns:
        dict with keys: 'tau', 'gamma', 'tau_err', 'gamma_err', 'strains', 'Amulgamma', 'coeffs', 'deg'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Extract data from WrappedDict
    gammas = np.array([WrappedDict[k]["gamma"] for k in sorted(WrappedDict.keys())])
    Areas = np.array([WrappedDict[k]["Area"] for k in sorted(WrappedDict.keys())])
    strains = np.array(sorted(WrappedDict.keys()))
    
    Amulgamma = Areas * gammas
    
    # Determine polynomial degree based on strain range
    if max(strains) > 0.009:
        deg = 2
        coeffs, covas = np.polyfit(strains, Amulgamma, deg=deg, cov=True)
        tau, gamma = coeffs[1:] / SurfaceArea
        tau_err, gamma_err = np.sqrt(np.diag(covas)[1:]) / SurfaceArea
    else:
        deg = 1
        coeffs, covas = np.polyfit(strains, Amulgamma, deg=deg, cov=True)
        tau, gamma = coeffs / SurfaceArea
        tau_err, gamma_err = np.sqrt(np.diag(covas)) / SurfaceArea
    
    print("gamma (eV/A^2): ", gamma, " +/- ", gamma_err)
    print("Tau_xx (eV/A^2): ", tau, " +/- ", tau_err)
    
    # Optional visualization
    if inspect and ProjectName is not None:
        _create_fit_plots(strains, Amulgamma, coeffs, deg, ProjectName)
    
    return {
        'tau': tau,
        'gamma': gamma,
        'tau_err': tau_err,
        'gamma_err': gamma_err,
        'strains': strains,
        'Amulgamma': Amulgamma,
        'coeffs': coeffs,
        'deg': deg
    }


def _create_fit_plots(strains, Amulgamma, coeffs, deg, ProjectName):
    """Helper function to create fit visualization plots."""
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    Fit_plot_path = os.path.join(ProjectName, 'Fit.png')
    Fit_errors_plot_path = os.path.join(ProjectName, 'Fit_errors.png')
    
    x = np.linspace(min(strains), max(strains), 100)
    
    if deg == 1:
        fit = coeffs[0] * x + coeffs[1]
        fit_at_strains = coeffs[0] * strains + coeffs[1]
    elif deg == 2:
        fit = coeffs[0] * (x**2) + coeffs[1] * x + coeffs[2]
        fit_at_strains = coeffs[0] * (strains**2) + coeffs[1] * strains + coeffs[2]
    
    # Create fit plot
    if not os.path.exists(Fit_plot_path):
        plt.figure()
        plt.axvline(x=0.0)
        plt.scatter(strains, Amulgamma, color='red', marker='s')
        plt.plot(x, fit, color='black')
        plt.xlabel('Strain')
        plt.ylabel('Surface Energy $(eV/\AA^2)$')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.grid()
        plt.savefig(Fit_plot_path, dpi=100)
        plt.close()
    
    # Create error plot
    if not os.path.exists(Fit_errors_plot_path):
        plt.figure()
        plt.axvline(x=0.0)
        errors = 100 * (Amulgamma - fit_at_strains) / fit_at_strains
        plt.scatter(strains, errors, color='black')
        plt.xlabel('Strain')
        plt.ylabel('Errors of the fit (%)')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.savefig(Fit_errors_plot_path, dpi=100)
        plt.close()

def compute_surface_stress_from_pressures(
    Pressure_tensor: np.ndarray,
    RelaxedCell: np.ndarray,
    CellwSurfaces: np.ndarray,
    SurfaceArea: float
) -> float:
    """
    Compute surface stress from pressure tensor components.
    
    Args:
        Pressure_tensor: Pressure tensor at zero strain (3,) or (3,3)
        RelaxedCell: Cell matrix of fully relaxed structure
        CellwSurfaces: Cell matrix of structure with surfaces
        SurfaceArea: Surface area at zero strain
        
    Returns:
        tau_from_pressures: Surface stress computed from pressure tensor
    """
    import numpy as np
    from scipy import constants
    
    convfactor2 = (constants.electron_volt / (10**(-30)) / (10**9))  # eV/A^3 to GPa
    
    # Calculate surface areas for normalization
    Ayz_surface = np.linalg.norm(np.cross(CellwSurfaces[1], CellwSurfaces[2]))
    Ayz_relaxed = np.linalg.norm(np.cross(RelaxedCell[1], RelaxedCell[2]))
    
    # Extract pressure components based on tensor shape
    if Pressure_tensor.shape == (3,):
        Pxx, Pyy, Pzz = Pressure_tensor / convfactor2
        PT = (Pxx + Pyy) / 2
    elif Pressure_tensor.shape == (3, 3):
        Pxx, Pyy, Pzz = np.diag(Pressure_tensor) / convfactor2
        PT = ((Pxx + Pyy) / 2) * (Ayz_surface / Ayz_relaxed)
    
    # Calculate volume and surface stress
    Volume = RelaxedCell[2, 2] * SurfaceArea
    tau_from_pressures = (Volume / (2 * SurfaceArea)) * (Pzz - PT)
    
    print("Tau from pressures (eV/A^2): ", tau_from_pressures)
    
    return tau_from_pressures


def archive_surface_stress_results(
    entry_data: dict,
    I: DataInputs.dataclass,
    Overwrite: bool = False
) -> dict:
    """
    Archive surface stress calculation results to a DataFrame stored as pickle.
    
    Args:
        entry_data: Dictionary containing all the computed results and metadata
        I: DataInputs dataclass containing input parameters
        Overwrite: If True, overwrite existing entries with matching parameters
        
    Returns:
        entry_data: The entry that was added/updated
    """
    import numpy as np
    import pandas as pd
    import os
    
    COLUMNS = [
        'Element', 'MillerIndices', 'acell', 'Lx', 'Ly', 'Lz',
        'HigherOrderModuli', 'Method', 'Engine',
        'Potential_ID', 'Encut', 'Kmesh', 'ProjectName', 'UserComment', 
        'SuperCellDimensions', 'Volume', 'minimizer_surfaces', 'minimizer_box',
        'etol', 'ftol', 'fmax_GeomOpt', 'fmax_OpenSurface', 'vacuum', 
        'Lx_surfaces', 'Ly_surfaces', 'Lz_surfaces', 'LoopOutputs', 'WorkflowComment'
    ]
    
    # Load or create DataFrame
    if os.path.isfile(I.StorageDataFrame):
        df = pd.read_pickle(I.StorageDataFrame)
        
        # Fix duplicate columns if they exist
        if df.columns.duplicated().any():
            print("Warning: Duplicate columns detected. Removing duplicates...")
            # Keep only the first occurrence of each column
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Ensure backward compatibility - add missing columns
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
                print(f"Added missing column: {col}")
    else:
        df = pd.DataFrame(columns=COLUMNS)
    
    # Check for duplicate entries
    mask = _check_for_duplicates(df, I)
    
    # Handle adding/updating entries
    if not mask.any():
        # No match found, add new entry
        df.loc[len(df)] = entry_data
        print("New entry added to the archive.")
    elif Overwrite:
        # Match found and Overwrite is True
        print("Entry already recorded. Overwriting...")
        matching_indices = df.index[mask].tolist()
        print(f"Removing existing entries at indices: {matching_indices}")
        df = df.drop(matching_indices)
        df.reset_index(drop=True, inplace=True)
        df.loc[len(df)] = entry_data
        print("Entry overwritten successfully.")
    else:
        # Match found but Overwrite is False
        print("---> Entry already recorded with the same parameters:")
        print("     Element, MillerIndices, SuperCellDimensions, InteratomicPotential,")
        print("     Encut, Kmesh, vacuum, and UserComment match existing entry.")
        print("     Set Overwrite=True to delete previous entry and archive the present values.")
    
    # Save the dataframe
    df.to_pickle(I.StorageDataFrame)
    df.to_pickle(os.path.join(I.ProjectName, 'Local_storage.pckl'))
    
    return entry_data


# def _check_for_duplicates(df, I):
#     """
#     Check if an entry with matching parameters already exists in the DataFrame.
    
#     Returns:
#         mask: Boolean Series indicating matching entries
#     """
#     import numpy as np
#     import pandas as pd
    
#     if df.empty:
#         return pd.Series([False] * len(df), index=df.index)
    
#     print(f"\n=== Checking for duplicates ===")
#     print(f"DataFrame has {len(df)} existing entries")
    
#     # Build mask step by step to avoid alignment issues
#     # Start with all True
#     mask = pd.Series([True] * len(df), index=df.index)
    
#     # Apply each condition
#     mask = mask & (df['Element'] == I.Element)
    
#     mask = mask & df['MillerIndices'].apply(
#         lambda x: np.array_equal(np.array(x), np.array(I.MillerIndices)) if x is not None else False
#     )
    
#     mask = mask & df['SuperCellDimensions'].apply(
#         lambda x: np.array_equal(np.array(x), np.array(I.SuperCellDimensions)) if x is not None else False
#     )
    
#     # Handle None comparison for Potential_ID
#     if I.InteratomicPotential is None:
#         potential_condition = df['Potential_ID'].isna() | (df['Potential_ID'] == I.InteratomicPotential)
#     else:
#         potential_condition = (df['Potential_ID'] == I.InteratomicPotential)
#     mask = mask & potential_condition
    
#     # Handle None comparison for Encut
#     if I.Encut is None:
#         encut_condition = df['Encut'].isna() | (df['Encut'] == I.Encut)
#     else:
#         encut_condition = (df['Encut'] == I.Encut)
#     mask = mask & encut_condition
    
#     # Handle None/array comparison for Kmesh
#     if I.Kmesh is None:
#         kmesh_condition = df['Kmesh'].isna() | (df['Kmesh'] == I.Kmesh)
#     else:
#         kmesh_condition = df['Kmesh'].apply(
#             lambda x: np.array_equal(np.array(x), np.array(I.Kmesh)) if x is not None else False
#         )
#     mask = mask & kmesh_condition
    
#     mask = mask & (df['vacuum'] == I.vacuum)
#     mask = mask & (df['UserComment'] == I.UserComment)
    
#     print(f"Total matches found: {mask.sum()}")
#     print("=" * 40)
    
#     return mask
def _check_for_duplicates(df, I):
    """
    Check if an entry with matching INPUT parameters already exists.
    Works for DataInputs, ElasticModuliInputs, and GammaSurfaceInputs.
    """
    import numpy as np
    import pandas as pd
    
    if df.empty:
        return pd.Series([False] * len(df), index=df.index)
    
    print(f"\n=== Checking for duplicates ===")
    print(f"DataFrame has {len(df)} existing entries")
    
    # Build mask step by step
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Always check Element
    mask = mask & (df['Element'] == I.Element)
    
    # Always check SuperCellDimensions
    mask = mask & df['SuperCellDimensions'].apply(
        lambda x: np.array_equal(np.array(x), np.array(I.SuperCellDimensions)) if x is not None else False
    )
    
    # Check MillerIndices (always present for surface stress and elastic moduli)
    mask = mask & df['MillerIndices'].apply(
        lambda x: np.array_equal(np.array(x), np.array(I.MillerIndices)) if x is not None else False
    )
    
    # Check DeformationAxis (only for elastic moduli)
    if hasattr(I, 'DeformationAxis') and 'DeformationAxis' in df.columns:
        mask = mask & (df['DeformationAxis'] == I.DeformationAxis)
    
    # Check DeformationType (only for elastic moduli)
    if hasattr(I, 'DeformationType') and 'DeformationType' in df.columns:
        mask = mask & (df['DeformationType'] == I.DeformationType)
    
    # Check StrainRange (only for elastic moduli)
    if hasattr(I, 'StrainMinimum') and hasattr(I, 'StrainMaximum') and 'StrainRange' in df.columns:
        expected_range = (I.StrainMinimum, I.StrainMaximum)
        mask = mask & df['StrainRange'].apply(
            lambda x: x == expected_range if x is not None else False
        )
    
    # Check vacuum (only for surface stress)
    if hasattr(I, 'vacuum') and 'vacuum' in df.columns:
        mask = mask & (df['vacuum'] == I.vacuum)
    
    # Handle None comparison for Potential_ID
    if I.InteratomicPotential is None:
        potential_condition = df['Potential_ID'].isna() | (df['Potential_ID'] == I.InteratomicPotential)
    else:
        potential_condition = (df['Potential_ID'] == I.InteratomicPotential)
    mask = mask & potential_condition
    
    # Handle None comparison for Encut
    if I.Encut is None:
        encut_condition = df['Encut'].isna() | (df['Encut'] == I.Encut)
    else:
        encut_condition = (df['Encut'] == I.Encut)
    mask = mask & encut_condition
    
    # Handle None/array comparison for Kmesh
    if I.Kmesh is None:
        kmesh_condition = df['Kmesh'].isna() | (df['Kmesh'] == I.Kmesh)
    else:
        kmesh_condition = df['Kmesh'].apply(
            lambda x: np.array_equal(np.array(x), np.array(I.Kmesh)) if x is not None else False
        )
    mask = mask & kmesh_condition
    
    # Always check UserComment
    mask = mask & (df['UserComment'] == I.UserComment)
    
    print(f"Total matches found: {mask.sum()}")
    print("=" * 40)
    
    return mask



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants


def convert_cauchy_to_pk2(final_cells, pressure_tensors, reference_cell):
    """
    Convert Cauchy stress to Second Piola-Kirchhoff stress and compute Green-Lagrange strain.
    
    Parameters
    ----------
    final_cells : array
        Array of deformed cell vectors, shape (n_points, 3, 3)
    pressure_tensors : array
        Array of pressure tensors (Cauchy stress), shape (n_points, 3, 3)
    reference_cell : array
        Reference (relaxed) cell vectors, shape (3, 3)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'pk2_stress': Second Piola-Kirchhoff stress tensors
        - 'green_lagrange_strain': Green-Lagrange strain tensors
        - 'deformation_gradients': Deformation gradient tensors
        - 'jacobians': Jacobian determinants
    """
    final_cells = np.asarray(final_cells)
    pressure_tensors = np.asarray(pressure_tensors)
    reference_cell = np.asarray(reference_cell)
    
    # Compute deformation gradients: F = deformed_cell · reference_cell⁻¹
    ref_inv = np.linalg.inv(reference_cell)
    deformation_gradients = np.array([cell @ ref_inv for cell in final_cells])
    
    # Compute Jacobians: J = det(F)
    jacobians = np.array([np.linalg.det(F) for F in deformation_gradients])
    
    # Convert Cauchy stress (pressure) to PK2 stress
    # Note: Cauchy stress σ = -P (pressure is negative stress)
    # Note: negative sign for convention
    cauchy_stress = -pressure_tensors
    
    # S = J · F⁻¹ · σ · F⁻ᵀ
    pk2_stress = np.array([
        jacobians[i] * np.linalg.inv(deformation_gradients[i]) @ cauchy_stress[i] @ np.linalg.inv(deformation_gradients[i]).T
        for i in range(len(deformation_gradients))
    ])
    
    # Compute Green-Lagrange strain: E = ½(Fᵀ·F - I)
    green_lagrange_strain = np.array([
        0.5 * (F.T @ F - np.eye(3))
        for F in deformation_gradients
    ])
    
    return {
        'pk2_stress': pk2_stress,
        'green_lagrange_strain': green_lagrange_strain,
        'deformation_gradients': deformation_gradients,
        'jacobians': jacobians
    }


def energy_model(x, *coeffs):
    """
    Energy density: ΔU/V₀ = C₂*x² + C₃*x³ + C₄*x⁴ + ...
    coeffs = (C2, C3, C4, ...)
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    for i, coeff in enumerate(coeffs):
        power = i + 2  # Start from x²
        result += coeff * x**power
    return result


def stress_model(x, *coeffs):
    """
    Stress: σ = C₂*x + C₃*x² + C₄*x³ + ...
    coeffs = (C2, C3, C4, ...)
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    for i, coeff in enumerate(coeffs):
        power = i + 1  # Start from x¹
        result += coeff * x**power
    return result

def format_value_with_error(value, error, min_sig_figs=1):
    """
    Format a value with its error, showing appropriate significant figures.
    
    Parameters:
    -----------
    value : float
        The measured value
    error : float
        The uncertainty/error
    min_sig_figs : int
        Minimum significant figures to show in the error (default: 1)
    
    Returns:
    --------
    str : Formatted string like "278.5±0.1" or "1.23±0.05"
    """
    import math
    import numpy as np
    
    # Handle special cases for value
    if np.isnan(value):
        return "NaN"
    if np.isinf(value):
        return "inf"
    
    # Handle special cases for error
    if np.isnan(error):
        return f"{value:.2f}±NaN"
    if np.isinf(error):
        return f"{value:.2f}±inf"
    if error == 0:
        return f"{value:.2f}±0.00"
    
    # Determine the order of magnitude of the error
    error_magnitude = math.floor(math.log10(abs(error)))
    
    # Round error to min_sig_figs significant figures
    error_rounded = round(error, -error_magnitude + min_sig_figs - 1)
    
    # Determine decimal places needed
    if error_magnitude < 0:
        decimal_places = abs(error_magnitude) + min_sig_figs - 1
    else:
        decimal_places = max(0, min_sig_figs - 1)
    
    # Round value to same decimal places
    value_rounded = round(value, decimal_places)
    
    # Format both with same precision
    format_str = f"{{:.{decimal_places}f}}"
    value_str = format_str.format(value_rounded)
    error_str = format_str.format(error_rounded)
    
    return f"{value_str}±{error_str}"


def extract_elastic_moduli(energies, pressure_tensors, final_cells, reference_cell, V0, 
                           deformation_axis, degree=3, slice_min=None, slice_max=None):
    """
    Extract elastic moduli from energy and pressure data using Green-Lagrange strain and PK2 stress.
    
    Parameters
    ----------
    energies : array
        Total energies (eV)
    pressure_tensors : array
        Pressure tensors (Cauchy stress), shape (n_points, 3, 3)
    final_cells : array
        Deformed cell vectors, shape (n_points, 3, 3)
    reference_cell : array
        Reference (relaxed) cell vectors, shape (3, 3)
    V0 : float
        Reference volume at zero strain (Ų)
    deformation_axis : int
        Deformation axis (0=x, 1=y, 2=z)
    degree : int
        Maximum polynomial degree (default: 3)
    slice_min : int, optional
        Minimum index for slicing arrays (default: None, use all data)
    slice_max : int, optional
        Maximum index for slicing arrays (default: None, use all data)
    
    Returns
    -------
    dict
        Results containing elastic constants from both methods
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy import constants
    
    # Convert to numpy arrays
    energies = np.asarray(energies)
    pressure_tensors = np.asarray(pressure_tensors)
    final_cells = np.asarray(final_cells)
    
    # Apply slicing if requested
    if slice_min is not None or slice_max is not None:
        slice_obj = slice(slice_min, slice_max)
        energies = energies[slice_obj]
        pressure_tensors = pressure_tensors[slice_obj]
        final_cells = final_cells[slice_obj]
        print(f"Using data slice [{slice_min}:{slice_max}], {len(energies)} points")
    
    # Convert Cauchy stress to PK2 stress and compute Green-Lagrange strain
    conversion_results = convert_cauchy_to_pk2(final_cells, pressure_tensors, reference_cell)
    
    # Extract Green-Lagrange strain and PK2 stress components
    green_lagrange_strain = conversion_results['green_lagrange_strain'][:, deformation_axis, deformation_axis]
    pk2_stress = conversion_results['pk2_stress'][:, deformation_axis, deformation_axis]  
    
    n_params = degree - 1  # degree 3 -> fit C2, C3, C4 (3 params)
    
    convfactor = constants.electron_volt / (1e-30) / (1e9)  # eV/Ų to GPa
    
    # Calculate strain energy density
    E0 = energies[np.argmin(np.abs(green_lagrange_strain))]
    deltaUbyV = (energies - E0) / V0
    
    # Fit energy density
    popt_energy, pcov_energy = curve_fit(
        lambda x, *c: energy_model(x, *c),
        green_lagrange_strain,
        deltaUbyV,
        p0=np.ones(n_params)
    )
    
    # Fit stress
    popt_stress, pcov_stress = curve_fit(
        lambda x, *c: stress_model(x, *c),
        green_lagrange_strain,
        pk2_stress,
        p0=np.ones(n_params)
    )
    
    # Convert energy coefficients to elastic constants
    energy_constants = {}
    energy_errors = {}
    perr_energy = np.sqrt(np.diag(pcov_energy))
    
    for i in range(n_params):
        n = i + 2  # C2, C3, C4, ...
        factor = n  # multiply by 2 for C2, 3 for C3, 4 for C4, etc.
        energy_constants[f'C{n}'] = popt_energy[i] * convfactor * factor
        energy_errors[f'C{n}_err'] = perr_energy[i] * convfactor * factor
    
    # Stress coefficients are already the elastic constants
    stress_constants = {}
    stress_errors = {}
    perr_stress = np.sqrt(np.diag(pcov_stress))
    
    for i in range(n_params):
        n = i + 2  # C2, C3, C4, ...
        stress_constants[f'C{n}'] = popt_stress[i]
        stress_errors[f'C{n}_err'] = perr_stress[i]
    
    # Calculate R²
    fitted_energy = energy_model(green_lagrange_strain, *popt_energy)
    ss_res_energy = np.sum((deltaUbyV - fitted_energy)**2)
    ss_tot_energy = np.sum((deltaUbyV - np.mean(deltaUbyV))**2)
    r_squared_energy = 1 - (ss_res_energy / ss_tot_energy) if ss_tot_energy != 0 else 0
    
    fitted_stress = stress_model(green_lagrange_strain, *popt_stress)
    ss_res_stress = np.sum((pk2_stress - fitted_stress)**2)
    ss_tot_stress = np.sum((pk2_stress - np.mean(pk2_stress))**2)
    r_squared_stress = 1 - (ss_res_stress / ss_tot_stress) if ss_tot_stress != 0 else 0
    
    # Create fine grid for plotting
    strain_fine = np.linspace(green_lagrange_strain.min(), green_lagrange_strain.max(), 200)
    fit_energy_fine = energy_model(strain_fine, *popt_energy)
    fit_stress_fine = stress_model(strain_fine, *popt_stress)
    
    # Error bands
    fit_energy_upper = energy_model(strain_fine, *(popt_energy + perr_energy))
    fit_energy_lower = energy_model(strain_fine, *(popt_energy - perr_energy))
    fit_stress_upper = stress_model(strain_fine, *(popt_stress + perr_stress))
    fit_stress_lower = stress_model(strain_fine, *(popt_stress - perr_stress))
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Energy density plot
    ax1.scatter(green_lagrange_strain, convfactor * deltaUbyV, label='Computed Values', s=50, alpha=0.7)
    ax1.plot(strain_fine, convfactor * fit_energy_fine, 'r-', label='Best fit', linewidth=2)
    ax1.fill_between(strain_fine, convfactor * fit_energy_lower, convfactor * fit_energy_upper, 
                     alpha=0.3, color='red', label='Uncertainty (±1σ)')
    ax1.set_xlabel('Green-Lagrange Strain', fontsize=12)
    ax1.set_ylabel(r'$\frac{U-U_0}{V_0}$ (GPa)', fontsize=12)
    ax1.set_title(f'Energy Method (R² = {r_squared_energy:.6f})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Stress plot
    ax2.scatter(green_lagrange_strain, pk2_stress, label='Computed Values', s=50, alpha=0.7)
    ax2.plot(strain_fine, fit_stress_fine, 'r-', label='Best fit', linewidth=2)
    ax2.fill_between(strain_fine, fit_stress_lower, fit_stress_upper, 
                     alpha=0.3, color='red', label='Uncertainty (±1σ)')
    ax2.set_xlabel('Green-Lagrange Strain', fontsize=12)
    ax2.set_ylabel('PK2 Stress (GPa)', fontsize=12)
    ax2.set_title(f'Stress Method (R² = {r_squared_stress:.6f})', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Print comparison table with smart formatting
    n_cols = n_params
    col_width = 20  # Reduced since formatted strings are more compact
    total_width = 20 + col_width * n_cols
    
    print("=" * total_width)
    
    # Header
    header = f"{'Method':<20}"
    for i in range(n_params):
        header += f"C{i+2} (GPa)".ljust(col_width)
    print(header)
    
    print("-" * total_width)
    
    # Energy row with smart formatting
    energy_row = f"{'Energy':<20}"
    for i in range(n_params):
        cn = f'C{i+2}'
        formatted = format_value_with_error(energy_constants[cn], energy_errors[f'{cn}_err'], min_sig_figs=2)
        energy_row += formatted.ljust(col_width)
    print(energy_row)
    
    # Stress row with smart formatting
    stress_row = f"{'Stress':<20}"
    for i in range(n_params):
        cn = f'C{i+2}'
        formatted = format_value_with_error(stress_constants[cn], stress_errors[f'{cn}_err'], min_sig_figs=2)
        stress_row += formatted.ljust(col_width)
    print(stress_row)
    
    print("=" * total_width)
    
    return {
        'energy': {
            'constants': energy_constants,
            'errors': energy_errors,
            'r_squared': r_squared_energy
        },
        'stress': {
            'constants': stress_constants,
            'errors': stress_errors,
            'r_squared': r_squared_stress
        }
    }


# def extract_elastic_moduli(energies, pressure_tensors, final_cells, reference_cell, V0, 
#                            deformation_axis, degree=3, slice_min=None, slice_max=None):
#     """
#     Extract elastic moduli from energy and pressure data using Green-Lagrange strain and PK2 stress.
    
#     Parameters
#     ----------
#     energies : array
#         Total energies (eV)
#     pressure_tensors : array
#         Pressure tensors (Cauchy stress), shape (n_points, 3, 3)
#     final_cells : array
#         Deformed cell vectors, shape (n_points, 3, 3)
#     reference_cell : array
#         Reference (relaxed) cell vectors, shape (3, 3)
#     V0 : float
#         Reference volume at zero strain (Ų)
#     deformation_axis : int
#         Deformation axis (0=x, 1=y, 2=z)
#     degree : int
#         Maximum polynomial degree (default: 3)
#     slice_min : int, optional
#         Minimum index for slicing arrays (default: None, use all data)
#     slice_max : int, optional
#         Maximum index for slicing arrays (default: None, use all data)
    
#     Returns
#     -------
#     dict
#         Results containing elastic constants from both methods
#     """
#     # Convert to numpy arrays
#     energies = np.asarray(energies)
#     pressure_tensors = np.asarray(pressure_tensors)
#     final_cells = np.asarray(final_cells)
    
#     # Apply slicing if requested
#     if slice_min is not None or slice_max is not None:
#         slice_obj = slice(slice_min, slice_max)
#         energies = energies[slice_obj]
#         pressure_tensors = pressure_tensors[slice_obj]
#         final_cells = final_cells[slice_obj]
#         print(f"Using data slice [{slice_min}:{slice_max}], {len(energies)} points")
    
#     # Convert Cauchy stress to PK2 stress and compute Green-Lagrange strain
#     conversion_results = convert_cauchy_to_pk2(final_cells, pressure_tensors, reference_cell)
    
#     # Extract Green-Lagrange strain and PK2 stress components
#     green_lagrange_strain = conversion_results['green_lagrange_strain'][:, deformation_axis, deformation_axis]
#     pk2_stress = conversion_results['pk2_stress'][:, deformation_axis, deformation_axis]  
    
#     n_params = degree - 1  # degree 3 -> fit C2, C3, C4 (3 params)
    
#     convfactor = constants.electron_volt / (1e-30) / (1e9)  # eV/Ų to GPa
    
#     # Calculate strain energy density
#     E0 = energies[np.argmin(np.abs(green_lagrange_strain))]
#     deltaUbyV = (energies - E0) / V0
    
#     # Fit energy density
#     popt_energy, pcov_energy = curve_fit(
#         lambda x, *c: energy_model(x, *c),
#         green_lagrange_strain,
#         deltaUbyV,
#         p0=np.ones(n_params)
#     )
    
#     # Fit stress
#     popt_stress, pcov_stress = curve_fit(
#         lambda x, *c: stress_model(x, *c),
#         green_lagrange_strain,
#         pk2_stress,
#         p0=np.ones(n_params)
#     )
    
#     # Convert energy coefficients to elastic constants
#     energy_constants = {}
#     energy_errors = {}
#     perr_energy = np.sqrt(np.diag(pcov_energy))
    
#     for i in range(n_params):
#         n = i + 2  # C2, C3, C4, ...
#         factor = n  # multiply by 2 for C2, 3 for C3, 4 for C4, etc.
#         energy_constants[f'C{n}'] = popt_energy[i] * convfactor * factor
#         energy_errors[f'C{n}_err'] = perr_energy[i] * convfactor * factor
    
#     # Stress coefficients are already the elastic constants
#     stress_constants = {}
#     stress_errors = {}
#     perr_stress = np.sqrt(np.diag(pcov_stress))
    
#     for i in range(n_params):
#         n = i + 2  # C2, C3, C4, ...
#         stress_constants[f'C{n}'] = popt_stress[i]
#         stress_errors[f'C{n}_err'] = perr_stress[i]
    
#     # Calculate R²
#     fitted_energy = energy_model(green_lagrange_strain, *popt_energy)
#     ss_res_energy = np.sum((deltaUbyV - fitted_energy)**2)
#     ss_tot_energy = np.sum((deltaUbyV - np.mean(deltaUbyV))**2)
#     r_squared_energy = 1 - (ss_res_energy / ss_tot_energy) if ss_tot_energy != 0 else 0
    
#     fitted_stress = stress_model(green_lagrange_strain, *popt_stress)
#     ss_res_stress = np.sum((pk2_stress - fitted_stress)**2)
#     ss_tot_stress = np.sum((pk2_stress - np.mean(pk2_stress))**2)
#     r_squared_stress = 1 - (ss_res_stress / ss_tot_stress) if ss_tot_stress != 0 else 0
    
#     # Create fine grid for plotting
#     strain_fine = np.linspace(green_lagrange_strain.min(), green_lagrange_strain.max(), 200)
#     fit_energy_fine = energy_model(strain_fine, *popt_energy)
#     fit_stress_fine = stress_model(strain_fine, *popt_stress)
    
#     # Error bands
#     fit_energy_upper = energy_model(strain_fine, *(popt_energy + perr_energy))
#     fit_energy_lower = energy_model(strain_fine, *(popt_energy - perr_energy))
#     fit_stress_upper = stress_model(strain_fine, *(popt_stress + perr_stress))
#     fit_stress_lower = stress_model(strain_fine, *(popt_stress - perr_stress))
    
#     # Create plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     # Energy density plot
#     ax1.scatter(green_lagrange_strain, convfactor * deltaUbyV, label='Computed Values', s=50, alpha=0.7)
#     ax1.plot(strain_fine, convfactor * fit_energy_fine, 'r-', label='Best fit', linewidth=2)
#     ax1.fill_between(strain_fine, convfactor * fit_energy_lower, convfactor * fit_energy_upper, 
#                      alpha=0.3, color='red', label='Uncertainty (±1σ)')
#     ax1.set_xlabel('Green-Lagrange Strain', fontsize=12)
#     ax1.set_ylabel(r'$\frac{U-U_0}{V_0}$ (GPa)', fontsize=12)
#     ax1.set_title(f'Energy Method (R² = {r_squared_energy:.6f})', fontsize=13)
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # Stress plot
#     ax2.scatter(green_lagrange_strain, pk2_stress, label='Computed Values', s=50, alpha=0.7)
#     ax2.plot(strain_fine, fit_stress_fine, 'r-', label='Best fit', linewidth=2)
#     ax2.fill_between(strain_fine, fit_stress_lower, fit_stress_upper, 
#                      alpha=0.3, color='red', label='Uncertainty (±1σ)')
#     ax2.set_xlabel('Green-Lagrange Strain', fontsize=12)
#     ax2.set_ylabel('PK2 Stress (GPa)', fontsize=12)
#     ax2.set_title(f'Stress Method (R² = {r_squared_stress:.6f})', fontsize=13)
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()
    
#     plt.tight_layout()
    
#     # Print comparison table
#     n_cols = n_params
#     col_width = 25
#     total_width = 20 + col_width * n_cols
    
#     print("=" * total_width)
    
#     # Header
#     header = f"{'Method':<20}"
#     for i in range(n_params):
#         header += f"C{i+2} (GPa)".ljust(col_width)
#     print(header)
    
#     print("-" * total_width)
    
#     # Energy row
#     energy_row = f"{'Energy':<20}"
#     for i in range(n_params):
#         cn = f'C{i+2}'
#         val_str = f"{energy_constants[cn]:>8.4f} ± {energy_errors[f'{cn}_err']:<8.4f}"
#         energy_row += val_str.ljust(col_width)
#     print(energy_row)
    
#     # Stress row
#     stress_row = f"{'Stress':<20}"
#     for i in range(n_params):
#         cn = f'C{i+2}'
#         val_str = f"{stress_constants[cn]:>8.4f} ± {stress_errors[f'{cn}_err']:<8.4f}"
#         stress_row += val_str.ljust(col_width)
#     print(stress_row)
    
#     print("=" * total_width)
    
#     return {
#         'energy': {
#             'constants': energy_constants,
#             'errors': energy_errors,
#             'r_squared': r_squared_energy
#         },
#         'stress': {
#             'constants': stress_constants,
#             'errors': stress_errors,
#             'r_squared': r_squared_stress
#         }
#     }

# return {
#         'energy': {
#             'constants': energy_constants,
#             'errors': energy_errors,
#             'r_squared': r_squared_energy
#         },
#         'stress': {
#             'constants': stress_constants,
#             'errors': stress_errors,
#             'r_squared': r_squared_stress
#         },
#         'conversion_results': conversion_results,
#         'figure': fig
#     }