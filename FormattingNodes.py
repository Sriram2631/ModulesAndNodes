import pyiron_workflow as pwf
import pandas as pd
from DataClassNode import DataInputs, WorkflowState
from typing import Union, Optional, Any

@pwf.as_function_node
def Strain_to_name(strain:float)->str:
    a = '%E' % strain
    name = a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
    #print("Strain value: ",name)
    return name

@pwf.as_function_node
def OutputDecider(strain, gamma, area, PressureTensor, FinalCell, ForceMax_w_surfaces):
    import numpy as np
    if strain==0.0 or np.isclose(strain,0.0,atol=1e-6):
        results_dict = {'gamma':gamma,
                        'Area':area,
                       'PressureTensor':PressureTensor,
                       'FinalCell':FinalCell,
                        'ForceMax':ForceMax_w_surfaces
                       }
    else:
        results_dict = {'gamma':gamma,
                        'Area':area,
                        'ForceMax':ForceMax_w_surfaces
                       }
    return results_dict

@pwf.as_function_node
def OutputDeciderBulk(U, PressureTensor, FinalCell, ForceMax):
    
    results_dict = {'Energy':U,
                   'PressureTensor':PressureTensor,
                   'FinalCell':FinalCell,
                    'ForceMax':ForceMax
                   }
    return results_dict

@pwf.as_function_node
def OutputRemap(df:pd.DataFrame, column:str = 'WrappedResults')->dict:
    WrappedDict = df.set_index('strain_value')[column].to_dict()
    return WrappedDict
    
@pwf.as_function_node
def MakeFits(WrappedDict, 
             FullyRelaxOutputDataClass,
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


# @pwf.as_function_node
# def WriteElasticModuliArchive(
#     moduli_results: dict,
#     I: DataInputs.dataclass,
#     GeomOptOutput: Union[DataInputs.dataclass, WorkflowState.dataclass],
#     LoopOutputs: dict,
#     inspect: bool = True,
#     Overwrite: bool = False,
#     debug: bool = False
# ) -> dict:
#     """
#     Archive elastic moduli results.
#
#     Args:
#         moduli_results: Output from extract_elastic_moduli function
#         I: ElasticModuliInputs dataclass
#         GeomOptOutput: Geometry optimization output
#         LoopOutputs: Dictionary containing all strain point calculations
#         inspect: Whether plots were created
#         Overwrite: Whether to overwrite existing entries
#         debug: Debug mode flag
#     """
#     import numpy as np
#     from AnalyseAndArchiveFunctions import archive_elastic_moduli_results
#
#     # Prepare data from inputs
#     RelaxedCell = np.array(GeomOptOutput.RelaxedStructure.cell.tolist())
#
#     # Generate workflow comment comparing methods
#     energy_c2 = moduli_results['energy']['constants']['C2']
#     stress_c2 = moduli_results['stress']['constants']['C2']
#     diff_percent = 100 * np.abs(energy_c2 - stress_c2) / np.mean([energy_c2, stress_c2])
#
#     if diff_percent < 1.0:
#         WorkflowComment = f'Excellent_{diff_percent:.3f}%'
#     else:
#         WorkflowComment = f'Check_{diff_percent:.3f}%'
#
#
#     if hasattr(I, 'DeformationAxis'):
#         DeformationAxis = I.DeformationAxis
#     else:
#         DeformationAxis = 0
#
#     # Prepare entry data
#     entry = {
#         'Element': I.Element,
#         'MillerIndices':I.MillerIndices,
#         'acell': GeomOptOutput.acell_relaxed,
#         'Lx': RelaxedCell[0, 0],
#         'Ly': RelaxedCell[1, 1],
#         'Lz': RelaxedCell[2, 2],
#         'Volume': np.linalg.det(RelaxedCell),
#
#         # Store the full moduli_results dictionary
#         'Moduli': moduli_results,
#
#         'DeformationAxis': DeformationAxis,
#         #'DeformationType': I.DeformationType,
#         'StrainRange': (I.StrainMinimum, I.StrainMaximum),
#         #'PolynomialDegree': I.PolynomialDegree,
#
#         'Method': I.calctype,
#         'Potential_ID': I.InteratomicPotential,
#         'Engine': I.calcengine,
#         'Encut': I.Encut,
#         'Kmesh': I.Kmesh,
#         'ProjectName': I.ProjectName,
#         'UserComment': I.UserComment,
#         'SuperCellDimensions': I.SuperCellDimensions,
#         'minimizer': I.MinimizerForSurfaces,
#         'etol': I.etol,
#         'ftol': I.ftol,
#         'fmax_GeomOpt': GeomOptOutput.ForceMax,
#
#         'ReferenceCell': RelaxedCell,
#         'LoopOutputs': LoopOutputs,
#         'WorkflowComment': WorkflowComment
#     }
#
#     # Archive the results
#     return archive_elastic_moduli_results(
#         entry_data=entry,
#         I=I,
#         Overwrite=Overwrite
#     )



# @pwf.as_function_node
# def WriteArchive(
#     WrappedDict: dict,
#     I: DataInputs.dataclass,
#     GeomOptOutput:Union[DataInputs.dataclass,WorkflowState.dataclass],
#     inspect: bool = True,
#     Overwrite: bool = False,
#     debug: bool = False
# ) -> dict:
#     """
#     Main function that computes surface stress and archives results.
#     """
#     import numpy as np
#     from AnalyseAndArchiveFunctions import compute_surface_stress_from_fit, compute_surface_stress_from_pressures, archive_surface_stress_results, _check_for_duplicates, _create_fit_plots
#
#     # Prepare data from inputs
#     RelaxedCell = np.array(GeomOptOutput.RelaxedStructure.cell.tolist())
#     CellwSurfaces = WrappedDict.get(0.0, {}).get('FinalCell')
#     Pressure_tensor = WrappedDict.get(0.0, {}).get('PressureTensor')
#     SurfaceArea = WrappedDict.get(0.0, {}).get('Area')
#
#     # 1. Compute surface stress from fitting
#     fit_results = compute_surface_stress_from_fit(
#         WrappedDict=WrappedDict,
#         SurfaceArea=SurfaceArea,
#         inspect=inspect,
#         ProjectName=I.ProjectName
#     )
#
#     # 2. Compute surface stress from pressures
#     tau_from_pressures = compute_surface_stress_from_pressures(
#         Pressure_tensor=Pressure_tensor,
#         RelaxedCell=RelaxedCell,
#         CellwSurfaces=CellwSurfaces,
#         SurfaceArea=SurfaceArea
#     )
#
#     # 3. Prepare entry data
#     if np.isclose(fit_results['tau'], tau_from_pressures, atol=1e-3):
#         WorkflowComment = 'Yes'
#     else:
#         WorkflowComment = 'No_' + str(np.abs(fit_results['tau'] - tau_from_pressures))
#
#     entry = {
#         'Element': I.Element,
#         'MillerIndices': I.MillerIndices,
#         'acell': GeomOptOutput.acell_relaxed,
#         'Lx': RelaxedCell[0, 0],
#         'Ly': RelaxedCell[1, 1],
#         'Lz': RelaxedCell[2, 2],
#         'gamma': WrappedDict.get(0.0, {}).get('gamma'),
#         'tauxx': fit_results['tau'],
#         'gamma_intercept': fit_results['gamma'],
#         'tau_from_pressures': tau_from_pressures,
#         'FitError_tau': fit_results['tau_err'],
#         'FitError_gamma': fit_results['gamma_err'],
#         'Pressure_Tensor': Pressure_tensor,
#         'Method': I.calctype,
#         'Potential_ID': I.InteratomicPotential,
#         'Engine': I.calcengine,
#         'Encut': I.Encut,
#         'Kmesh': I.Kmesh,
#         'ProjectName': I.ProjectName,
#         'UserComment': I.UserComment,
#         'SuperCellDimensions': I.SuperCellDimensions,
#         'Volume': RelaxedCell[2, 2] * SurfaceArea,
#         'minimizer_surfaces': I.MinimizerForSurfaces,
#         'minimizer_box': I.MinimizerForCompleteRelaxation,
#         'etol': I.etol,
#         'ftol': I.ftol,
#         'fmax_GeomOpt': GeomOptOutput.ForceMax, #_RelaxedStructure,
#         'fmax_OpenSurface': WrappedDict.get(0.0, {}).get('ForceMax'),
#         'vacuum': I.vacuum,
#         'Lx_surfaces': CellwSurfaces[0, 0],
#         'Ly_surfaces': CellwSurfaces[1, 1],
#         'Lz_surfaces': CellwSurfaces[2, 2],
#         'LoopOutputs': WrappedDict,
#         'WorkflowComment': WorkflowComment
#     }
#
#     # 4. Archive the results
#     return archive_surface_stress_results(
#         entry_data=entry,
#         I=I,
#         Overwrite=Overwrite
#     )