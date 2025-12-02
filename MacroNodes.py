import pyiron_workflow as pwf
from DataClassNode import DataInputs, WorkflowState
from typing import Union, Optional, Any

@pwf.as_macro_node
def SurfaceEnergyForStrainValue(
    self,
    IdataClass: Union[DataInputs.dataclass,WorkflowState.dataclass],
    strain_value,
):  
    from CalculationNodes import ForceMinimization
    from FormattingNodes import Strain_to_name, OutputDecider
    from ArithmeticNodes import compute_gamma
    from StructureManipulationNodes import apply_strain, add_vacuum
    
    self.comm = Strain_to_name(strain=strain_value)

    self.StrainingStructure = apply_strain(I=IdataClass, 
                                           strain=strain_value,
                                           deformation_axis=0
                                          )

    self.PeriodicCellMinimization = ForceMinimization(I = self.StrainingStructure,
                                                      Structure_variable_name='StrainedStructure',
                                                      poisson_relaxation=False,
                                                      return_as_dataclass=True,
                                                      comment=self.comm)

    self.OpeningSurfaces = add_vacuum(self.PeriodicCellMinimization, 
                                      Structure_variable_name="Structure_Minimized",
                                      override_vacuum=True)

    self.MinimizationWithSurfaces = ForceMinimization(I = self.OpeningSurfaces,
                                                      Structure_variable_name="Structure_w_vacuum",
                                                      poisson_relaxation=False,
                                                      return_as_dataclass=True,
                                                      comment=self.comm)

    self.gamma_area = compute_gamma(I_Periodic = self.PeriodicCellMinimization,
                                    I_Surfaces = self.MinimizationWithSurfaces,
                                    verbose = IdataClass.verbose)
    
    self.gamma = self.gamma_area.outputs.gamma
    self.area = self.gamma_area.outputs.area

    self.WrappedResults = OutputDecider(
        strain=strain_value,
        gamma=self.gamma,
        area=self.area,
        PressureTensor=self.MinimizationWithSurfaces.outputs.Output.Pressures, #_Minimization,
        FinalCell=self.MinimizationWithSurfaces.outputs.Output.Cell, #_Minimization,
        ForceMax_w_surfaces=self.MinimizationWithSurfaces.outputs.Output.ForceMax #_Minimization
    )

    return self.WrappedResults

@pwf.as_macro_node
def MinimizationAtDifferentStrains(
    self,
    IdataClass: Union[DataInputs.dataclass,WorkflowState.dataclass],
    strain_value,
):  
    from CalculationNodes import ForceMinimization
    from FormattingNodes import Strain_to_name
    from ArithmeticNodes import compute_gamma
    from StructureManipulationNodes import apply_strain, add_vacuum
    
    self.comm = Strain_to_name(strain=strain_value)

    self.StrainingStructure = apply_strain(
        IdataClass, strain_value
    )

    self.PeriodicCellMinimization = ForceMinimization(self.StrainingStructure,
                                                      Structure_variable_name='StrainedStructure', 
                                                      comment=self.comm, poisson_relaxation=True,
                                                      return_as_dataclass=False)

    return self.PeriodicCellMinimization



@pwf.as_macro_node
def WrappedForLoop(self,IdataClass, strain_list): 
    
    self.df = pwf.Workflow.create.for_node(
        body_node_class=MinimizationAtDifferentStrains,
        iter_on="strain_value",
        IdataClass = IdataClass,
        strain_value= strain_list
    )
    
    return self.df


@pwf.as_macro_node
def FaultEnergiesAtDifferentDeformations(self, fraction: float, I: Union[DataInputs.dataclass, WorkflowState.dataclass]):
    from StructureManipulationNodes import DeformStructure
    from CalculationNodes import ForceMinimization
    from FormattingNodes import Strain_to_name
    from ArithmeticNodes import compute_gamma

    self.comm = Strain_to_name(strain=fraction)

    self.SlippingStructure = DeformStructure(I=I,
                                             Mode=I.DeformationMode,
                                             fraction=fraction
                                             )
    self.SetSelectiveDynamics = set_selective_dynamics(self.SlippingStructure,
                                                       Structure_variable_name='DeformedStructure')

    self.PeriodicCellMinimization = ForceMinimization(I=self.SetSelectiveDynamics.outputs.I_out,
                                                      Structure_variable_name=self.SetSelectiveDynamics.outputs.AlteredStructureName,
                                                      # 'DeformedStructure_seldyn',  # Structure variable name needs to be added with a _seldyn
                                                      comment=self.comm,
                                                      poisson_relaxation=False,
                                                      return_as_dataclass=True
                                                      )
    self.gamma_area = compute_gamma(I_Periodic=I,
                                    I_Surfaces=self.PeriodicCellMinimization,
                                    verbose=False)

    self.gamma = self.gamma_area.outputs.gamma
    self.DeformedStructure_final = self.PeriodicCellMinimization.outputs.Output.Structure_Minimized

    return self.gamma, self.DeformedStructure_final
