import pyiron_workflow as pwf
from dataclasses import field
from typing import Union, Optional, Any
import pandas as pd
import numpy as np
import os
import ase

@pwf.as_dataclass_node
class DataInputs:
    # Required fields for all calc engines
    Element: str
    MillerIndices: tuple
    SuperCellDimensions: list[int]
    acell: float
    vacuum: int = 10
    calcengine: str = 'Lammps'  # Make this early so __post_init__ can use it
    
    # Conditionally relevant fields - these will be set in __post_init__
    InteratomicPotential: Optional[Union[pd.DataFrame, str]] = None
    Encut: Optional[int] = None
    Kmesh: Optional[list[int]] = None
    MinimizerForCompleteRelaxation: Optional[str] = None
    MinimizerForSurfaces: Optional[str] = None
    
    # Common fields with defaults
    
    etol: float = 0.0
    ftol: float = 1e-8
    StrainMinimum: float = -0.0001
    StrainMaximum: float = 0.0001
    StrainStateCount: int = 21
    StorageDataFrame: Union[str, os.PathLike] = 'Storage.pckl'
    UserComment: str = ''
    calctype: str = 'relax'
    verbose: bool = False
    delete_existing_jobs: bool = False
    delete_aborted_jobs: bool = True
    decompress: bool = False
    submit_job: bool = False
    queue: str = 'cmmg'
    
    
    # Computed fields (set in __post_init__, but can be overridden)
    ProjectName: Optional[str] = None  # Allow manual override
    Strainlist: list[float] = field(default_factory=list) #, init=False)
    
    def __post_init__(self):
        """Set engine-specific defaults and compute derived fields."""
        
        import numpy as np
        
        # Set defaults based on calc engine
        if self.calcengine.upper() == 'LAMMPS':
            self._set_lammps_defaults()
        elif self.calcengine.upper() == 'VASP':
            self._set_vasp_defaults()
        else:
            raise ValueError(f"Unknown calc engine: {self.calcengine}")
        
        # Compute derived fields
        self._compute_project_name()
        self._compute_strain_list()
    
    def _set_lammps_defaults(self):
        """Set LAMMPS-specific field values."""
        # LAMMPS doesn't use these DFT parameters
        self.Encut = None  # More explicit than 0
        self.Kmesh = None
        
        # Set defaults if not provided
        print(self.InteratomicPotential)
        if self.InteratomicPotential is None:
            raise ValueError("InteratomicPotential is required for LAMMPS")
        if self.MinimizerForCompleteRelaxation is None:
            self.MinimizerForCompleteRelaxation = 'cg'
        if self.MinimizerForSurfaces is None:
            self.MinimizerForSurfaces = 'fire'
        #self.vacuum = vacuum #100.0
            
    
    def _set_vasp_defaults(self):
        """Set VASP-specific field values."""
        # VASP doesn't use these classical MD parameters  
        self.InteratomicPotential = None
        self.MinimizerForCompleteRelaxation = None
        self.MinimizerForSurfaces = None
        self.submit_job=True
        # Set defaults if not provided
        if self.Encut is None:
            self.Encut = 750
        if self.Kmesh is None:
            self.Kmesh = [12, 12, 12]
            
        if self.vacuum>25.0:
            self.vacuum=10.0

        self.StrainMaximum=0.1
        self.StrainMinimum=-0.1
        self.StrainStateCount=11
        
    
    def _compute_project_name(self):
        """Compute project name based on engine and parameters."""
        # Only compute if not manually set
        if self.ProjectName is not None:
            return  # Keep the manually provided name
            
        if self.calcengine.upper() == 'LAMMPS':
            self.ProjectName = (
                f"{self.Element}_"
                f"{self.Pot_to_name()}_"
                f"{''.join(str(i) for i in self.MillerIndices)}_"
                f"{'x'.join(str(i) for i in self.SuperCellDimensions)}_"
                f"{self.MinimizerForSurfaces}_"
                f"{self.UserComment}"
            )
        elif self.calcengine.upper() == 'VASP':
            self.ProjectName = (
                f"{self.Element}_"
                f"{self.Encut}_"
                f"{'x'.join(str(i) for i in self.Kmesh)}_"
                f"{''.join(str(i) for i in self.MillerIndices)}_"
                f"{'x'.join(str(i) for i in self.SuperCellDimensions)}_"
                f"{self.UserComment}"
            )
    
    def _compute_strain_list(self):
        """Compute the strain list."""
        self.Strainlist = [
            round(strain, 6) 
            for strain in np.linspace(
                self.StrainMinimum, 
                self.StrainMaximum, 
                num=self.StrainStateCount, 
                endpoint=True
            ).tolist()
        ]
    
    def Pot_to_name(self) -> str:
        """Extract potential name for project naming."""
        if self.InteratomicPotential is None:
            return "NoPot"
            
        # Handle DataFrame case
        if isinstance(self.InteratomicPotential, pd.DataFrame):
            potential = self.InteratomicPotential['Name'].iloc[0]  # Use iloc for safety
        # Handle string case
        elif isinstance(self.InteratomicPotential, str):
            potential = self.InteratomicPotential
        else:
            return "UnknownPot"
        
        if 'MO_' in potential:
            IDs = potential.split('_')
            name = f"{IDs[0]}-{IDs[2]}-{IDs[3]}"
        else:
            name = potential
        
        return name
    def copy(self):
        """Create a deep copy of the dataclass."""
        from copy import deepcopy
        return deepcopy(self)
    
    def is_lammps(self) -> bool:
        """Check if using LAMMPS engine."""
        return self.calcengine.upper() == 'LAMMPS'
    
    def is_vasp(self) -> bool:
        """Check if using VASP engine."""
        return self.calcengine.upper() == 'VASP'
    
    def get_relevant_fields(self) -> dict:
        """Return only the fields relevant to the current calc engine."""
        base_fields = {
            'Element': self.Element,
            'MillerIndices': self.MillerIndices,
            'SuperCellDimensions': self.SuperCellDimensions,
            'acell': self.acell,
            'calcengine': self.calcengine,
            'vacuum': self.vacuum,
            'etol': self.etol,
            'ftol': self.ftol,
            'calctype': self.calctype,
            'ProjectName': self.ProjectName,
            'Strainlist': self.Strainlist,
        }
        
        if self.is_lammps():
            base_fields.update({
                'InteratomicPotential': self.InteratomicPotential,
                'MinimizerForCompleteRelaxation': self.MinimizerForCompleteRelaxation,
                'MinimizerForSurfaces': self.MinimizerForSurfaces,
            })
        elif self.is_vasp():
            base_fields.update({
                'Encut': self.Encut,
                'Kmesh': self.Kmesh,
            })
        
        return base_fields

@pwf.as_dataclass_node
class WorkflowState:
    """Lean workflow state for node-to-node communication."""
    
    # ===== Essential for computations =====
    ProjectName: str
    calcengine: str
    etol: float
    ftol: float
    verbose: bool
    vacuum: int
    calctype: str = 'relax'
    DeformationMode: str = 'AFFINE'
    
    # Engine-specific parameters
    InteratomicPotential: Optional[Union[pd.DataFrame, str]] = None
    MinimizerForSurfaces: Optional[str] = None
    Encut: Optional[int] = None
    Kmesh: Optional[list[int]] = None
    
    # ===== Structure attributes (added dynamically through workflow) =====
    RelaxedStructure: Optional[ase.Atoms] = None
    StrainedStructure: Optional[ase.Atoms] = None
    Structure_Minimized: Optional[ase.Atoms] = None
    Structure_w_vacuum: Optional[ase.Atoms] = None
    
    # ===== Computed results =====
    acell_relaxed: Optional[Union[np.ndarray, float]] = None
    Cell_RelaxedStructure: Optional[np.ndarray] = None
    
    Energy: Optional[float] = None
    Pressures: Optional[np.ndarray] = None
    ForceMax: Optional[float] = None
    Cell: Optional[np.ndarray] = None
    
    def copy(self):
        """Create a deep copy of the workflow state."""
        from copy import deepcopy
        return deepcopy(self)
    
    def populate_from_geometry_optimization(
        self,
        I,  # DataInputs
        RelaxedStructure,
        acell_relaxed,
        Cell_RelaxedStructure,
        Energy=None,
        Pressures=None,
        ForceMax=None
    ):
        """
        Populate this WorkflowState from DataInputs + GeometryOptimization results.
        
        This is the bridge from the heavy DataInputs to the lean WorkflowState.
        """
        self.ProjectName = I.ProjectName
        self.calcengine = I.calcengine
        self.etol = I.etol
        self.ftol = I.ftol
        self.verbose = I.verbose
        self.vacuum = I.vacuum
        self.calctype = I.calctype
        self.InteratomicPotential = I.InteratomicPotential
        self.MinimizerForSurfaces = I.MinimizerForSurfaces
        self.Encut = I.Encut
        self.Kmesh = I.Kmesh
        self.RelaxedStructure = RelaxedStructure
        self.acell_relaxed = acell_relaxed
        self.Cell_RelaxedStructure = Cell_RelaxedStructure
        self.Energy = Energy
        self.Pressures = Pressures
        self.ForceMax = ForceMax
        self.Cell = Cell_RelaxedStructure
        
        return self
    
    def is_lammps(self) -> bool:
        """Check if using LAMMPS engine."""
        return self.calcengine.upper() == 'LAMMPS'
    
    def is_vasp(self) -> bool:
        """Check if using VASP engine."""
        return self.calcengine.upper() == 'VASP'





@pwf.as_dataclass_node
class GammaSurfaceInputs:
    # Required fields for all calc engines
    Element: str
    SuperCellDimensions: list[int]
    acell: float
    vacuum: int = 5
    calcengine: str = 'Lammps'  # Make this early so __post_init__ can use it
    
    # Conditionally relevant fields - these will be set in __post_init__
    InteratomicPotential: Optional[Union[pd.DataFrame, str]] = None
    Encut: Optional[int] = None
    Kmesh: Optional[list[int]] = None
    MinimizerForCompleteRelaxation: Optional[str] = None
    MinimizerForSurfaces: Optional[str] = None
    
    # Common fields with defaults
    etol: float = 0.0
    ftol: float = 1e-8
    MillerIndices: tuple = (1,1,1)
    
    # GSFE-specific parameters (replacing strain parameters)
    FractionMinimum: float = 0.0
    FractionMaximum: float = 1.0
    FractionStateCount: int = 51
    DeformationMode: str = 'AFFINE'
    
    StorageDataFrame: Union[str, os.PathLike] = 'Storage.pckl'
    UserComment: str = ''
    calctype: str = 'relax'
    verbose: bool = False
    delete_existing_jobs: bool = False
    delete_aborted_jobs: bool = True
    decompress: bool = False
    submit_job: bool = False
    queue: str = 'cmmg'
    
    # Computed fields (set in __post_init__, but can be overridden)
    ProjectName: Optional[str] = None  # Allow manual override
    Fractionlist: list[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Set engine-specific defaults and compute derived fields."""
        
        import numpy as np

        if self.DeformationMode.upper() not in ['AFFINE', 'ALIAS']:
            raise ValueError(
                f"DeformationMode must be 'AFFINE' or 'ALIAS', got '{self.DeformationMode}'"
            )
        self.DeformationMode = self.DeformationMode.upper()
        
        
        # Set defaults based on calc engine
        if self.calcengine.upper() == 'LAMMPS':
            self._set_lammps_defaults()
        elif self.calcengine.upper() == 'VASP':
            self._set_vasp_defaults()
        else:
            raise ValueError(f"Unknown calc engine: {self.calcengine}")
        
        # Compute derived fields
        self._compute_project_name()
        self._compute_fraction_list()
    
    def _set_lammps_defaults(self):
        """Set LAMMPS-specific field values."""
        # LAMMPS doesn't use these DFT parameters
        self.Encut = None  # More explicit than 0
        self.Kmesh = None
        
        # Set defaults if not provided
        print(self.InteratomicPotential)
        if self.InteratomicPotential is None:
            raise ValueError("InteratomicPotential is required for LAMMPS")
        if self.MinimizerForCompleteRelaxation is None:
            self.MinimizerForCompleteRelaxation = 'cg'
        if self.MinimizerForSurfaces is None:
            self.MinimizerForSurfaces = 'fire'
    
    def _set_vasp_defaults(self):
        """Set VASP-specific field values."""
        # VASP doesn't use these classical MD parameters  
        self.InteratomicPotential = None
        self.MinimizerForCompleteRelaxation = None
        self.MinimizerForSurfaces = None
        self.submit_job = True
        
        # Set defaults if not provided
        if self.Encut is None:
            self.Encut = 750
        if self.Kmesh is None:
            self.Kmesh = [12, 12, 12]
            
        if self.vacuum > 25.0:
            self.vacuum = 10.0
        
        # GSFE-specific defaults for VASP
        # Typically scan the full Burgers vector for GSFE
        self.FractionMaximum = 1.0
        self.FractionMinimum = 0.0
        self.FractionStateCount = 11
    
    def _compute_project_name(self):
        """Compute project name based on engine and parameters."""
        # Only compute if not manually set
        if self.ProjectName is not None:
            return  # Keep the manually provided name
            
        if self.calcengine.upper() == 'LAMMPS':
            self.ProjectName = (
                f"{self.Element}_"
                f"{self.Pot_to_name()}_"
                f"{'x'.join(str(i) for i in self.SuperCellDimensions)}_"
                f"{self.MinimizerForSurfaces}_"
                f"GSFE_"
                f"{self.UserComment}"
            )
        elif self.calcengine.upper() == 'VASP':
            self.ProjectName = (
                f"{self.Element}_"
                f"{self.Encut}_"
                f"{'x'.join(str(i) for i in self.Kmesh)}_"
                f"{'x'.join(str(i) for i in self.SuperCellDimensions)}_"
                f"GSFE_"
                f"{self.UserComment}"
            )
    
    def _compute_fraction_list(self):
        """Compute the fraction list for GSFE calculations."""
        self.Fractionlist = [
            round(fraction, 6) 
            for fraction in np.linspace(
                self.FractionMinimum, 
                self.FractionMaximum, 
                num=self.FractionStateCount, 
                endpoint=True
            ).tolist()
        ]
        #self.Fractionlist.reverse()
    
    def Pot_to_name(self) -> str:
        """Extract potential name for project naming."""
        if self.InteratomicPotential is None:
            return "NoPot"
            
        # Handle DataFrame case
        if isinstance(self.InteratomicPotential, pd.DataFrame):
            potential = self.InteratomicPotential['Name'].iloc[0]  # Use iloc for safety
        # Handle string case
        elif isinstance(self.InteratomicPotential, str):
            potential = self.InteratomicPotential
        else:
            return "UnknownPot"
        
        if 'MO_' in potential:
            IDs = potential.split('_')
            name = f"{IDs[0]}-{IDs[2]}-{IDs[3]}"
        else:
            name = potential
        
        return name
    
    def is_lammps(self) -> bool:
        """Check if using LAMMPS engine."""
        return self.calcengine.upper() == 'LAMMPS'
    
    def is_vasp(self) -> bool:
        """Check if using VASP engine."""
        return self.calcengine.upper() == 'VASP'
    
    def get_relevant_fields(self) -> dict:
        """Return only the fields relevant to the current calc engine."""
        base_fields = {
            'Element': self.Element,
            'MillerIndices': self.MillerIndices,
            'SuperCellDimensions': self.SuperCellDimensions,
            'acell': self.acell,
            'calcengine': self.calcengine,
            'vacuum': self.vacuum,
            'etol': self.etol,
            'ftol': self.ftol,
            'calctype': self.calctype,
            'ProjectName': self.ProjectName,
            'Fractionlist': self.Fractionlist,
        }
        
        if self.is_lammps():
            base_fields.update({
                'InteratomicPotential': self.InteratomicPotential,
                'MinimizerForCompleteRelaxation': self.MinimizerForCompleteRelaxation,
                'MinimizerForSurfaces': self.MinimizerForSurfaces,
            })
        elif self.is_vasp():
            base_fields.update({
                'Encut': self.Encut,
                'Kmesh': self.Kmesh,
            })
        
        return base_fields