import ase
import numpy as np
def _get_structure(
        structure: ase.Atoms,
        cell: np.ndarray,
        indices: np.ndarray,
        positions: np.ndarray | None = None,
        unwrapped_positions: np.ndarray | None = None,
        total_displacements: np.ndarray | None = None,
        *,
        wrap_atoms: bool = True,
    ) -> ase.Atoms:
    
    from structuretoolkit.common import center_coordinates_in_unit_cell
    from pyiron import ase_to_pyiron
    """Return an updated `Atoms` object based on the provided information.

    Parameters
    ----------
    structure : Atoms
        The reference atomic structure.
    cell : ndarray
        The simulation cell to assign to the new structure.
    indices : ndarray
        Indices of the atoms to include in the new snapshot.
    positions : ndarray, optional
        Wrapped atomic positions.
    unwrapped_positions : ndarray, optional
        Unwrapped atomic positions.
    total_displacements : ndarray, optional
        Total atomic displacements to be added to the initial positions.
    wrap_atoms : bool, optional
        Whether to wrap atoms inside the unit cell (default is True).

    Returns
    -------
    Atoms
        The newly constructed atomic structure with updated positions and cell.

    """
    if indices is not None and len(indices) != len(structure):
        snapshot = Atoms(
            positions=np.zeros((*indices.shape, 3)),
            cell=cell,
            pbc=structure.pbc,
        )
        snapshot.set_array("indices", indices)
    else:
        snapshot = structure.copy()
        if cell is not None:
            snapshot.cell = cell
        if indices is not None:
            snapshot.set_array("indices", indices)

    if wrap_atoms:
        snapshot.positions = positions
        snapshot = center_coordinates_in_unit_cell(snapshot)
    elif unwrapped_positions is not None:
        snapshot.positions = unwrapped_positions
    else:
        snapshot.positions += total_displacements
        
    snapshot = ase_to_pyiron(snapshot)

    return snapshot