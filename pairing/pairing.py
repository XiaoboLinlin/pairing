from copy import deepcopy
import itertools
import numpy as np
import mdtraj as md
from pairing.utils.misc import make_comtrj
import math


def calc_indirect(direct_array):
    """
    Calculate indirect matrices for all frames of trajectory
    """
    indirect_results = []
    for matrix in direct_array:
        indirect = _generate_indirect_connectivity(matrix)
        indirect_results.append(np.asarray(indirect))
    return(indirect_results)

def calc_reduc(indirect_array):
    """
    Reduce indirect matrices for all frames of trajectory
    """
    reduc_results = []
    for matrix in indirect_array:
        reduc = _generate_clusters(matrix)
        reduc_results.append(np.asarray(reduc))
    return(reduc_results)


def check_pairs(trj, cutoff, first_direct):
    """
    trj = trj_tail_carbon
    Checks pairs at various frames against direct correlation matrix 
    from frame zero. Note: Because the function only checks existing pairs, the existing pairs will gradually decrease. 
    """
    direct_list = []
    for frame in trj:
        c = deepcopy(first_direct)
        matrix = _check_direct(c, frame, cutoff)
        direct_list.append(matrix)
    return direct_list


def calc_direct(trj, cutoff=1.0):
    """
    calculate direct matrices for all frames of trajectory
    """
#     com_trj = make_comtrj(trj)
    direct_list = []
    for frame in trj:
        direct = _generate_direct_correlation(frame, cutoff)
        direct_list.append(direct)
    
    return direct_list


# def _generate_direct_correlation(trj, cutoff=1.0):
#     """
#     Generate direct correlation matrix from a COM-based mdtraj.Trajectory.

#     Parameters
#     ----------
#     trj : mdtraj.Trajectory
#         Trajectory for which "atom" sites are to be considered
#     cutoff : float, default = 0.8
#         Distance cutoff below which two sites are considered paired

#     Returns
#     -------
#     direct_corr : np.ndarray, dtype=np.int32
#         Direct correlation matrix
#     """
#     size = trj.top.n_residues
#     direct_corr = np.zeros((size, size), dtype=np.int32)

#     for row in range(size):
#         for col in range(size):
#             if row == col:
#                 direct_corr[row, col] = 1
#             else:
#                 dist = md.compute_distances(trj, atom_pairs=[(row, col)])
#                 if dist < cutoff:
#                     direct_corr[row, col] = 1
#                     direct_corr[col, row] = 1

#     return direct_corr

def _generate_direct_correlation(trj_tail_carbon, cutoff_input=1.0):
    """
    This is a special function for carbon atoms in cation tails. 
    Generate direct correlation matrix from a COM-based mdtraj.Trajectory.

    Parameters
    ----------
    trj_tail_carbon : mdtraj.Trajectory
        trj_tail_carbon is for all carbons in nonpolar tails.
        Trajectory for which "atom" sites are to be considered
    cutoff : float, default = 0.8
        Distance cutoff below which two sites are considered paired

    Returns
    -------
    direct_corr : np.ndarray, dtype=np.int32
        Direct correlation matrix
    """

    size = trj_tail_carbon.top.n_residues
    size_total = trj_tail_carbon.top.n_atoms
    direct_corr = np.zeros((size, size), dtype=np.int32)
    np.fill_diagonal(direct_corr, 1)
    tail_atom = size_total/size
    for i in range(size):
        i_start = int(i * tail_atom)
        i_end = int(i * tail_atom + tail_atom)
        index = np.array([j for j in range(i_start, i_end)])
        neighor_atoms = md.compute_neighbors(trj_tail_carbon, cutoff = cutoff_input, query_indices = index)
        diff = np.setdiff1d(neighor_atoms, index)
        for atom in diff:
            tail_index = math.ceil((atom+1)/tail_atom) -1 # from atom index to get tail index
            direct_corr[i][tail_index] = 1
    return direct_corr



def _check_direct(direct_corr, trj_tail_carbon, cutoff):
    """
    trj_tail_carbon = frame
    
    Check if paired atoms are still paired
    """
    size = trj_tail_carbon.top.n_residues
    size_total = trj_tail_carbon.top.n_atoms
    direct_corr = np.zeros((size, size), dtype=np.int32)
    tail_atom = size_total/size
    
    for row in range(size):
        for col in range(size):
            if direct_corr[row][col] == 1:
                dist = md.compute_distances(trj_tail_carbon, 
                                            atom_pairs=[(row_i, col_j)])
                if dist < cutoff:
                    continue
                else:
                    direct_corr[row][col] = 0
                    direct_corr[col][row] = 0

    return direct_corr


def _generate_indirect_connectivity(direct_corr):
    """
    Parameters
    ----------
    direct_corr: np.ndarray, dtype=np.int32
    direct correlation matrix

    Returns
    -------
    indirect: np.ndarray, dtype=np.int32
    indirect connectivity matrix
    """
    c = deepcopy(direct_corr)
    for row in c:
        ones = np.where(row == 1)[0]
        if len(ones) == 1:
            continue
        else:
            intersect = np.maximum.reduce(
                    [c[:,ele] for ele in ones])
            for ele in ones:
                c[:,ele] = intersect
    indirect = c

    return indirect


def _generate_clusters(indirect):
    """
    Generate clusters by reducing the indirect matrix

    Parameters
    ----------
    indirect_corr : numpy.ndarray, dtype=np.int32
        Indirect corrlation matrix

    Returns
    -------
    clusters : numpy.ndarray, dtype=np.int32
        Matrix in which each column represents a cluster and corresponding
        row indices match the indices of sites in a given cluster.
    """
    clusters = np.unique(indirect, axis=1)
    return clusters


def analyze_clusters(clusters):
    """
    Find the average and standard deviation of cluster sizes.

    Parameters
    ----------
    clusters : numpy.ndarray, dtype=np.int32
        Matrix in which each column represents a cluster and corresponding

    Returns
    -------
    avg : float
        Average cluster size
    stdev : float
        Standard deviation of cluster sizes
    """
    cluster_sizes = np.sum(clusters, axis=0)
    avg = np.mean(cluster_sizes)
    stdev = np.std(cluster_sizes)
    return avg, stdev
