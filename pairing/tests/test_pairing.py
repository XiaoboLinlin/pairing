import numpy as np
import mdtraj as md

from pairing.utils.io import get_fn
import pairing
from pairing.utils.misc import make_comtrj


def test_generate_direct_correlation():
    trj = md.load(get_fn('sevick1988.gro'))

    ref = np.asarray([[1, 0, 0, 0, 1],
                      [0, 1, 1, 0, 0],
                      [0, 1, 1, 0, 1],
                      [0, 0, 0, 1, 0],
                      [1, 0, 1, 0, 1]], dtype=np.int32)

    gen = pairing.pairing._generate_direct_correlation(trj, cutoff=0.8)

    assert (ref == gen).all()


def test_sevick1988():
    """Test the system desribed in the appendix of Sevick 1988,
    doi 10.1063/1.454720"""
    c_D = np.asarray([[1, 0, 0, 0, 1],
                      [0, 1, 1, 0, 0],
                      [0, 1, 1, 0, 1],
                      [0, 0, 0, 1, 0],
                      [1, 0, 1, 0, 1]], dtype=np.int32)

    c_I = np.asarray([[1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [0, 0, 0, 1, 0],
                      [1, 1, 1, 0, 1]], dtype=np.int32)

    assert (c_I == pairing.pairing._generate_indirect_connectivity(c_D)).all()


def test_40_atoms():
    trj = md.load(get_fn('sevick1988.gro'))
    direct = pairing.pairing._generate_direct_correlation(trj, cutoff=0.8)
    indirect = pairing.pairing._generate_indirect_connectivity(direct)

    assert indirect.dtype == np.int32


def test_indirect_matrix_reduction():
    trj = md.load(get_fn('sevick1988.gro'))
    direct = pairing.pairing._generate_direct_correlation(trj, cutoff=0.8)
    indirect = pairing.pairing._generate_indirect_connectivity(direct)

    c_R = np.asarray([[0, 1],
                      [0, 1],
                      [0, 1],
                      [1, 0],
                      [0, 1]])

    assert (c_R == pairing.pairing._generate_clusters(indirect)).all()


def test_cluster_analysis():
    trj = md.load(get_fn('sevick1988.gro'))
    direct = pairing.pairing._generate_direct_correlation(trj, cutoff=0.8)
    indirect = pairing.pairing._generate_indirect_connectivity(direct)
    reduction = pairing.pairing._generate_clusters(indirect)

    assert pairing.analyze_clusters(reduction) == (2.5, 1.5)


def test_len_direct():
    trj = md.load(get_fn('tip3p_test.trr'), top=get_fn('tip3p_test.gro'))
    direct = pairing.calc_direct(trj, cutoff=0.3)

    assert len(direct) == trj.n_frames


def test_len_indirect():
    trj = md.load(get_fn('tip3p_test.trr'), top=get_fn('tip3p_test.gro'))
    direct = pairing.calc_direct(trj, cutoff=0.3)
    indirect = pairing.calc_indirect(direct)

    assert len(indirect) == trj.n_frames


def test_check_pairs():
    ref = np.asarray([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
    trj = md.load(get_fn('tip3p_test.trr'), top=get_fn('tip3p_test.gro'))
    first = make_comtrj(trj[0])
    first_direct = pairing.pairing._generate_direct_correlation(
            first, cutoff=0.3)
    check = pairing.check_pairs(trj, 0.3, first_direct)

    assert (check[20] == ref).all()


def test_len_check_pairs():
    trj = md.load(get_fn('tip3p_test.trr'), top=get_fn('tip3p_test.gro'))
    first = make_comtrj(trj[0])
    first_direct = pairing.pairing._generate_direct_correlation(
            first, cutoff=0.3)
    check = pairing.check_pairs(trj, 0.3, first_direct)

    assert len(check) == trj.n_frames
