"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""


from source.TDD import TDD, Node, Index, Ini_TDD, get_index_2_key, Find_Or_Add_Unique_table, equal_tolerance
import numpy as np
import unittest

# Can change if you want to test the DDs with more successors. Not intended to change.
NUMBER_OF_SUCCESSORS = 2


class TestToArray(unittest.TestCase):
    """
        Suite designed to testing the method 'to_array' of the class TDD.

        In the 'auto' tests, we create the TDDs by using the methods given by the library.
        In the 'manually' tests, we create the TDDs by manually setting all the information they needed.
    """

    def test_identity_tdd(self):
        tdd = Ini_TDD([])
        self.assertEqual(tdd.to_array(), 1)

    def test_make_tdd_rank_0_manually(self):
        Ini_TDD([])
        node = Node(-1)
        tdd = TDD(node)
        tdd.index_2_key = {-1: -1}
        tdd.key_2_index = {-1: -1}
        self.assertEqual(tdd.to_array(), 1)

    def test_make_tdd_rank_0_auto(self):
        Ini_TDD([])
        node = Find_Or_Add_Unique_table(-1)
        idx_2_key, key_2_idx = get_index_2_key([])
        tdd = TDD(node)
        tdd.index_2_key = idx_2_key
        tdd.key_2_index = key_2_idx
        self.assertEqual(tdd.to_array(), 1)

    def test_make_tdd_rank_1_manually(self):
        Ini_TDD(["x0"])
        indices = [Index("x0")]
        node_head = Node(0)
        node_head.idx = 1
        node_edge = Node(-1)
        node_head.successor = [node_edge] * NUMBER_OF_SUCCESSORS
        tdd = TDD(node_head)
        tdd.index_2_key = {-1: -1, "x0": 0}
        tdd.key_2_index = {-1: -1, 0: "x0"}
        tdd.index_set = indices
        tdd.key_width = {0: NUMBER_OF_SUCCESSORS}
        self.assertTrue(equal_tolerance(tdd.to_array(), [1]*NUMBER_OF_SUCCESSORS))

    def test_make_tdd_rank_1_auto(self):
        Ini_TDD(["x0"])
        indices = [Index("x0")]
        node_edge = Find_Or_Add_Unique_table(-1)
        node_head = Find_Or_Add_Unique_table(0,
                                             [np.float64(1)]*NUMBER_OF_SUCCESSORS,
                                             [node_edge]*NUMBER_OF_SUCCESSORS)
        idx_2_key, key_2_idx = get_index_2_key(indices)
        tdd = TDD(node_head)
        tdd.index_2_key = idx_2_key
        tdd.key_2_index = key_2_idx
        tdd.index_set = indices
        tdd.key_width = {0: NUMBER_OF_SUCCESSORS}
        self.assertTrue(equal_tolerance(tdd.to_array(), [1]*NUMBER_OF_SUCCESSORS))

    def test_make_tdd_rank_2_auto(self):
        Ini_TDD(["x0", "x1"])
        indices = [Index("x0"), Index("x1")]
        node_edge = Find_Or_Add_Unique_table(-1)
        node_middle = Find_Or_Add_Unique_table(0,
                                             [np.float64(1)]*NUMBER_OF_SUCCESSORS,
                                             [node_edge]*NUMBER_OF_SUCCESSORS)
        node_head = Find_Or_Add_Unique_table(0,
                                             [np.float64(1)]*NUMBER_OF_SUCCESSORS,
                                             [node_middle]*NUMBER_OF_SUCCESSORS)
        idx_2_key, key_2_idx = get_index_2_key(indices)
        tdd = TDD(node_head)
        tdd.index_2_key = idx_2_key
        tdd.key_2_index = key_2_idx
        tdd.index_set = indices
        tdd.key_width = {0: NUMBER_OF_SUCCESSORS, 1: NUMBER_OF_SUCCESSORS}
        self.assertTrue(equal_tolerance(tdd.to_array(), [[1]*NUMBER_OF_SUCCESSORS]*NUMBER_OF_SUCCESSORS))


if __name__ == '__main__':
    unittest.main()
