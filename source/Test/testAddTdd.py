from source.TDD import TDD, Node, Index, Ini_TDD, get_index_2_key, Find_Or_Add_Unique_table, add, get_identity_tdd
import numpy as np
import unittest

NUMBER_OF_SUCCESSORS = 2


class TestAddTdd(unittest.TestCase):

    def test_add_same_identity_tdd(self):
        tdd = Ini_TDD([])
        add_tdd = add(tdd, tdd)
        self.assertEqual({-1: -1}, add_tdd.index_2_key)
        self.assertEqual({-1: -1}, add_tdd.key_2_index)
        self.assertEqual(2, add_tdd.to_array())

    def test_add_different_identity_tdd(self):
        tdd = Ini_TDD([])
        tdd2 = get_identity_tdd()
        add_tdd = add(tdd, tdd2)
        self.assertEqual({-1: -1}, add_tdd.index_2_key)
        self.assertEqual({-1: -1}, add_tdd.key_2_index)
        self.assertEqual(2, add_tdd.to_array())

    def test_add_same_tdd_rank_0_manually(self):
        Ini_TDD([])
        node = Node(-1)
        tdd = TDD(node)
        tdd.index_2_key = {-1: -1}
        tdd.key_2_index = {-1: -1}
        add_tdd = add(tdd, tdd)
        self.assertEqual({-1: -1}, add_tdd.index_2_key)
        self.assertEqual({-1: -1}, add_tdd.key_2_index)
        self.assertEqual(2, add_tdd.to_array())

    def test_add_different_tdd_rank_0_manually(self):
        Ini_TDD([])
        node = Node(-1)
        tdd = TDD(node)
        tdd.index_2_key = {-1: -1}
        tdd.key_2_index = {-1: -1}
        tdd2 = TDD(node)
        tdd2.weight = 2
        tdd2.index_2_key = {-1: -1}
        tdd2.key_2_index = {-1: -1}
        add_tdd = add(tdd, tdd2)
        self.assertEqual({-1: -1}, add_tdd.index_2_key)
        self.assertEqual({-1: -1}, add_tdd.key_2_index)
        self.assertEqual(3, add_tdd.to_array())

    def test_add_different_tdd_rank_0_auto(self):
        Ini_TDD([])
        node = Find_Or_Add_Unique_table(-1)
        idx_2_key, key_2_idx = get_index_2_key([])
        tdd = TDD(node)
        tdd.index_2_key = idx_2_key
        tdd.key_2_index = key_2_idx
        tdd2 = TDD(node)
        tdd2.weight = 2
        tdd.index_2_key = idx_2_key
        tdd.key_2_index = key_2_idx
        add_tdd = add(tdd, tdd2)
        self.assertEqual({-1: -1}, add_tdd.index_2_key)
        self.assertEqual({-1: -1}, add_tdd.key_2_index)
        self.assertEqual(3, add_tdd.to_array())

    def test_add_same_tdd_rank_1_manually(self):
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
        add_tdd = add(tdd, tdd)
        self.assertEqual({-1: -1, "x0": 0}, add_tdd.index_2_key)
        self.assertEqual({-1: -1, 0: "x0"}, add_tdd.key_2_index)
        self.assertTrue(np.array_equal([2]*NUMBER_OF_SUCCESSORS, add_tdd.to_array()))

    def test_add_different_tdd_rank_1_manually(self):
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
        node_head = Node(0)
        node_head.idx = 2
        node_head.successor = [node_edge] * NUMBER_OF_SUCCESSORS
        tdd2 = TDD(node_head)
        tdd2.index_2_key = {-1: -1, "x0": 0}
        tdd2.key_2_index = {-1: -1, 0: "x0"}
        tdd2.index_set = indices
        tdd2.key_width = {0: NUMBER_OF_SUCCESSORS}
        add_tdd = add(tdd, tdd2)
        self.assertEqual({-1: -1, "x0": 0}, add_tdd.index_2_key)
        self.assertEqual({-1: -1, 0: "x0"}, add_tdd.key_2_index)
        self.assertTrue(np.array_equal([2]*NUMBER_OF_SUCCESSORS, add_tdd.to_array()))

    def test_add_different_tdd_rank_1_auto(self):
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
        tdd2 = TDD(node_head)
        tdd2.weight = 2
        tdd2.index_2_key = idx_2_key
        tdd2.key_2_index = key_2_idx
        tdd2.index_set = indices
        tdd2.key_width = {0: NUMBER_OF_SUCCESSORS}
        add_tdd = add(tdd, tdd2)
        self.assertEqual({-1: -1, "x0": 0}, add_tdd.index_2_key)
        self.assertEqual({-1: -1, 0: "x0"}, add_tdd.key_2_index)
        self.assertTrue(np.array_equal([3]*NUMBER_OF_SUCCESSORS, add_tdd.to_array()))


if __name__ == '__main__':
    unittest.main()
