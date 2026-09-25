"""Behavior locks for the shared scalar/batch MACE graph primitives.

These CPU tests inspect graph/model inputs, not MACE model predictions.
"""

import unittest

import torch
from ase import Atoms

from maple.function.calculator.mace import _batch_graph, _common


class MACEBatchHelperParityTests(unittest.TestCase):
    def test_batch_one_hot_preserves_required_dtype_and_tuple_error(self):
        with self.assertRaises(TypeError):
            _batch_graph.one_hot_node_attrs(torch.tensor([1]), [1])
        with self.assertRaisesRegex(
            ValueError,
            r"Atomic number\(s\) \[8\] not in AtomicNumberTable \[1, 6\]",
        ):
            _batch_graph.one_hot_node_attrs(
                torch.tensor([8]), (1, 6), dtype=torch.float64
            )

    def test_one_hot_dtype_table_order_and_missing_element(self):
        numbers = torch.tensor([6, 1, 6], dtype=torch.long)
        for helper in (_common.one_hot_node_attrs, _batch_graph.one_hot_node_attrs):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(helper=helper.__module__, dtype=dtype):
                    attrs = helper(numbers, [1, 6, 8], dtype=dtype)
                    self.assertEqual(attrs.dtype, dtype)
                    torch.testing.assert_close(
                        attrs,
                        torch.tensor(
                            [[0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=dtype
                        ),
                        rtol=0,
                        atol=0,
                    )
            with self.subTest(helper=helper.__module__, case="missing element"):
                with self.assertRaisesRegex(
                    ValueError,
                    r"Atomic number\(s\) \[8\] not in AtomicNumberTable \[1, 6\]",
                ):
                    helper(torch.tensor([1, 8]), [1, 6], dtype=torch.float64)

    def test_radius_cutoff_edge_order_and_shift_dtype(self):
        for helper in (_common.radius_graph_no_pbc, _batch_graph.radius_graph_no_pbc):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(helper=helper.__module__, dtype=dtype):
                    positions = torch.tensor(
                        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        dtype=dtype,
                    )
                    edge_index, shifts = helper(positions, 0.75)
                    torch.testing.assert_close(
                        edge_index,
                        torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(edge_index.dtype, torch.long)
                    self.assertEqual(shifts.dtype, dtype)
                    self.assertEqual(tuple(shifts.shape), (4, 3))
                    self.assertTrue(torch.equal(shifts, torch.zeros_like(shifts)))

            # FP64 distinguishes the existing 1e-12 cutoff slack from a
            # strictly farther pair. Do not replace this with a new tolerance.
            for x, expected_edges in ((1.0, 2), (1.0 + 2e-12, 0)):
                with self.subTest(helper=helper.__module__, distance=x):
                    positions = torch.tensor(
                        [[0.0, 0.0, 0.0], [x, 0.0, 0.0]], dtype=torch.float64
                    )
                    edge_index, shifts = helper(positions, 1.0)
                    self.assertEqual(tuple(edge_index.shape), (2, expected_edges))
                    self.assertEqual(tuple(shifts.shape), (expected_edges, 3))

    def test_batched_graph_b1_and_disconnected_multi_edge_order(self):
        first = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        second = torch.tensor(
            [[10.0, 0.0, 0.0], [10.5, 0.0, 0.0]], dtype=torch.float64
        )
        cases = (
            (first, [3], [[0, 1, 1, 2], [1, 2, 0, 1]]),
            (
                torch.cat([first, second]),
                [3, 2],
                [[0, 1, 1, 2, 3, 4], [1, 2, 0, 1, 4, 3]],
            ),
        )
        for positions, counts, expected in cases:
            with self.subTest(counts=counts):
                edge_index, shifts = _batch_graph.batched_radius_graph_no_pbc(
                    positions, counts, 0.75
                )
                torch.testing.assert_close(
                    edge_index, torch.tensor(expected), rtol=0, atol=0
                )
                self.assertEqual(shifts.dtype, positions.dtype)
                self.assertEqual(tuple(shifts.shape), (len(expected[0]), 3))
                self.assertTrue(torch.equal(shifts, torch.zeros_like(shifts)))

    def test_batch_builders_preserve_b1_and_disconnected_inputs(self):
        first = Atoms(
            "H3", positions=[[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]]
        )
        second = Atoms("CH", positions=[[10, 0, 0], [10.5, 0, 0]])
        for atoms_list, expected_ptr in (([first], [0, 3]), ([first, second], [0, 3, 5])):
            for dtype in (torch.float32, torch.float64):
                for builder in (
                    _batch_graph.build_mace_data_dict_batch,
                    _batch_graph.build_mace_tuple_batch,
                ):
                    with self.subTest(
                        builder=builder.__name__, count=len(atoms_list), dtype=dtype
                    ):
                        result = builder(
                            atoms_list,
                            atomic_numbers=[1, 6],
                            r_max=0.75,
                            device=torch.device("cpu"),
                            dtype=dtype,
                            requires_grad=True,
                        )
                        if builder is _batch_graph.build_mace_data_dict_batch:
                            data, local_or_ghost, counts = result
                            positions = data["positions"]
                            attrs = data["node_attrs"]
                            edge_index = data["edge_index"]
                            shifts = data["shifts"]
                            batch, ptr = data["batch"], data["ptr"]
                            self.assertEqual(tuple(local_or_ghost.shape), (len(positions),))
                            self.assertEqual(data["unit_shifts"].dtype, dtype)
                        else:
                            (positions, attrs, edge_index, shifts, batch, ptr), counts = result
                        self.assertEqual(counts, [len(atoms) for atoms in atoms_list])
                        self.assertEqual(ptr.tolist(), expected_ptr)
                        self.assertEqual(batch.tolist(), [0] * 3 + [1] * (len(positions) - 3))
                        self.assertEqual(positions.dtype, dtype)
                        self.assertTrue(positions.requires_grad)
                        self.assertEqual(attrs.dtype, dtype)
                        self.assertEqual(shifts.dtype, dtype)
                        self.assertTrue(torch.equal(shifts, torch.zeros_like(shifts)))
                        expected_edges = (
                            [[0, 1, 1, 2], [1, 2, 0, 1]]
                            if len(atoms_list) == 1
                            else [[0, 1, 1, 2, 3, 4], [1, 2, 0, 1, 4, 3]]
                        )
                        torch.testing.assert_close(
                            edge_index, torch.tensor(expected_edges), rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()
