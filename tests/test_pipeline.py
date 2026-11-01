import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.adapters import prepare_cresci2017, prepare_mgtab
from src.models import GraphSAGE, MLP, RelationalGNN, normalized_adjacency
from scripts.pretrain_graph import FeatureEncoder


class PipelineTests(unittest.TestCase):
    def test_dataset_discovery_and_tensor_shapes(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "data/raw/mgtab/MGTAB/features.pt").is_file())
        data = torch.load(root / "data/processed/mgtab/data.pt", map_location="cpu", weights_only=True)
        self.assertEqual(tuple(data["x"].shape), (10199, 788))
        self.assertEqual(data["edge_index"].shape[1], data["edge_type"].numel())

    def test_mgtab_row_alignment_and_labels(self):
        root = Path(__file__).resolve().parents[1]
        data = torch.load(root / "data/processed/mgtab/data.pt", map_location="cpu", weights_only=True)
        self.assertEqual(data["y"].numel(), data["x"].shape[0])
        self.assertEqual(torch.bincount(data["y"]).tolist(), [7451, 2748])
        self.assertLessEqual(int(data["edge_index"].max()), data["x"].shape[0] - 1)

    def test_cresci15_official_splits_are_disjoint(self):
        root = Path(__file__).resolve().parents[1]
        data = torch.load(root / "data/processed/cresci2015/data.pt", map_location="cpu", weights_only=True)
        splits = data["splits"]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(set(splits[a].tolist()) & set(splits[b].tolist()))
        self.assertEqual(sum(v.numel() for v in splits.values()), 5301)

    def test_all_models_forward_backward_and_sparse_aggregation(self):
        x = torch.randn(6, 4)
        edge = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        edge_type = torch.tensor([0, 1, 0, 1, 0])
        adjacency = normalized_adjacency(edge, 6)
        models = [MLP(4, 8), GraphSAGE(4, 8), RelationalGNN(4, 2, 8)]
        args = [(x, edge, edge_type), (x, adjacency, edge_type),
                (x, tuple(normalized_adjacency(edge, 6, edge_type == r) for r in range(2)), edge_type)]
        for model, model_args in zip(models, args):
            output = model(*model_args)
            self.assertEqual(output.shape, (6,))
            output.sum().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_weighted_sparse_adjacency_and_link_pretraining_step(self):
        edge = torch.tensor([[0, 2], [1, 1]])
        adjacency = normalized_adjacency(edge, 3, edge_weight=torch.tensor([1.0, 3.0]))
        dense = adjacency.to_dense()
        self.assertAlmostEqual(float(dense[1, 0]), 0.25, places=5)
        self.assertAlmostEqual(float(dense[1, 2]), 0.75, places=5)
        encoder = FeatureEncoder(4, 5)
        z = encoder(torch.randn(6, 4))
        loss = torch.nn.functional.binary_cross_entropy_with_logits((z[0] * z[1]).sum().reshape(1), torch.ones(1))
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in encoder.parameters()))

    def test_checkpoint_round_trip_and_prediction_sample(self):
        model = MLP(3, 4)
        sample = torch.randn(2, 3)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), path)
            loaded = MLP(3, 4); loaded.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad(): probs = torch.sigmoid(loaded(sample))
            self.assertEqual(tuple(probs.shape), (2,))
            self.assertTrue(torch.all((probs >= 0) & (probs <= 1)))

    def test_duplicate_account_id_is_reported_and_deduplicated(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            source, output = Path(tmp) / "raw", Path(tmp) / "processed"
            source.mkdir()
            for group, uid in (("genuine_accounts", "1"), ("social_spambots_1", "1")):
                archive = source / f"{group}.csv.zip"
                with zipfile.ZipFile(archive, "w") as zf:
                    zf.writestr(f"{group}/users.csv", pd.DataFrame({"id": [uid], "followers_count": [3]}).to_csv(index=False))
            report = prepare_cresci2017(source, output)
            self.assertEqual(report["duplicate_account_ids_removed"], 1)
            self.assertEqual(len(pd.read_csv(output / "accounts.csv")), 1)

    def test_malformed_mgtab_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            source = Path(tmp); torch.save(torch.ones(3, 2), source / "features.pt")
            with self.assertRaises(FileNotFoundError): prepare_mgtab(source, source / "out")


if __name__ == "__main__": unittest.main()
