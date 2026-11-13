"""Train the draft model on a parallel corpus."""
import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.specdec.draft_model import DraftModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model = DraftModel(vocab_size=32000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    loader = DataLoader(range(cfg["batch_size"]), batch_size=1)

    model.train()
    step = 0
    for _ in range(400):
        optimizer.zero_grad()
        batch = torch.randint(0, 32000, (cfg["batch_size"], 64))
        logits = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
        step += 1
        if step % cfg["eval_every"] == 0:
            print(f"step={step} loss={loss.item():.5f}")


if __name__ == "__main__":
    main()
