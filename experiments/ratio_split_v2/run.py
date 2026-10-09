"""Repeat affected comparisons with source provenance assigned before insertion."""

from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "enhancement_v1"))
import mc as base
sys.path.insert(0, str(HERE.parent / "review_round3"))
from ratio import diagnose


class SplitTrainer(base.TransitH2OTrainerV7):
    def save_policy_artifact(self, path, **kwargs):
        replay = self.replay_buffer
        size = replay.online_size
        ids = replay._online_trajectory_ids[:size]
        mask = replay._online_ratio_validation[:size]
        np.testing.assert_array_equal(mask, replay._online_trajectory_validation_mask(ids))
        train_ids = sorted(set(ids[~mask].tolist()))
        validation_ids = sorted(set(ids[mask].tolist()))
        assert train_ids and validation_ids and not set(train_ids).intersection(validation_ids)
        result = super().save_policy_artifact(path, **kwargs)
        out = Path(path).parent
        report = diagnose(self)
        report.update(source_train_days=train_ids, source_validation_days=validation_ids,
                      split_revision="day_id_before_replay_append")
        assert report["validation_available"]
        base.write(out / "results/ratio_final.json", report)
        torch.save(dict(config=base.asdict(self.config), actor=self.actor.state_dict(),
                        ratio=self.ratio.state_dict(), total_updates=self.total_updates),
                   out / "ratio_export.pt")
        return result


if __name__ == "__main__":
    base.TransitH2OTrainerV7 = SplitTrainer
    base.main()
    print("DONE ratio split correction", flush=True)
