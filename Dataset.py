# Dataset.py

import os, re, torch, pathlib
from torch.utils.data import Dataset

class DefectSynthesisDataset(Dataset):
    """
    root_dir/
        ├─ audiojack/
        │   ├─ OK/        scratch_0001.pt …
        │   ├─ NG/        scratch_0001.pt …
        │   └─ Full_NG/   scratch_0001.pt …
        ├─ bottle_cap/ ...
        └─ lost+found/    # ← 자동으로 제외
    """

    def __init__(
        self,
        root_dir,
        status_dirs=("OK", "NG", "Full_NG"),
        transform=None,
    ):
        self.root_dir = pathlib.Path(root_dir)
        self.status_dirs = status_dirs
        self.transform = transform

        self.samples = []  # (ok_path, ng_path, full_path, class_id, defect_id)

        self.class2idx = {}
        self.defect2idx = {}

        # ────────────────────────────────
        # 1. 경로만 모아 두기 (lazy loading)
        # ────────────────────────────────
        for cls_idx, cls_path in enumerate(sorted(self.root_dir.iterdir())):
            if not cls_path.is_dir() or cls_path.name == "lost+found":
                continue
            self.class2idx.setdefault(cls_path.name, cls_idx)

            # 각 status 폴더에서 동일한 파일명을 기준으로 매칭
            ok_dir, ng_dir, full_dir = [cls_path / s for s in self.status_dirs]
            ok_files = {f.name: f for f in ok_dir.glob("*.pt")}
            ng_files = {f.name: f for f in ng_dir.glob("*.pt")}
            full_files = {f.name: f for f in full_dir.glob("*.pt")}

            # 세 폴더에 모두 존재하는 파일만 선택
            common_keys = ok_files.keys() & ng_files.keys() & full_files.keys()

            for fname in sorted(common_keys):
                defect = self._extract_defect_type(fname)
                ok_list   = torch.load(ok_files[fname])
                ng_list   = torch.load(ng_files[fname])
                full_list = torch.load(full_files[fname])

                assert len(ok_list) == len(ng_list) == len(full_list)
                for img_idx in range(len(ok_list)):
                    self.defect2idx.setdefault(defect, len(self.defect2idx))
                    self.samples.append(
                        (
                            ok_files[fname],
                            ng_files[fname],
                            full_files[fname],
                            img_idx,
                            cls_idx,
                            self.defect2idx[defect],
                        )
                    )

    # ────────────────────────────────
    # 2. 유틸
    # ────────────────────────────────
    @staticmethod
    def _extract_defect_type(filename: str) -> str:
        return re.split(r"[_\-]", pathlib.Path(filename).stem)[0]

    # ────────────────────────────────
    # 3. 필수 메서드
    # ────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ok_path, ng_path, full_path, img_idx, cls_id, defect_id = self.samples[idx]

        # .pt 파일 실제 로드 (lazy)
        ok_tensor   = torch.load(ok_path)[img_idx]
        ng_tensor   = torch.load(ng_path)[img_idx]
        full_tensor = torch.load(full_path)[img_idx]

        if self.transform is not None:
            ok_tensor   = self.transform(ok_tensor)
            ng_tensor   = self.transform(ng_tensor)
            full_tensor = self.transform(full_tensor)

        return {
            "ok": ok_tensor,           # OK 이미지 (i2i 입력)
            "ng_mask": ng_tensor,      # NG 마스크 (Control image)
            "ng_full": full_tensor,    # 합성·실측 NG 이미지 (target)
            "class_id": torch.tensor(cls_id, dtype=torch.long),
            "defect_id": torch.tensor(defect_id, dtype=torch.long),
        }
