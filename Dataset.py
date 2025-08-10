#Dataset.py
import os, re, torch, pathlib
from torch.utils.data import Dataset

class DefectSynthesisDataset(Dataset):
    """
    데이터 구조
    root_dir/
        ├─ audiojack/ OK/ NG/ Full_NG/*.pt
        ├─ bottle_cap/ ...
        └─ …
    각 .pt 파일 == 이미지 텐서 리스트(list[Tensor])
    """

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        status_dirs=("OK", "NG", "Full_NG"),
        transform=None,
        cache_in_ram: bool = True,          # .pt → RAM 캐싱 여부
        max_samples: int | None = None,     # 디버그용 부분 샘플
    ):
        self.root_dir     = pathlib.Path(root_dir)
        self.status_dirs  = status_dirs
        self.transform    = transform
        self.cache_in_ram = cache_in_ram
        self.max_samples  = max_samples

        self.class2idx, self.defect2idx = {}, {}
        self.samples: list[tuple[pathlib.Path, pathlib.Path, pathlib.Path,
                                 int, int, int]] = []   # (ok_p, ng_p, tgt_p, img_idx, cls_id, dft_id)
        self._file_cache: dict[pathlib.Path, list[torch.Tensor]] = {}   # RAM 캐시

        # ────────────────────────────────────────────────
        # 1. 경로 수집
        # ────────────────────────────────────────────────
        for cls_idx, cls_path in enumerate(sorted(self.root_dir.iterdir())):
            if not cls_path.is_dir() or cls_path.name == "lost+found":
                continue

            # 상태 폴더 누락 시 스킵
            if any(not (cls_path / s).is_dir() for s in self.status_dirs):
                print(f"[WARN] {cls_path.name} 에 상태 폴더 누락 → 건너뜀")
                continue

            self.class2idx.setdefault(cls_path.name, cls_idx)
            ok_dir, ng_dir, full_dir = [cls_path / s for s in self.status_dirs]

            ok_files   = {f.name: f for f in ok_dir.glob("*.pt")}
            ng_files   = {f.name: f for f in ng_dir.glob("*.pt")}
            full_files = {f.name: f for f in full_dir.glob("*.pt")}

            for fname in sorted(ok_files.keys() & ng_files.keys() & full_files.keys()):
                defect      = self._extract_defect_type(fname)
                ok_list     = self._load_pt(ok_files[fname])
                ng_list     = self._load_pt(ng_files[fname])
                full_list   = self._load_pt(full_files[fname])
                assert len(ok_list) == len(ng_list) == len(full_list)

                self.defect2idx.setdefault(defect, len(self.defect2idx))
                for img_idx in range(len(ok_list)):
                    self.samples.append(
                        (ok_files[fname], ng_files[fname], full_files[fname],
                         img_idx, cls_idx, self.defect2idx[defect])
                    )
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        break
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
            if self.max_samples and len(self.samples) >= self.max_samples:
                break

        print(f"[INFO] 샘플 {len(self.samples):,}개 | 클래스 {len(self.class2idx)} | 결함 {len(self.defect2idx)}")

    # ────────────────────────────────────────────────
    # 2. .pt 로더 (GPU → CPU 강제 + RAM 캐시)
    # ────────────────────────────────────────────────
    def _load_pt(self, path: pathlib.Path):
        """
        .pt 파일을 반드시 CPU Tensor로 로드.
        cache_in_ram=True이면 파일당 한 번만 읽고 self._file_cache에 저장.
        """
        def to_cpu(storage, _):       # GPU storage를 CPU로 강제 매핑
            return storage.cpu()

        if not self.cache_in_ram:     # 캐시 OFF
            return torch.load(path, map_location=to_cpu)

        if path not in self._file_cache:
            data = torch.load(path, map_location=to_cpu)
            # 리스트 형태라면 각 원소도 CPU 보장
            if isinstance(data, list):
                data = [t.cpu() for t in data]
            else:
                data = data.cpu()
            self._file_cache[path] = data

        return self._file_cache[path]

    @staticmethod
    def _extract_defect_type(filename: str) -> str:
        """ scratch_0001.pt → scratch """
        return re.split(r"[_\-]", pathlib.Path(filename).stem)[0] or "unknown"

    # ────────────────────────────────────────────────
    # 3. 필수 메서드
    # ────────────────────────────────────────────────
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        ok_p, ng_p, tgt_p, img_idx, cls_id, dft_id = self.samples[idx]

        ok   = self._load_pt(ok_p)[img_idx]
        mask = self._load_pt(ng_p)[img_idx]
        tgt  = self._load_pt(tgt_p)[img_idx]

        if self.transform:
            ok, mask, tgt = self.transform(ok), self.transform(mask), self.transform(tgt)

        sample = {
            "ok": ok,
            "ng_mask": mask,
            "ng_full": tgt,
            "class_id":  torch.tensor(cls_id, dtype=torch.long),
            "defect_id": torch.tensor(dft_id, dtype=torch.long),
        }
        # 텐서 참조 충돌 방지
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}

