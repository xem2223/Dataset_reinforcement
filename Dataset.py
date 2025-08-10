# Dataset.py
import os, re, torch, pathlib
from torch.utils.data import Dataset
from lru_cache import LRUCache

class DefectSynthesisDataset(Dataset):
    """
    디렉토리 구조(원본)
      root_dir/
        ├─ <class>/OK/*.pt
        ├─ <class>/NG/*.pt
        └─ <class>/Full_NG/*.pt
      각 .pt = list[Tensor]

    사전계산 사용 시(옵션)
      root_dir/
        ├─ <class>/OK_lat/*.pt         # list[Tensor: (4, 64, 64)]  fp16
        ├─ <class>/Full_NG_lat/*.pt    # list[Tensor: (4, 64, 64)]  fp16
        └─ <class>/NG_cond/*.pt        # list[Tensor: (3, 512, 512)] in [-1, 1], fp16
    """

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        status_dirs=("OK", "NG", "Full_NG"),
        transform=None,                   # (선택) 공용 변환 (이미지에만 적용 권장)
        transform_ok=None,                # (선택) OK 전용 변환
        transform_mask=None,              # (선택) 마스크 전용 변환
        transform_tgt=None,               # (선택) Full_NG 전용 변환
        cache_in_ram: bool = False,       # True면 LRU 캐시 사용 (RAM 여유 있을 때만)
        max_samples: int | None = None,
        use_latent: bool = True,         # True면 OK/Full_NG를 라티ent로 로드
        use_precond: bool = True,        # True면 NG를 전처리 조건(cond)로 로드
        lru_max_items: int | None = 256,  # LRU: 항목 수 제한
        lru_max_bytes: int | None = None  # LRU: 바이트 제한(예: 4*1024**3)
    ):
        self.root_dir     = pathlib.Path(root_dir)
        self.status_dirs  = status_dirs
        self.max_samples  = max_samples

        # 변환 설정: 개별 transform_*가 지정되지 않았으면 공용 transform을 사용
        self.transform_ok   = transform_ok   if transform_ok   is not None else transform
        self.transform_mask = transform_mask if transform_mask is not None else transform
        self.transform_tgt  = transform_tgt  if transform_tgt  is not None else transform

        self.cache_in_ram = cache_in_ram
        self._file_cache  = (LRUCache(max_items=lru_max_items, max_bytes=lru_max_bytes)
                             if cache_in_ram else None)

        self.use_latent  = use_latent
        self.use_precond = use_precond

        self.class2idx, self.defect2idx = {}, {}
        # (ok_path, ng_path, tgt_path, img_idx, class_id, defect_id)
        self.samples: list[tuple[pathlib.Path, pathlib.Path, pathlib.Path, int, int, int]] = []

        # ────────────────────────────────────────────────
        # 1) 클래스/파일 수집
        # ────────────────────────────────────────────────
        for cls_idx, cls_path in enumerate(sorted(self.root_dir.iterdir())):
            if not cls_path.is_dir() or cls_path.name == "lost+found":
                continue

            # 필수 상태 폴더 확인
            if any(not (cls_path / s).is_dir() for s in self.status_dirs):
                print(f"[WARN] {cls_path.name} 에 상태 폴더 누락 → 건너뜀")
                continue

            self.class2idx.setdefault(cls_path.name, cls_idx)
            ok_dir, ng_dir, full_dir = [cls_path / s for s in self.status_dirs]

            # 사전계산 경로(존재 시 사용)
            ok_lat_dir   = cls_path / "OK_lat"
            full_lat_dir = cls_path / "Full_NG_lat"
            ng_cond_dir  = cls_path / "NG_cond"

            ok_files   = {f.name: f for f in ok_dir.glob("*.pt")}
            ng_files   = {f.name: f for f in ng_dir.glob("*.pt")}
            full_files = {f.name: f for f in full_dir.glob("*.pt")}

            # 세 폴더 교집합 파일명만 사용
            for fname in sorted(ok_files.keys() & ng_files.keys() & full_files.keys()):
                defect = self._extract_defect_type(fname)

                # 길이 일치 확인을 위해 원본 리스트 길이 확인(최초 1회 로드)
                ok_list   = self._load_pt(ok_files[fname])
                ng_list   = self._load_pt(ng_files[fname])
                full_list = self._load_pt(full_files[fname])
                assert len(ok_list) == len(ng_list) == len(full_list), f"길이 불일치: {cls_path.name}/{fname}"

                self.defect2idx.setdefault(defect, len(self.defect2idx))

                # 사전계산 파일 경로(있으면 대체)
                ok_sel   = ok_lat_dir   / fname if (self.use_latent and (ok_lat_dir   / fname).exists()) else ok_files[fname]
                tgt_sel  = full_lat_dir / fname if (self.use_latent and (full_lat_dir / fname).exists()) else full_files[fname]
                ng_sel   = ng_cond_dir  / fname if (self.use_precond and (ng_cond_dir  / fname).exists()) else ng_files[fname]

                for img_idx in range(len(ok_list)):
                    self.samples.append((ok_sel, ng_sel, tgt_sel,
                                         img_idx, cls_idx, self.defect2idx[defect]))
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        break
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
            if self.max_samples and len(self.samples) >= self.max_samples:
                break

        print(f"[INFO] 샘플 {len(self.samples):,}개 | 클래스 {len(self.class2idx)} | 결함 {len(self.defect2idx)}")
        if self.cache_in_ram:
            print(f"[INFO] LRU: max_items={self._file_cache.max_items} max_bytes={self._file_cache.max_bytes}")

    # ────────────────────────────────────────────────
    # 2) .pt 로더 (GPU → CPU 강제 + 선택적 LRU 캐시)
    # ────────────────────────────────────────────────
    def _load_pt(self, path: pathlib.Path):
        """
        .pt 파일을 반드시 CPU Tensor로 로드.
        cache_in_ram=True이면 파일당 한 번만 읽고 LRU에 저장.
        """
        def to_cpu(storage, _):  # GPU storage를 CPU로 강제 매핑
            return storage.cpu()

        if not self.cache_in_ram:
            return torch.load(path, map_location=to_cpu)

        # LRU 존재 시 경로 기반 캐시 활용
        if path in self._file_cache:
            return self._file_cache.get(path)

        data = torch.load(path, map_location=to_cpu)
        if isinstance(data, list):
            data = [t.cpu() for t in data]
        elif torch.is_tensor(data):
            data = data.cpu()

        self._file_cache.put(path, data)  # 필요 시 자동 퇴출
        return data

    @staticmethod
    def _extract_defect_type(filename: str) -> str:
        """ scratch_0001.pt → scratch """
        return re.split(r"[_\-]", pathlib.Path(filename).stem)[0] or "unknown"

    # ────────────────────────────────────────────────
    # 3) 필수 메서드
    # ────────────────────────────────────────────────
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int):
        ok_p, ng_p, tgt_p, img_idx, cls_id, dft_id = self.samples[idx]

        ok_or_lat   = self._load_pt(ok_p)[img_idx]   # 텐서 또는 라티ent 텐서
        mask_or_cond= self._load_pt(ng_p)[img_idx]   # 마스크 또는 전처리 cond
        tgt_or_lat  = self._load_pt(tgt_p)[img_idx]  # 타겟 이미지 또는 라티ent

        sample = {
            "class_id":  torch.tensor(cls_id, dtype=torch.long),
            "defect_id": torch.tensor(dft_id, dtype=torch.long),
        }

        if self.use_latent:
            # 라티ent는 변환 적용하지 않음
            sample["ok_lat"] = ok_or_lat
            sample["ng_lat"] = tgt_or_lat
        else:
            ok   = ok_or_lat
            tgt  = tgt_or_lat
            # (선택) 변환 적용 — 이미지에만
            if self.transform_ok:
                ok = self.transform_ok(ok)
            if self.transform_tgt:
                tgt = self.transform_tgt(tgt)
            sample["ok"] = ok
            sample["ng_full"] = tgt

        if self.use_precond:
            # cond는 이미 [3,512,512] / [-1,1]로 저장되어 있다는 가정
            sample["cond"] = mask_or_cond
        else:
            mask = mask_or_cond
            if self.transform_mask:
                mask = self.transform_mask(mask)
            sample["ng_mask"] = mask

        # 텐서 참조 충돌 방지: 캐시 데이터의 in-place 오염 방지
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample.items()}

    # ────────────────────────────────────────────────
    # 4) 디버그 헬퍼
    # ────────────────────────────────────────────────
    def cache_stats(self):
        if not self.cache_in_ram:
            return {"enabled": False}
        c = self._file_cache
        return {
            "enabled": True,
            "items": len(c._store),
            "hits": c.hits,
            "misses": c.misses,
            "max_items": c.max_items,
            "max_bytes": c.max_bytes,
            "cur_bytes": getattr(c, "cur_bytes", None),
        }


