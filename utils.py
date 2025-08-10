# utils.py
import os, math, random
from typing import Dict, Tuple, List
import torch
from torch.utils.data import Subset, ConcatDataset

# --------------------------- seeds ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------- prompt cache -------------------------

@torch.no_grad()
def build_prompt_cache(
    idx2class: Dict[int, str],
    idx2defect: Dict[int, str],
    tokenizer,
    text_encoder,
    device: torch.device,
):
    """
    (class_id, defect_id) -> 고정 길이(보통 77) CLIP 텍스트 임베딩을 GPU에 미리 만들어 저장
    padding="max_length", truncation=True, max_length=tokenizer.model_max_length로 길이를 항상 동일하게 맞춰서, 
    이후 torch.cat 때 크기 불일치가 안 나게 함.
    반환: dict[(c_id, d_id)] = Tensor[1, MAX_LEN, hidden] (fp16 모델이면 fp16).
    """
    cache: Dict[Tuple[int,int], torch.Tensor] = {}
    MAX_LEN = tokenizer.model_max_length  # 보통 77
    text_encoder.eval()
    for c_id, c_name in idx2class.items():
        for d_id, d_name in idx2defect.items():
            ptxt = f"{c_name} {d_name} defect"
            enc = tokenizer(
                [ptxt],
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = text_encoder(**enc).last_hidden_state  # [1, MAX_LEN, hidden]
            cache[(c_id, d_id)] = emb
    return cache

"""
배치의 class_id, defect_id를 이용해 캐시에서 임베딩을 꺼내 배치 차원으로 concat
반환: (B, MAX_LEN, hidden)
"""
def get_text_embeds(class_ids: torch.Tensor, defect_ids: torch.Tensor, prompt_cache: dict):
    embs = [prompt_cache[(int(c.item()), int(d.item()))] for c, d in zip(class_ids, defect_ids)]
    return torch.cat(embs, dim=0)  # (B, MAX_LEN, hidden)

# --------------------- class selection -----------------------

"""
"a,b,c" 같은 문자열을 ["a","b","c"]로 파싱(공백 제거, 빈 토큰 제외).
CLI 인자 --new_classes, --old_classes 처리용.
"""
def parse_classes_arg(csv: str) -> List[str]:
    return [s.strip() for s in csv.split(",") if s.strip()]

"""
리플레이 샘플링 : 새 클래스 전부 + 구 클래스 일부(비율만큼)을 합쳐 ConcatDataset으로 반환
1. class2idx로 클래스명→id 찾아 새/구 클래스의 인덱스 목록 분리
2. 새 클래스 인덱스는 전부 사용
3. 구 클래스 인덱스에서 len(new) * replay_ratio 만큼 랜덤 샘플
4. ConcatDataset([new_subset, old_subset]) 반환 (replay_ratio<=0거나 구 세트가 없으면 새 세트만 사용)
"""
def make_replay_dataset(ds, old_names: List[str], new_names: List[str], replay_ratio: float, seed: int):
    idx2class = {v:k for k,v in ds.class2idx.items()}
    old_ids = {cid for cid, name in idx2class.items() if name in old_names}
    new_ids = {cid for cid, name in idx2class.items() if name in new_names}

    idx_old_all = [i for i, tpl in enumerate(ds.samples) if tpl[4] in old_ids]
    idx_new_all = [i for i, tpl in enumerate(ds.samples) if tpl[4] in new_ids]

    if not idx_new_all:
        raise ValueError("새 클래스(new_classes) 샘플이 0개입니다. 이름이 맞는지 확인하세요.")
    if not idx_old_all or replay_ratio <= 0:
        return Subset(ds, idx_new_all)

    replay_count = int(len(idx_new_all) * replay_ratio)
    rnd = random.Random(seed)
    rnd.shuffle(idx_old_all)
    idx_old_replay = idx_old_all[:replay_count]

    return ConcatDataset([Subset(ds, idx_new_all), Subset(ds, idx_old_replay)])

# --------------------- training helpers ----------------------

def steps_per_epoch(n_samples: int, batch_size: int) -> int:
    return math.ceil(n_samples / max(1, batch_size))

def make_dataloader(dataset_or_concat, batch_size: int, num_workers: int, prefetch_factor: int,
                    persistent_workers: bool, pin_memory: bool, timeout: int = 120):
    """ num_workers==0일 때 prefetch/persistent 옵션을 안전하게 건너뛰기 """
    from torch.utils.data import DataLoader
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        timeout=timeout,
    )
    if num_workers > 0:
        dl_kwargs.update(
            dict(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
        )
    else:
        dl_kwargs.update(dict(persistent_workers=False))  # 안전
    return DataLoader(dataset_or_concat, **dl_kwargs)
