# infer_lora.py — LoRA merge + img2img check_inputs hotfix + tensor I/O
import os, argparse, random, traceback, types
import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)

# Dataset import (Dataset2가 없으면 Dataset로 폴백)
try:
    from Dataset2 import DefectSynthesisDataset  # latent/cond 전용
except Exception:
    from Dataset import DefectSynthesisDataset

# ───────────────────── Utils (torchvision 없이) ─────────────────────
def tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    """
    t: [3,H,W] or [1,3,H,W], float, 0..1 -> PIL RGB
    """
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().clamp(0, 1)
    arr = (t * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)

def cond_to_tensor01(cond: torch.Tensor) -> torch.Tensor:
    """
    cond: [3,512,512] in [-1,1] -> [1,3,512,512] float32 CPU (0..1)
    """
    t = cond.detach().to(dtype=torch.float32)      # CPU float32
    t = (t.clamp(-1, 1) + 1.0) / 2.0               # 0..1
    return t.unsqueeze(0)                          # [1,3,H,W]

def vae_latent_to_tensor01(vae: AutoencoderKL, lat4: torch.Tensor) -> torch.Tensor:
    """
    lat4: [4,64,64] or [1,4,64,64] -> decode -> [1,3,512,512] float32 CPU (0..1)
    """
    if lat4.dim() == 3:
        lat4 = lat4.unsqueeze(0)
    lat4 = lat4.to(device=vae.device, dtype=vae.dtype)
    with torch.no_grad():
        img = vae.decode(lat4 / vae.config.scaling_factor).sample  # [-1,1]
    img = (img.clamp(-1, 1) + 1.0) / 2.0                           # 0..1
    return img.detach().to(device="cpu", dtype=torch.float32)

# ───────────────────────────── Main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/opt/dlami/nvme")
    ap.add_argument("--resume_lora", type=str, required=True, help="LoRA 체크포인트 디렉토리")
    ap.add_argument("--out", type=str, default="./samples")

    ap.add_argument("--mode", type=str, default="img2img", choices=["img2img", "text2img"],
                    help="img2img(OK 고정) 또는 text2img(배경 자유)")
    ap.add_argument("--controlnet_id", type=str, default="lllyasviel/sd-controlnet-scribble")
    ap.add_argument("--sd_id", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--vae_id", type=str, default="runwayml/stable-diffusion-v1-5")

    ap.add_argument("--classes", type=str, default="", help="샘플링할 클래스 CSV(비우면 전체에서 랜덤)")
    ap.add_argument("--n", type=int, default=4, help="생성할 샘플 수")

    ap.add_argument("--steps", type=int, default=30, help="num_inference_steps")
    ap.add_argument("--scale", type=float, default=7.5, help="guidance_scale")
    ap.add_argument("--cond_scale", type=float, default=1.0, help="controlnet_conditioning_scale")
    ap.add_argument("--strength", type=float, default=0.3, help="img2img strength (0.2~0.4 권장)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.isdir(args.resume_lora):
        raise FileNotFoundError(f"--resume_lora not found: {args.resume_lora}")

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Dataset & picks
    ds = DefectSynthesisDataset(args.root, cache_in_ram=False)
    idx2class  = {v: k for k, v in ds.class2idx.items()}
    idx2defect = {v: k for k, v in ds.defect2idx.items()}

    want = set(s.strip() for s in args.classes.split(",") if s.strip()) if args.classes else None
    pool = []
    for i, tpl in enumerate(ds.samples):
        # (ok_lat_path, ng_cond_path, full_lat_path, img_idx, class_id, defect_id)
        class_id = tpl[4]
        cname = idx2class[class_id]
        if (want is None) or (cname in want):
            pool.append(i)
    if not pool:
        raise ValueError("선택된 클래스에서 샘플을 찾지 못했어요. --classes 이름/데이터 폴더 확인!")

    rng = random.Random(args.seed)
    picks = rng.sample(pool, k=min(args.n, len(pool)))

    # 2) Models — LoRA를 베이스 ControlNet에 병합(merge)해서 순수 ControlNetModel로 사용
    base_cn = ControlNetModel.from_pretrained(args.controlnet_id, torch_dtype=torch.float16)
    peft_cn = PeftModel.from_pretrained(base_cn, args.resume_lora)

    # merge_and_unload 우선 시도
    try:
        controlnet = peft_cn.merge_and_unload()
        print("[INFO] LoRA merged into ControlNet (merge_and_unload)")
    except Exception as e:
        print(f"[WARN] merge_and_unload failed: {e}")
        # 폴백: base model 추출
        controlnet = getattr(peft_cn, "base_model", peft_cn)
        # peft 구조에 따라 .model 안에 실제 diffusers 모듈이 있을 수 있음
        if hasattr(controlnet, "model"):
            controlnet = controlnet.model
        # 최종 방어선
        if not isinstance(controlnet, ControlNetModel):
            print("[WARN] Could not recover pure ControlNetModel. Falling back to base_cn.")
            controlnet = base_cn
    controlnet = controlnet.to(device).eval()

    vae = AutoencoderKL.from_pretrained(args.vae_id, subfolder="vae", torch_dtype=torch.float16).to(device).eval()

    if args.mode == "img2img":
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.sd_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        ).to(device)

        # --- 핫픽스: diffusers 0.25.x 의 img2img+ControlNet check_inputs 우회 ---
        def _noop_check_inputs(self, *a, **kw):  # 검증만 비활성화
            return None
        pipe.check_inputs = types.MethodType(_noop_check_inputs, pipe)
        print("[HOTFIX] check_inputs bypass enabled for img2img")
        # ----------------------------------------------------------------------

    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.sd_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        ).to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_attention_slicing()

    # 3) Loop
    for k, idx in enumerate(tqdm(picks, desc="Generating")):
        sample = ds[idx]
        cname = idx2class[int(sample["class_id"])]
        dname = idx2defect[int(sample["defect_id"])]
        prompt = f"{cname} {dname} defect, high quality, detailed"

        # ControlNet cond / OK init → CPU float32 [1,3,H,W] 0..1
        cond_t = cond_to_tensor01(sample["cond"])
        gen = torch.Generator().manual_seed(args.seed + k)  # CPU generator

        try:
            if args.mode == "img2img":
                ok_t = vae_latent_to_tensor01(vae, sample["ok_lat"])
                if ok_t.shape[-2:] != cond_t.shape[-2:]:
                    ok_t = torch.nn.functional.interpolate(
                        ok_t, size=cond_t.shape[-2:], mode="bilinear", align_corners=False
                    )

                out_imgs = pipe(
                    prompt=prompt,
                    image=ok_t,                               # CPU float32 [1,3,H,W]
                    control_image=cond_t,                     # CPU float32 [1,3,H,W]
                    strength=float(args.strength),
                    num_inference_steps=int(args.steps),
                    guidance_scale=float(args.scale),
                    controlnet_conditioning_scale=float(args.cond_scale),
                    generator=gen,
                ).images
            else:
                out_imgs = pipe(
                    prompt=prompt,
                    control_image=cond_t,                     # CPU float32 [1,3,H,W]
                    num_inference_steps=int(args.steps),
                    guidance_scale=float(args.scale),
                    controlnet_conditioning_scale=float(args.cond_scale),
                    generator=gen,
                    height=cond_t.shape[-2], width=cond_t.shape[-1],
                ).images
        except AssertionError:
            print("[DEBUG] call args:")
            print("  mode:", args.mode)
            print("  prompt_is_str:", isinstance(prompt, str))
            print("  cond_t:", cond_t.shape, cond_t.dtype, cond_t.min().item(), cond_t.max().item())
            if args.mode == "img2img":
                print("  ok_t:", ok_t.shape, ok_t.dtype, ok_t.min().item(), ok_t.max().item())
                print("  strength:", args.strength)
            print("  steps/scale/cond_scale:", args.steps, args.scale, args.cond_scale)
            traceback.print_exc()
            raise

        out = out_imgs[0]  # PIL
        stem = f"{k:02d}_{cname}_{dname}_seed{args.seed+k}"
        out.save(os.path.join(args.out, f"{stem}.png"))
        tensor01_to_pil(cond_t).save(os.path.join(args.out, f"{stem}_COND.png"))
        if args.mode == "img2img":
            tensor01_to_pil(ok_t).save(os.path.join(args.out, f"{stem}_OK.png"))

    print(f"[DONE] saved to: {args.out}")

if __name__ == "__main__":
    main()
