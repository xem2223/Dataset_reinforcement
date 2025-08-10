# train_second.py
# 이전 LoRA 체크포인트를 불러와서 새 6개 클래스를 중심으로 학습을 이어가되, 
# 이전 6개 클래스의 일부(리플레이)를 섞어 망각을 줄이는 스크립트
import os, argparse, torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from peft import PeftModel
from bitsandbytes.optim import AdamW8bit
from diffusers import (
    ControlNetModel, StableDiffusionControlNetPipeline,
    DDPMScheduler, AutoencoderKL
)
from Dataset import DefectSynthesisDataset  # latent/cond 전용 Dataset 사용
from utils import (
    set_seed, build_prompt_cache, get_text_embeds,
    parse_classes_arg, make_replay_dataset,
    steps_per_epoch, make_dataloader, try_load_old_classes_from_ckpt
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--resume_lora", type=str, required=True, help="기존 LoRA 체크포인트 디렉토리")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_second")
    parser.add_argument("--new_classes", type=str, default=None,help="쉼표구분 새 클래스(미지정 시 자동)")
    parser.add_argument("--old_classes", type=str, default=None,help="쉼표구분 기존 클래스(미지정 시 체크포인트/classes.txt에서 자동)")
    parser.add_argument("--replay_ratio", type=float, default=0.25)

    parser.add_argument("--vae_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--sd_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_id", type=str, default="lllyasviel/sd-controlnet-scribble")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=2000)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed) # 파이썬/토치 난수 고정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # -------- Dataset (latent/cond 전용) --------
    # DefectSynthesisDataset(args.root) 로 latent/cond 전용 데이터셋 전체를 로드
    ds_all = DefectSynthesisDataset(args.root, cache_in_ram=False)
    all_names = sorted(ds_all.class2idx.keys())
    idx2class = {v:k for k,v in ds_all.class2idx.items()}
    idx2defect = {v:k for k,v in ds_all.defect2idx.items()}
     
    # 새/옛 클래스명 리스트를 파싱
    new_names = parse_classes_arg(args.new_classes) if args.new_classes else None
    old_names = parse_classes_arg(args.old_classes) if args.old_classes else None
    
    if old_names is None:
        old_names = try_load_old_classes_from_ckpt(args.resume_lora)

    if new_names is None:
        # 새 클래스 = 전체 − (알고 있는 OLD)
        if old_names:
            new_names = [n for n in all_names if n not in set(old_names)]
        else:
            # OLD 정보를 못 찾으면 전체를 새로 본다(리플레이는 자연히 꺼짐)
            print("[WARN] old_classes를 찾지 못했습니다(체크포인트에 classes.txt 없음)")
            new_names = all_names
    
    # 유효성 체크
    if not new_names:
        raise ValueError("new_classes가 비었습니다. 경로/폴더명을 확인하세요.")
    print(f"[AUTO] new_classes: {new_names}")
    print(f"[AUTO] old_classes: {old_names or []}")
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "classes.txt"), "w", encoding="utf-8") as f:
            for n in sorted(set(new_names + (old_names or []))):
                f.write(n + "\n")
    except Exception as e:
        print(f"[WARN] classes.txt 저장 실패: {e}")
        
    # make_replay_dataset(...) : 새 클래스 샘플 전체 + 옛 클래스 일부(새 개수 × replay_ratio)를 랜덤 선택해 ConcatDataset으로 결합
    train_set = make_replay_dataset(ds_all, old_names or [], new_names, args.replay_ratio, args.seed)
   
    loader = make_dataloader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        timeout=120,
    )

    # -------- Text parts & prompt cache --------
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=torch.float16
    ).to(device).eval()
    # build_prompt_cache(...): (class_id, defect_id) 조합에 대해 고정 길이(77) 임베딩을 GPU에 미리 생성/저장.
    # 학습 루프에서 get_text_embeds(...)로 즉시 꺼내 쓰므로 토크나이저/텍스트 인코더 호출 0.
    prompt_cache = build_prompt_cache(idx2class, idx2defect, tokenizer, text_encoder, device)

    # -------- Models / Pipeline (resume LoRA) --------
    base_controlnet = ControlNetModel.from_pretrained(args.controlnet_id, torch_dtype=torch.float16)
    # 이전 LoRA 가중치를 베이스에 결합(이어 학습 준비)
    controlnet = PeftModel.from_pretrained(base_controlnet, args.resume_lora).to(device)
    controlnet.train()

    vae = AutoencoderKL.from_pretrained(args.vae_id, subfolder="vae", torch_dtype=torch.float16).to(device).eval()

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_id,
        controlnet=controlnet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
    ).to(device)
    
    # 메모리 최적화
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_attention_slicing()
    pipe.controlnet.enable_gradient_checkpointing()
    
    # 학습 파라미터는 ControlNet(LoRA)만 requires_grad=True; UNet/VAE/Text는 동결
    for p in pipe.unet.parameters():         p.requires_grad = False
    for p in pipe.vae.parameters():          p.requires_grad = False
    for p in pipe.text_encoder.parameters(): p.requires_grad = False
    for p in pipe.controlnet.parameters():   p.requires_grad = True

    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_id, subfolder="scheduler")
    # AdamW8bit(bitsandbytes)로 LoRA 파라미터만 최적화
    optimizer = AdamW8bit(
        filter(lambda p: p.requires_grad, pipe.controlnet.parameters()),
        lr=args.lr,
    )

    # -------- Train loop --------
    # 배치 키: ok_lat, ng_lat, cond, class_id, defect_id
    spe = steps_per_epoch(len(train_set), args.batch_size)
    print(f"[INFO] steps_per_epoch ≈ {spe}")
     
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            lat_in  = batch["ok_lat"].to(device, torch.float16, non_blocking=True)
            lat_tgt = batch["ng_lat"].to(device, torch.float16, non_blocking=True)
            cond    = batch["cond"].to(device,   torch.float16, non_blocking=True)
            class_ids  = batch["class_id"]
            defect_ids = batch["defect_id"]
            bsz = lat_in.size(0)

            noise     = torch.randn_like(lat_in)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            noisy_lat = noise_scheduler.add_noise(lat_in, noise, timesteps)

            text_emb = get_text_embeds(class_ids, defect_ids, prompt_cache)

            optimizer.zero_grad(set_to_none=True)

            ctrl_out = pipe.controlnet(
                sample=noisy_lat,
                timestep=timesteps,
                encoder_hidden_states=text_emb,
                controlnet_cond=cond,
                return_dict=True,
            )
            lat_pred = pipe.unet(
                sample=noisy_lat,
                timestep=timesteps,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=ctrl_out.down_block_res_samples,
                mid_block_additional_residual=ctrl_out.mid_block_res_sample,
                return_dict=True,
            ).sample

            loss = F.mse_loss(lat_pred.float(), lat_tgt.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"lora_step{global_step}")
                controlnet.save_pretrained(save_dir)
                pbar.set_postfix(loss=f"{loss.item():.4f}", save=save_dir)

        avg = total_loss / max(1, len(loader))
        print(f"Epoch {epoch} finished | avg loss: {avg:.4f}")
        controlnet.save_pretrained(os.path.join(args.output_dir, f"lora_epoch{epoch}"))

if __name__ == "__main__":
    main()
