클래스, 결함 종류를 프롬프트로 바꿔가며 다양한 NG 이미지 합성 (데이터 증강 시나리오)
제조 도메인에 최적화된 제조 결함 이미지 생성 

SDv1.5 + ControlNet 기반 i2i pipeline

     
      root_dir/
        ├─ <class>/OK/*.pt
        ├─ <class>/NG/*.pt
        └─ <class>/Full_NG/*.pt
      각 .pt = list[Tensor]

    사전계산 사용 시(옵션)
      root_dir/
        ├─ <class>/OK_lat/*.pt         
        ├─ <class>/Full_NG_lat/*.pt   
        └─ <class>/NG_cond/*.pt       

* infer_lora.py 사용방법
  python infer_lora.py \
  --resume_lora /home/ubuntu/i2i/checkpoints_second/lora_step8000 \
  --classes 'audiojack,bottle_cap' \
  --mode img2img --strength 0.5 \
  --n 10 --steps 40 --scale 6.8 --cond_scale 1.6 --seed 123 \
  --out ./samples_img2img
