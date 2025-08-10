클래스, 결함 종류를 프롬프트로 바꿔가며 다양한 NG 이미지 합성 (데이터 증강 시나리오)
제조 도메인에 최적화된 제조 결함 이미지 생성 

SDv1.5 + ControlNet 기반 i2i pipeline

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
