#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성능 테스트 스크립트 - 최적화된 embed_features.py 성능 측정
"""
import os
import time
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import yaml

# 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from embed_features import build_adapter, device_autoselect, read_processed_meta, pil_open_rgb

def simple_performance_test():
    """간단한 성능 테스트"""
    print("=" * 50)
    print("성능 테스트 시작")
    print("=" * 50)
    
    # 설정 로드
    config_path = Path("config.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    
    # 디바이스 설정
    device = device_autoselect(cfg["base"].get("device", "cpu"))
    print(f"사용 디바이스: {device}")
    
    # GPU 정보 출력
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 메모리: {gpu_memory:.1f}GB")
    
    # 어댑터 생성
    print("\n어댑터 로딩 중...")
    adapter = build_adapter(cfg, device)
    print(f"어댑터: {cfg['embedding']['model_source']}, 차원: {adapter.feature_dim()}")
    
    # 메타 데이터 로드
    processed_dir = Path("../processed")
    meta_df = read_processed_meta(processed_dir)
    
    # 테스트용 샘플 선택 (처음 50개)
    test_samples = meta_df.head(50)
    valid_samples = []
    
    print(f"\n유효한 이미지 경로 확인 중...")
    for _, row in test_samples.iterrows():
        if os.path.isfile(row["crop_path"]):
            valid_samples.append(row)
    
    print(f"유효한 샘플 수: {len(valid_samples)}")
    
    if len(valid_samples) == 0:
        print("유효한 샘플이 없습니다.")
        return
    
    # 성능 테스트 - 배치 크기별
    batch_sizes = [1, 4, 8, 16, 32]
    results = []
    
    for batch_size in batch_sizes:
        if batch_size > len(valid_samples):
            continue
            
        print(f"\n배치 크기 {batch_size} 테스트...")
        
        # 테스트 이미지들 로드
        test_images = []
        for i in range(min(batch_size, len(valid_samples))):
            img_path = valid_samples[i]["crop_path"]
            try:
                img = pil_open_rgb(img_path)
                test_images.append(img)
            except Exception as e:
                print(f"이미지 로딩 실패: {img_path}, {e}")
        
        if not test_images:
            continue
        
        # 워밍업 (첫 실행에만)
        if batch_size == batch_sizes[0]:
            print("GPU 워밍업 중...")
            inputs = adapter.preprocess([test_images[0]])
            _ = adapter.forward_features(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        # 성능 측정 (여러 번 실행하여 평균)
        times = []
        for run in range(3):  # 3회 실행
            start_time = time.time()
            
            inputs = adapter.preprocess(test_images)
            features = adapter.forward_features(inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # GPU 작업 완료 대기
                
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(test_images) / avg_time
        
        result = {
            'batch_size': batch_size,
            'avg_time': avg_time,
            'std_time': std_time,
            'throughput': throughput,
            'images_processed': len(test_images)
        }
        results.append(result)
        
        print(f"  평균 시간: {avg_time:.3f}±{std_time:.3f}초")
        print(f"  처리량: {throughput:.1f} 이미지/초")
        print(f"  특징 크기: {features.shape}")
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("성능 테스트 결과 요약")
    print("=" * 50)
    print(f"{'배치크기':<8} {'평균시간(초)':<12} {'처리량(img/s)':<15} {'특징크기'}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['batch_size']:<8} {result['avg_time']:<12.3f} {result['throughput']:<15.1f} ({adapter.feature_dim()},)")
    
    # 최적 배치 크기 찾기
    best_result = max(results, key=lambda x: x['throughput'])
    print(f"\n최적 배치 크기: {best_result['batch_size']} (처리량: {best_result['throughput']:.1f} 이미지/초)")
    
    return results

def memory_usage_test():
    """메모리 사용량 테스트"""
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없어 메모리 테스트를 건너뜁니다.")
        return
    
    print("\n" + "=" * 30)
    print("GPU 메모리 사용량 테스트")
    print("=" * 30)
    
    # 초기 메모리 사용량
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"초기 메모리: {initial_memory:.1f}MB")
    
    # 설정 로드 및 모델 로드
    config_path = Path("config.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    
    device = "cuda:0"
    adapter = build_adapter(cfg, device)
    
    after_model_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"모델 로드 후: {after_model_memory:.1f}MB (증가: {after_model_memory - initial_memory:.1f}MB)")
    
    # 배치 크기별 메모리 사용량
    batch_sizes = [1, 4, 8, 16, 32]
    dummy_img = Image.new('RGB', (224, 224), color='black')
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        before_batch = torch.cuda.memory_allocated() / 1024**2
        
        # 배치 처리
        test_images = [dummy_img] * batch_size
        inputs = adapter.preprocess(test_images)
        _ = adapter.forward_features(inputs)
        
        after_batch = torch.cuda.memory_allocated() / 1024**2
        batch_memory = after_batch - before_batch
        
        print(f"배치 크기 {batch_size:2d}: {batch_memory:6.1f}MB 사용")

if __name__ == "__main__":
    try:
        results = simple_performance_test()
        memory_usage_test()
        
        print("\n테스트 완료!")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()