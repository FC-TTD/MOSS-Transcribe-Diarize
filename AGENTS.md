# MOSS 歌词识别 Hub 适配

- Canonical 为 FC-TTD/MOSS-Transcribe-Diarize 的 hub 分支；用户2026-10-27明确允许本分支提交推送、独立Runtime机器凭据与真实验收后Backup→Edge接管。
- 原生实现以deploy/native/provenance.json的实际镜像/挂载源码为准。保留原HF、CUDA、FP32、原始生成参数与时间戳字幕/HTML/文件合同，不改为MOSS-TTS，不自动CPU/offload/量化或重放任务。
- HTTP父进程只做文件与队列，Torch/权重仅SDK指定GPU引擎内。原始单工作队列及共享NAS worker_id/文件锁保持；新actor不得将旧Backup的在途任务标成中断。
- 控制与推理身份独立，凭据仅服务器私密文件；任何unknown关闭准入并保留输入、历史和证据，不手改Hub库/journal。
- 通过真实转写、原API/UI、活动/队列和drain/NVML验收后才接管原域名。旧服务排空前不停止；中文UI不小于12px。
