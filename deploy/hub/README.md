# Native MOSS-ASR managed overlay

Preserve the Backup pinned HF/CUDA/FP32 implementation and shared NAS run history. HTTP and the serial queue stay in the CPU parent; the SDK engine alone imports Torch and original ModelRunner. Fresh actor worker ownership, callback-free plain-data RPC, native generation options, atomic metadata and filesystem edit locks remain. No automatic inference replay, CPU fallback or precision change.

Build only runtime through ./deploy.sh build --component runtime --work-dir <evidence> --launcher <registered launcher>. Committed read-only snapshot, pinned 0.3.0 SDK/base/frontend, shared native cache, identical repeat digest are required. Native UI/API plus real timestamped transcript, queue completion and drain/NVML evidence must pass before Backup/domain takeover. Credentials are server-only; prepare stopped candidate before initial registration.
