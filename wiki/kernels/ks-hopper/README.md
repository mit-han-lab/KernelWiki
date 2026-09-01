# NVIDIA Hopper 架构 Kernel 性能优化 Skills

本目录用于存放面向 **NVIDIA Hopper 架构 GPU** 的 Kernel 性能优化 Skill。这些 Skill 服务于 Hopper 架构芯片上的高性能计算、AI 推理和训练任务，旨在沉淀可复用的分析方法、优化流程、架构知识、工具用法和验证规范，帮助代码智能体稳定完成 Hopper Kernel 性能优化。

其他 NVIDIA GPU 架构及 AMD、昇腾等非 Hopper 平台的专用优化 Skill 不应存放在本目录中。若某项方法具有跨平台通用性，也应围绕 Hopper 架构的实际约束、实现和验证方式编写。

## 已收录 Skills

- [`profile-and-optimize-vllm-kernels`](profile-and-optimize-vllm-kernels/SKILL.md)：从真实 vLLM E2E、mapping/formal trace 和 leaf kernel 占比出发，完成 Hopper kernel 热点选择、源码优化、精度门禁、镜像替换及交错 E2E 验收；包含 FlashMLA split/combine 案例。

## 目标

- 将 NVIDIA Hopper 架构 Kernel 优化经验整理为可执行、可复用的 Skill。
- 覆盖从性能分析、瓶颈定位到实现、验证和回归测试的完整流程。
- 统一 Hopper 架构上不同编程模型和算子类型的 Skill 文档结构。
- 沉淀 Hopper 架构的关键硬件约束、常见陷阱和性能判断依据，减少重复试错。

## 建议覆盖范围

本目录中的 Skill 可以包括但不限于：

- 面向 Hopper 架构的 CUDA、Triton、CUTLASS、C/C++ Kernel 开发与优化。
- 访存合并、数据布局、共享内存、缓存和流水线优化。
- 向量化、并行划分、线程块配置及负载均衡。
- Tensor Core、异步拷贝、矩阵指令及 Hopper 架构特性的使用。
- Attention、GEMM、MoE、归一化、归约等典型算子优化。
- HBM 访存、计算吞吐、Occupancy、指令流水线和同步开销分析。
- Nsight Compute、Nsight Systems、性能计数器等 NVIDIA 分析工具的使用。
- 数值精度、正确性验证、性能基准和回归测试。

## Skill 编写规范

每个 `SKILL` 建议包含以下内容：

1. **名称与简介**：说明 Skill 解决的问题及适用场景。
2. **触发条件**：明确智能体应在什么任务中使用该 Skill。
3. **前置条件**：列出 Hopper GPU 型号、CUDA 环境、软件版本、工具链和输入材料。
4. **工作流程**：提供可执行的分析、修改、测试和迭代步骤。
5. **优化原则**：记录 Hopper 架构的关键性能指标、判断依据及平台约束。
6. **验证要求**：同时定义正确性、数值精度和性能验证方式。
7. **常见问题**：说明高频错误、无效优化和排查方法。
8. **参考资料**：链接到 `references/` 中按需读取的详细资料。

## 优化任务的基本原则

- **先测量，再优化**：先建立可靠基线并确认瓶颈，避免凭直觉修改。
- **正确性优先**：性能结果必须建立在正确性和数值精度满足要求的基础上。
- **控制变量**：尽量一次验证一个核心优化点，保留优化前后的数据。
- **只以 Hopper 架构为目标平台**：优化设计、性能结论和验收结果均以 NVIDIA Hopper 架构 GPU 为准。
- **关注真实负载**：测试应覆盖目标 Hopper GPU 实际使用中的典型形状、数据类型和运行环境。
- **结果可复现**：记录 Hopper GPU 型号及配置、驱动和 CUDA 版本、软件版本、编译参数、输入规模及计时方法。
- **明确特化边界**：如需针对特定形状、数据类型或并行配置特化，应明确适用范围并保留合理的回退路径。

## 维护约定

- 本目录只接收以 NVIDIA Hopper 架构 GPU 为目标硬件的 Kernel 性能优化 Skill。
- 优先将可在 Hopper 架构芯片上复用的方法写入 Skill，将特定任务的实验结果保留在对应项目中。
- 修改优化规则时同步更新验证方法，避免文档与实际工具链脱节。
- 引入新工具链时，明确版本范围以及与 Hopper 架构 GPU 的兼容性要求。
- 定期清理失效链接、过时参数和已被替代的优化建议。
