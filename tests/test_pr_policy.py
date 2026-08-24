import hashlib
import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _yaml_compat import yaml  # noqa: E402

from pr_policy import (  # noqa: E402
    SUPPORTED_EXACT_ARCHITECTURES,
    architecture_matches_filter,
    body_contract_errors,
    classify_scope,
    cuda_translation_unit_device_signal,
    derive_architectures,
    derive_metadata,
    device_code_pattern_sha256,
    parse_git_diff_files,
    python_dsl_pattern_sha256,
    render_generated_body,
    upstream_files_sha256,
)


def changed(path, patch="", status=""):
    return {"filename": path, "patch": patch, "status": status}


class ArchitecturePolicyTests(unittest.TestCase):
    def test_full_pull_diff_parser_preserves_tail_device_evidence(self):
        parsed = parse_git_diff_files(
            "diff --git a/src/host.cpp b/src/host.cpp\n"
            "index 111..222 100644\n--- a/src/host.cpp\n+++ b/src/host.cpp\n"
            "@@ -1 +1 @@\n-old();\n+host();\n"
            "diff --git a/src/kernel.h b/src/kernel.h\n"
            "new file mode 100644\nindex 000..333\n--- /dev/null\n+++ b/src/kernel.h\n"
            "@@ -0,0 +1 @@\n+CUTLASS_DEVICE void kernel();\n"
        )
        self.assertEqual(["src/host.cpp", "src/kernel.h"], [row["filename"] for row in parsed])
        self.assertEqual(["modified", "added"], [row["status"] for row in parsed])
        scope = classify_scope("Kernel update", "", parsed)
        self.assertTrue(scope.retain)
        self.assertEqual("device-code-signal", scope.rule)
        self.assertEqual(("src/kernel.h",), scope.evidence_paths)

    def test_no_evidence_is_visible_unknown_not_default(self):
        archs, disposition, evidence = derive_architectures(
            "Improve GEMM", "No hardware is named.", [changed("csrc/gemm.cu")]
        )
        self.assertEqual([], archs)
        self.assertEqual("unknown", disposition)
        self.assertEqual("unknown", evidence[0]["architecture"])

    def test_generic_blackwell_is_family_only(self):
        archs, disposition, _ = derive_architectures("Blackwell tuning", "", [])
        self.assertEqual(["blackwell"], archs)
        self.assertEqual("family", disposition)

    def test_generic_non_blackwell_families_are_family_only(self):
        for family in ("turing", "hopper", "ampere", "ada"):
            with self.subTest(family=family):
                archs, disposition, _ = derive_architectures(f"Tune for {family}", "", [])
                self.assertEqual([family], archs)
                self.assertEqual("family", disposition)

    def test_documented_b200_and_gb200_mapping_is_sm100(self):
        archs, disposition, evidence = derive_architectures(
            "B200 benchmark", "Also measured on GB200.", []
        )
        self.assertEqual(["sm100"], archs)
        self.assertEqual("exact", disposition)
        self.assertTrue(all(row.get("mapping_source", "").startswith("https://") for row in evidence))

    def test_product_multipliers_and_blackwell_ultra_mapping(self):
        archs, disposition, _ = derive_architectures(
            "Compare 8xH100, 1xB200, B300, and GB300", "", []
        )
        self.assertEqual(["sm90", "sm100", "sm103"], archs)
        self.assertEqual("exact", disposition)

    def test_camel_case_exact_architecture_is_detected(self):
        archs, disposition, _ = derive_architectures(
            "FMHA update", "", [changed("src/FmhaSm100aKernel.cuh")]
        )
        self.assertEqual(["sm100a"], archs)
        self.assertEqual("exact", disposition)

    def test_portable_sm80_atom_is_not_target_architecture(self):
        archs, disposition, _ = derive_architectures(
            "Hopper attention", "Use BatchPrefillWithPagedKVCacheSM90Run.",
            [changed("include/attention/hopper.cuh", "+using Copy = SM80_CP_ASYNC_CACHEALWAYS_ZFILL;")],
        )
        self.assertEqual(["sm90"], archs)
        self.assertEqual("exact", disposition)

    def test_compatibility_header_is_not_target_architecture(self):
        archs, disposition, _ = derive_architectures(
            "Blackwell attention", "", [changed(
                "include/attention/blackwell/sm100_kernel.cuh",
                '+#include "cutlass/arch/memory_sm80.h"\n+using Copy = SM80_CP_ASYNC_CACHEALWAYS<int>;',
            )],
        )
        self.assertEqual(["sm100"], archs)
        self.assertEqual("exact", disposition)

    def test_explicit_patch_guard_is_architecture_evidence(self):
        archs, disposition, _ = derive_architectures(
            "Dispatch kernel", "", [changed(
                "src/kernel.cu", "+#if defined(CUTE_ARCH_MMA_SM100_ENABLED)\n+run_kernel();\n+#endif",
            )],
        )
        self.assertEqual(["sm100"], archs)
        self.assertEqual("exact", disposition)

    def test_numeric_cuda_arch_guard_is_exact_evidence(self):
        archs, disposition, evidence = derive_architectures(
            "Dispatch kernel", "", [changed(
                "src/kernel.cu",
                "+#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)\n+run_kernel();\n+#endif",
            )],
        )
        self.assertEqual(["sm100"], archs)
        self.assertEqual("exact", disposition)
        self.assertIn("cuda-arch-guard", {row["basis"] for row in evidence})

    def test_ampere_sm87_and_sm88_are_supported_exact_targets(self):
        archs, disposition, _ = derive_architectures(
            "Add SM87 and SM88 kernels", "", []
        )
        self.assertEqual(["sm87", "sm88"], archs)
        self.assertEqual("exact", disposition)

    def test_architecture_vocabulary_matches_supported_ampere_spellings(self):
        architectures = set(
            yaml.safe_load((ROOT / "data" / "tags.yaml").read_text())["architectures"]
        )
        self.assertEqual(set(SUPPORTED_EXACT_ARCHITECTURES), architectures - {
            "blackwell", "hopper", "ampere", "ada", "turing"
        })

    def test_invalid_suffix_and_overlong_sm_tokens_are_not_exact_targets(self):
        for token in ("sm80a", "sm90f", "sm100x", "SM100X", "sm1000"):
            with self.subTest(token=token):
                archs, disposition, _ = derive_architectures(
                    f"Tune {token}", "", []
                )
                self.assertEqual([], archs)
                self.assertEqual("unknown", disposition)

    def test_contrastive_product_exception_is_not_a_target(self):
        archs, disposition, _ = derive_architectures(
            "CPU offloading",
            "This can be slow, unless in GH200 systems, where coherence can help.",
            [],
        )
        self.assertEqual([], archs)
        self.assertEqual("unknown", disposition)

    def test_underscore_helper_is_architecture_evidence(self):
        archs, _, _ = derive_architectures(
            "Dispatch kernel",
            "",
            [changed(
                "src/kernel.py",
                "+from utils import is_sm_100f\n+# Tuned on B200 for Blackwell\n+if is_sm_100f(): run_kernel()",
            )],
        )
        self.assertEqual(["sm100f"], archs)

    def test_target_product_comment_is_fallback_architecture_evidence(self):
        archs, _, _ = derive_architectures(
            "Transpose kernel",
            "",
            [changed(
                "src/kernel.py",
                "+# Tuned on B200 for this bandwidth-bound kernel\n+run_kernel()",
            )],
        )
        self.assertEqual(["sm100"], archs)

    def test_camel_case_sm_suffix_in_added_implementation_is_detected(self):
        archs, _, _ = derive_architectures(
            "Grouped GEMM",
            "",
            [changed(
                "src/grouped_gemm.cu",
                "+__global__ void grouped() { auto status = GroupGEMMSM100(args); }",
            )],
        )
        self.assertEqual(["sm100"], archs)

    def test_turing_target_is_not_replaced_by_rejected_sm80_lowering(self):
        archs, disposition, _ = derive_architectures(
            "Add native SM75 MMA support",
            "T.gemm could lower to SM80-only instructions on Turing GPUs.",
            [changed("src/mma_sm75.h", "+select_sm75_mma();")],
        )
        self.assertEqual(["sm75"], archs)
        self.assertEqual("exact", disposition)

    def test_range_boundary_is_not_reported_as_target_architecture(self):
        archs, disposition, _ = derive_architectures(
            "Fix compatibility dispatch",
            "The macro covers kernels less than sm80; this repairs compatibility with sm75.",
            [changed("src/sampling.cu", "+DISPATCH_COMPUTE_CAP_NUM_THREADS(...);")],
        )
        self.assertEqual(["sm75"], archs)
        self.assertEqual("exact", disposition)

    def test_unsupported_architecture_can_be_the_bugfix_target(self):
        archs, disposition, _ = derive_architectures(
            "Fix registration on unsupported architectures",
            "The extension fails to link on architectures like SM121.",
            [changed("csrc/registration.cu", "+TORCH_LIBRARY_IMPL(...);")],
        )
        self.assertEqual(["sm121"], archs)
        self.assertEqual("exact", disposition)

    def test_incidental_architecture_in_non_target_test_is_not_evidence(self):
        archs, disposition, _ = derive_architectures(
            "Fix parser",
            "",
            [
                changed("src/kernel.cu", "+__global__ void kernel() {}"),
                changed("tests/test_ci.py", '+gpu = "B200"'),
            ],
        )
        self.assertEqual([], archs)
        self.assertEqual("unknown", disposition)

    def test_commented_out_architecture_guard_is_not_target_evidence(self):
        archs, disposition, _ = derive_architectures(
            "Hopper attention",
            "Runs on H100.",
            [changed(
                "hopper/flash_api.cpp",
                "+// bool is_sm75 = props.major == 7 && props.minor == 5;\n"
                "+bool is_sm90 = props.major == 9 && props.minor == 0;",
            )],
        )
        self.assertEqual(["sm90"], archs)
        self.assertEqual("exact", disposition)

    def test_generated_ci_bot_architecture_text_is_ignored(self):
        body = """Blackwell kernel work.
<!-- This is an auto-generated comment: release notes by coderabbit.ai -->
CI example: run on H100
<!-- end of auto-generated comment: release notes by coderabbit.ai -->
## GitHub Bot Help
Try H200 if CI fails.
"""
        archs, disposition, _ = derive_architectures(
            "Blackwell kernel", body, [changed("src/blackwell/gemm.cu")]
        )
        self.assertEqual(["blackwell"], archs)
        self.assertEqual("family", disposition)

    def test_exact_and_different_family_are_mixed(self):
        archs, disposition, _ = derive_architectures("SM90 and Blackwell", "", [])
        self.assertEqual(["sm90", "blackwell"], archs)
        self.assertEqual("mixed", disposition)

    def test_hopper_products_do_not_append_sm100(self):
        archs, disposition, _ = derive_architectures("H100 and H200", "Hopper path", [])
        self.assertEqual(["sm90"], archs)
        self.assertEqual("exact", disposition)

    def test_client_sm120_path_does_not_append_sm100(self):
        archs, _, _ = derive_architectures("Blackwell client kernel", "", [changed("csrc/gemm_sm120.cu")])
        self.assertEqual(["sm120"], archs)

    def test_reported_pr_number_seed_is_resolved_by_actual_sglang_b200_evidence(self):
        archs, _, _ = derive_architectures(
            "Avoid conversions for DeepSeek-V3 on Blackwell",
            "Tested measurements on B200.",
            [changed("python/sglang/srt/models/deepseek_v2.py")],
        )
        self.assertEqual(["sm100"], archs)

    def test_reported_pr_number_seed_is_resolved_by_actual_flashinfer_sm120_path(self):
        archs, _, _ = derive_architectures(
            "MXFP4 group GEMM on GeForce", "", [changed("csrc/group_gemm_sm120_binding.cu")]
        )
        self.assertEqual(["sm120"], archs)

    def test_multiple_exact_architectures_are_preserved(self):
        archs, _, _ = derive_architectures(
            "SM90 and SM100 kernels", "", [changed("src/sm120/kernel.cu")]
        )
        self.assertEqual(["sm90", "sm100", "sm120"], archs)


class ScopeAndMetadataPolicyTests(unittest.TestCase):
    def test_plural_communication_kernel_title_is_distributed_scope(self):
        decision = classify_scope(
            "SM-constraint Communication Kernels",
            "Add cross-device all-reduce.",
            [changed("csrc/flashinfer_comm_ops.cu", "+__global__ void reduce() {}")],
        )
        self.assertFalse(decision.retain)
        self.assertEqual("distributed-system-implementation-exclusion", decision.rule)

    def test_modified_cuda_hunk_with_device_code_is_positive_kernel_evidence(self):
        decision = classify_scope(
            "Implement GEMM", "", [changed("csrc/gemm.cu", "+__global__ void gemm() {}", "modified")]
        )
        self.assertTrue(decision.retain)
        self.assertEqual("cuda-cute-ptx-device-source", decision.rule)

    def test_modified_host_only_cuda_translation_unit_is_not_kernel_evidence(self):
        files = [changed(
            "csrc/ops.cu",
            '+m.def("gemm", &gemm, "GEMM (CUDA)");',
            status="modified",
        )]
        self.assertFalse(classify_scope("Register GEMM", "", files).retain)

    def test_cuda_tile_shape_variables_alone_are_not_device_evidence(self):
        files = [changed(
            "csrc/launcher.cu",
            "+int64_t block_k = weights.sizes()[3];\n+int num_warps = 4;",
            status="modified",
        )]
        self.assertFalse(classify_scope("Tune launcher", "", files).retain)

    def test_cuda_kernel_launch_syntax_alone_is_not_device_evidence(self):
        files = [changed(
            "csrc/launcher.cu",
            "+gemm_kernel<<<grid, block, 0, stream>>>(input, output);",
            status="modified",
        )]
        self.assertFalse(classify_scope("Tune launcher", "", files).retain)

    def test_cuda_launch_config_fields_alone_are_not_device_evidence(self):
        text = "config.gridDim = grid;\nconfig.blockDim = block;\ncudaLaunchKernelEx(&config, fn);"
        self.assertFalse(cuda_translation_unit_device_signal(text))
        files = [changed("csrc/launcher.cu", "+" + text.replace("\n", "\n+"))]
        self.assertFalse(classify_scope("Tune launcher", "", files).retain)

    def test_cuda_builtins_require_device_style_member_access(self):
        self.assertTrue(cuda_translation_unit_device_signal("int i = blockIdx.x * blockDim.x + threadIdx.x;"))
        self.assertFalse(cuda_translation_unit_device_signal("// threadIdx.x\nconst char* s = \"__global__\";"))

    def test_cuda_host_template_type_alone_is_not_device_evidence(self):
        self.assertFalse(cuda_translation_unit_device_signal(
            "int main() { launch<GemmType::Normal>(); }"
        ))

    def test_complete_cuda_tile_shape_variables_are_not_device_evidence(self):
        files = [{
            **changed("csrc/launcher.cu", "+update_dispatch();", status="modified"),
            "complete_file_evidence_complete": True,
            "complete_file_sha256": "a" * 64,
            "complete_file_device_signal": False,
            "complete_file_device_pattern_sha256": device_code_pattern_sha256(),
        }]
        self.assertFalse(classify_scope("Tune launcher", "", files).retain)

    def test_modified_cuda_can_use_complete_file_device_evidence(self):
        files = [{
            **changed("csrc/gemm.cu", "+update_dispatch();", status="modified"),
            "complete_file_evidence_complete": True,
            "complete_file_sha256": "a" * 64,
            "complete_file_device_signal": True,
            "complete_file_device_pattern_sha256": device_code_pattern_sha256(),
        }]
        decision = classify_scope("Tune GEMM", "", files)
        self.assertTrue(decision.retain)
        self.assertEqual(("csrc/gemm.cu",), decision.evidence_paths)

    def test_stale_complete_file_signal_policy_is_not_positive_evidence(self):
        files = [{
            **changed("csrc/gemm.cu", "+update_dispatch();", status="modified"),
            "complete_file_evidence_complete": True,
            "complete_file_sha256": "a" * 64,
            "complete_file_device_signal": True,
            "complete_file_device_pattern_sha256": "b" * 64,
        }]
        self.assertFalse(classify_scope("Tune GEMM", "", files).retain)

    def test_new_host_only_cuda_translation_unit_is_not_kernel_evidence(self):
        files = [changed(
            "csrc/cuda_view.cu",
            "+torch::Tensor view(torch::Tensor x) {\n"
            "+  cudaHostGetDevicePointer(&ptr, x.data_ptr(), 0);\n"
            "+  return torch::from_blob(ptr, x.sizes());\n+}",
            status="added",
        )]
        self.assertFalse(classify_scope("Add UVA tensor view", "", files).retain)

    def test_new_cuda_translation_unit_with_device_kernel_is_retained(self):
        files = [changed(
            "csrc/gemm.cu", "+__global__ void gemm_kernel() {}", status="added"
        )]
        self.assertTrue(classify_scope("Add GEMM", "", files).retain)

    def test_tests_and_benchmarks_do_not_count(self):
        decision = classify_scope(
            "Benchmark GEMM", "", [changed("tests/test.cu"), changed("benchmarks/gemm.cu")]
        )
        self.assertFalse(decision.retain)

    def test_host_python_under_fused_moe_is_not_kernel_evidence(self):
        files = [changed("vllm/model_executor/layers/fused_moe/layer.py", "+def place_experts():\n+    pass")]
        decision = classify_scope("Static placement", "", files)
        metadata = derive_metadata("Static placement", "", files, decision)
        self.assertFalse(decision.retain)
        self.assertNotIn("kernel-fusion", metadata["techniques"])

    def test_python_requires_identifiable_dsl_kernel_content(self):
        files = [changed("kernels/softmax.py", "+@triton.jit\n+def softmax_kernel(x):\n+    pid = tl.program_id(0)")]
        decision = classify_scope("Triton softmax", "", files)
        self.assertTrue(decision.retain)
        self.assertEqual("python-dsl-device-kernel", decision.rule)
        self.assertIn("triton", derive_metadata("Triton softmax", "", files, decision)["languages"])

    def test_cpu_block_constants_are_not_gpu_device_evidence(self):
        files = [changed(
            "sgl-kernel/csrc/cpu/gemm_fp8.cpp",
            "+template <typename T, int BLOCK_M, int BLOCK_N> void gemm();",
        )]
        decision = classify_scope("CPU AMX GEMM", "", files)
        self.assertFalse(decision.retain)
        self.assertNotIn("cuda-cpp", derive_metadata("CPU AMX GEMM", "", files, decision)["languages"])

    def test_cpu_directory_is_out_of_gpu_scope_even_with_device_like_type(self):
        files = [changed(
            "sgl-kernel/csrc/cpu/gemm_fp8.cpp",
            "+using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmType>;",
        )]
        decision = classify_scope("CPU GEMM adapter", "", files)
        self.assertFalse(decision.retain)
        self.assertNotIn("cuda-cpp", derive_metadata("CPU GEMM adapter", "", files, decision)["languages"])

    def test_python_comment_with_block_name_is_not_kernel_evidence(self):
        files = [changed(
            "python/sglang/srt/configs/update_config.py",
            "+# Align block_n before loading weights.",
        )]
        self.assertFalse(classify_scope("Update configuration", "", files).retain)

    def test_host_python_block_arithmetic_rejects_negative_full_file_receipt(self):
        files = [{
            **changed("layers/quantization/fp8.py", "+num_blocks = (n + block_n - 1) // block_n"),
            "complete_file_python_dsl_evidence_complete": True,
            "complete_file_python_dsl_sha256": "a" * 64,
            "complete_file_python_dsl_signal": False,
            "complete_file_python_dsl_languages": [],
            "complete_file_python_dsl_pattern_sha256": python_dsl_pattern_sha256(),
        }]
        self.assertFalse(classify_scope("Update FP8 shapes", "", files).retain)

    def test_weak_python_tuning_can_use_complete_file_dsl_receipt(self):
        files = [{
            **changed("ops/mamba_ssm.py", "+num_warps = 8"),
            "complete_file_python_dsl_evidence_complete": True,
            "complete_file_python_dsl_sha256": "a" * 64,
            "complete_file_python_dsl_signal": True,
            "complete_file_python_dsl_languages": ["triton"],
            "complete_file_python_dsl_pattern_sha256": python_dsl_pattern_sha256(),
        }]
        decision = classify_scope("Tune Mamba kernel", "", files)
        self.assertTrue(decision.retain)
        self.assertEqual(["triton"], derive_metadata("Tune Mamba kernel", "", files, decision)["languages"])

    def test_stale_python_dsl_receipt_is_not_positive_evidence(self):
        files = [{
            **changed("ops/mamba_ssm.py", "+num_warps = 8"),
            "complete_file_python_dsl_evidence_complete": True,
            "complete_file_python_dsl_sha256": "a" * 64,
            "complete_file_python_dsl_signal": True,
            "complete_file_python_dsl_languages": ["triton"],
            "complete_file_python_dsl_pattern_sha256": "b" * 64,
        }]
        self.assertFalse(classify_scope("Tune Mamba kernel", "", files).retain)

    def test_upstream_hash_ignores_historical_receipt_not_required_by_policy(self):
        base = changed("csrc/kernel.cu", "+__global__ void kernel() {}", "modified")
        stale = {
            **base,
            "complete_file_evidence_complete": True,
            "complete_file_sha256": "a" * 64,
            "complete_file_device_signal": True,
            "complete_file_device_pattern_sha256": device_code_pattern_sha256(),
        }
        self.assertEqual(upstream_files_sha256([base]), upstream_files_sha256([stale]))

    def test_explicit_dsl_kernel_path_can_identify_modified_kernel_body(self):
        files = [changed(
            "python/cute_dsl_kernels/blackwell/top_k.py",
            "+griddepcontrol_wait()\n+griddepcontrol_launch_dependents()",
        )]
        decision = classify_scope("Enable PDL in top-k", "", files)
        self.assertTrue(decision.retain)
        self.assertEqual("python-dsl-device-kernel", decision.rule)

    def test_cutedsl_source_tree_identifies_kernel_implementation(self):
        files = [changed(
            "examples/python/CuTeDSL/blackwell/dense_gemm.py",
            "+acc_pipeline.consumer_release(state)",
        )]
        self.assertTrue(classify_scope("Fix accumulator overlap", "", files).retain)

    def test_generic_python_kernel_directory_is_not_enough(self):
        files = [changed("python/kernels/fused_moe.py", "+def configure():\n+    pass")]
        self.assertFalse(classify_scope("Update routing", "", files).retain)

    def test_named_distributed_exclusions_have_no_kernel_exception(self):
        for token in ("EPLB", "DeepEP", "DualPipe"):
            with self.subTest(token=token):
                decision = classify_scope(token + " CUDA kernel", "", [changed("csrc/real_kernel.cu")])
                self.assertFalse(decision.retain)
                self.assertEqual("hard-distributed-system-exclusion", decision.rule)

    def test_incidental_distributed_body_and_test_name_do_not_erase_kernel(self):
        files = [
            changed("src/gemm.cu", "+__global__ void gemm() {}"),
            changed("tests/test_deepep.cu", "+TEST(DeepEP, benchmark) {}"),
        ]
        decision = classify_scope(
            "Optimize local GEMM", "Benchmark command: --backend deepep", files
        )
        self.assertTrue(decision.retain)
        self.assertEqual(("src/gemm.cu",), decision.evidence_paths)

    def test_distributed_implementation_only_is_removed(self):
        for path in (
            "src/allReduceFusionKernels.cu",
            "csrc/all_reduce.cuh",
            "kernels/nvshmem_collective.cu",
            "src/communicationKernels.cu",
        ):
            with self.subTest(path=path):
                decision = classify_scope("Optimize kernels", "", [changed(path)])
                self.assertFalse(decision.retain)
                self.assertEqual("distributed-system-implementation-exclusion", decision.rule)

    def test_mixed_distributed_and_single_device_paths_retain_only_local_kernel(self):
        decision = classify_scope("Optimize kernels", "", [
            changed("src/allReduceFusionKernels.cu"),
            changed("src/layernorm.cu", "+__global__ void layernorm() {}"),
        ])
        self.assertTrue(decision.retain)
        self.assertEqual(("src/layernorm.cu",), decision.evidence_paths)

    def test_cutlass_collective_is_not_distributed_by_name_alone(self):
        decision = classify_scope(
            "Build GEMM collective", "", [changed("cutlass/gemm/collective/sm100_mma.cuh")]
        )
        self.assertTrue(decision.retain)

    def test_license_only_device_file_change_is_not_positive(self):
        patch = "+// SPDX-License-Identifier: Apache-2.0\n-// Copyright 2025\n+// Copyright 2026"
        self.assertFalse(classify_scope("Update license", "", [changed("src/gemm.cuh", patch)]).retain)

    def test_all_qualifying_paths_are_preserved(self):
        files = [
            changed(f"src/kernel_{i}.cu", "+__global__ void kernel() {}", "added")
            for i in range(12)
        ]
        decision = classify_scope("Add kernels", "", files)
        self.assertEqual(12, len(decision.evidence_paths))

    def test_metadata_uses_architecture_in_path_after_eighth_file(self):
        files = [
            changed(f"src/kernel_{i}.cu", "+__global__ void kernel() {}", "added")
            for i in range(9)
        ]
        files.append(changed(
            "src/sm103/tcgen05_kernel.cu", "+__global__ void kernel() {}", "added"
        ))
        decision = classify_scope("Add kernels", "", files)
        metadata = derive_metadata("Add kernels", "", files, decision)
        self.assertIn("tcgen05", metadata["hardware_features"])

    def test_bare_fused_path_does_not_create_fusion_tag(self):
        files = [changed("csrc/fused_moe/gemm.cu")]
        decision = classify_scope("Implement GEMM", "", files)
        metadata = derive_metadata("Implement GEMM", "", files, decision)
        self.assertNotIn("kernel-fusion", metadata["techniques"])

    def test_explicit_fused_kernel_semantics_create_fusion_tag(self):
        files = [changed("csrc/gemm.cu", "+__global__ void gemm() {}", "added")]
        decision = classify_scope("Add fused kernel for GEMM", "", files)
        metadata = derive_metadata("Add fused kernel for GEMM", "", files, decision)
        self.assertIn("kernel-fusion", metadata["techniques"])

    def test_negative_compatibility_inventory_is_not_positive_metadata(self):
        files = [changed("csrc/cpu_offload.cuh")]
        decision = classify_scope("CPU offloading", "", files)
        metadata = derive_metadata(
            "CPU offloading",
            "The implementation is not compatible with quantization, LoRA, MLA, etc.",
            files,
            decision,
        )
        self.assertNotIn("quantization", metadata["kernel_types"])
        self.assertNotIn("mla", metadata["kernel_types"])

    def test_linear_scan_comment_is_not_parallel_scan_metadata(self):
        files = [changed(
            "csrc/moe.cu",
            "+// subsequent tokens use linear scan because experts are sorted\n"
            "+expert = findTotalEltsLessThanTarget(token);",
        )]
        decision = classify_scope("Optimize MoE", "", files)
        metadata = derive_metadata("Optimize MoE", "", files, decision)
        self.assertNotIn("scan", metadata["kernel_types"])

    def test_block_scan_implementation_is_scan_metadata(self):
        files = [changed(
            "csrc/moe.cu",
            "+__global__ void scan() { using Scan = cub::BlockScan<int, 128>; }",
            "added",
        )]
        decision = classify_scope("Optimize MoE", "", files)
        metadata = derive_metadata("Optimize MoE", "", files, decision)
        self.assertIn("scan", metadata["kernel_types"])


class GeneratedBodyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "generate_pr_pages", SCRIPTS / "generate-pr-pages.py"
        )
        cls.generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.generator)

    def test_template_contains_only_upstream_excerpt_and_fixed_material(self):
        payload = {
            "number": 7,
            "title": "A title that must not become a Problem section",
            "user": {"login": "author"},
            "created_at": "2026-01-02T03:04:05Z",
            "html_url": "https://github.com/example/project/pull/7",
            "merge_commit_sha": "a" * 40,
            "body": "Upstream sentence with 12.5 GB/s exactly.",
        }
        files = [changed("csrc/kernel_sm120.cu", "+__global__ void kernel() {}")]
        rendered = self.generator.generate_page(
            "example/project", payload, files, "ignored legacy reason", "2026-08-18"
        )
        match = re.match(r"^---\n(.*?)\n---\n\n(.*)$", rendered, re.DOTALL)
        self.assertIsNotNone(match)
        frontmatter = yaml.safe_load(match.group(1))
        body = match.group(2)
        self.assertIn(payload["body"], body)
        self.assertNotIn(payload["title"], body)
        self.assertNotIn("## Problem", body)
        self.assertNotIn("near-peak", body)
        self.assertEqual([], body_contract_errors(frontmatter, body))
        self.assertEqual(
            hashlib.sha256(payload["body"].encode()).hexdigest(),
            frontmatter["upstream_body_sha256"],
        )

    def test_crlf_body_digest_matches_text_reader_normalization(self):
        payload = {
            "number": 8,
            "title": "CRLF kernel body",
            "user": {"login": "author"},
            "created_at": "2026-01-02T03:04:05Z",
            "html_url": "https://github.com/example/project/pull/8",
            "merge_commit_sha": "b" * 40,
            "body": "First line.\r\n\r\nSecond line.",
        }
        rendered = self.generator.generate_page(
            "example/project",
            payload,
            [changed("csrc/kernel.cu", "+__global__ void kernel() {}")],
            "ignored legacy reason",
            "2026-08-18",
        )
        normalized = rendered.replace("\r\n", "\n")
        match = re.match(r"^---\n(.*?)\n---\n\n(.*)$", normalized, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertEqual([], body_contract_errors(yaml.safe_load(match.group(1)), match.group(2)))

    def test_authoritative_changed_file_total_is_not_replaced_by_api_listing_size(self):
        payload = {
            "number": 9,
            "title": "Large kernel update",
            "user": {"login": "author"},
            "created_at": "2026-01-02T03:04:05Z",
            "html_url": "https://github.com/example/project/pull/9",
            "merge_commit_sha": "c" * 40,
            "changed_files": 3750,
            "body": "Upstream body.",
        }
        rendered = self.generator.generate_page(
            "example/project",
            payload,
            [changed("csrc/kernel.cu", "+__global__ void kernel() {}")],
            "ignored legacy reason",
            "2026-08-18",
        )
        match = re.match(r"^---\n(.*?)\n---\n\n(.*)$", rendered, re.DOTALL)
        frontmatter = yaml.safe_load(match.group(1))
        self.assertEqual(3750, frontmatter["changed_files_count"])
        self.assertEqual(1, frontmatter["changed_files_enumerated_count"])
        self.assertFalse(frontmatter["changed_files_listing_complete"])
        self.assertFalse(frontmatter["changed_paths_complete"])
        self.assertIn("3749 additional changed file(s)", match.group(2))

    def test_truncation_on_whitespace_is_idempotent(self):
        upstream = "x" * (1200 - 5) + " tail " + "more"
        body, digest = render_generated_body(upstream, ["csrc/kernel.cu"], 1)
        frontmatter = {
            "body_contract": "upstream-pr-v1",
            "upstream_excerpt_sha256": digest,
            "upstream_body_sha256": "a" * 64,
            "upstream_files_sha256": "b" * 64,
            "changed_paths": ["csrc/kernel.cu"],
            "changed_files_count": 1,
        }
        self.assertEqual([], body_contract_errors(frontmatter, body))

    def test_upstream_trailing_whitespace_is_canonicalized(self):
        body, _ = render_generated_body(
            "First line.  \nSecond line.\t\n", ["csrc/kernel.cu"], 1
        )
        self.assertNotIn("  \n", body)
        self.assertNotIn("\t\n", body)

    def test_upstream_test_output_cannot_look_like_a_conflict_marker(self):
        body, _ = render_generated_body(
            "```\n======= 4 passed in 1.66s =======\n```", ["test_kernel.py"], 1
        )
        self.assertIn("\\======= 4 passed", body)
        self.assertNotIn("\n======= 4 passed", body)

    def test_contract_rejects_injected_generated_prose(self):
        body, digest = render_generated_body("Upstream only.", ["csrc/kernel.cu"], 1)
        frontmatter = {
            "body_contract": "upstream-pr-v1",
            "upstream_excerpt_sha256": digest,
            "upstream_body_sha256": "a" * 64,
            "upstream_files_sha256": "b" * 64,
            "changed_paths": ["csrc/kernel.cu"],
            "changed_files_count": 1,
        }
        injected = body + "\n## Performance\n\nInvented 900 TFLOPS.\n"
        self.assertTrue(body_contract_errors(frontmatter, injected))


class ArchitectureQuerySemanticsTests(unittest.TestCase):
    def test_exact_family_and_unknown_sets_are_disjoint_and_hierarchical(self):
        rows = [
            (["sm100"], "exact"),
            (["sm100a"], "exact"),
            (["sm120"], "exact"),
            (["blackwell"], "family"),
            (["sm90"], "exact"),
            ([], "unknown"),
        ]
        sm100 = {i for i, row in enumerate(rows) if architecture_matches_filter(*row, "sm100")}
        blackwell = {i for i, row in enumerate(rows) if architecture_matches_filter(*row, "blackwell")}
        unknown = {i for i, row in enumerate(rows) if architecture_matches_filter(*row, "unknown")}
        self.assertEqual({0}, sm100)
        self.assertEqual({0, 1, 2, 3}, blackwell)
        self.assertEqual({5}, unknown)
        self.assertTrue(sm100 <= blackwell)

    def test_documented_product_alias_is_exact(self):
        self.assertTrue(architecture_matches_filter(["sm100"], "exact", "B200"))
        self.assertFalse(architecture_matches_filter(["blackwell"], "family", "B200"))


if __name__ == "__main__":
    unittest.main()
