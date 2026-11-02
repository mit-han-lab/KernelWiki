import tempfile
import unittest
from pathlib import Path
from repo_import_support import ROOT, CODE
from kernel_entries import python_kernel_entries, cuda_kernel_entries


class EntryTests(unittest.TestCase):
    def test_cuda_braced_defaults_keep_real_name_and_body(self):
        source = ('__global__ __launch_bounds__(128) void compute(Params p = Params(), '
                  'Scale s = {}, int n = nested(1, call(2))) { use(p, s, n); }\n'
                  '__global__ void prototype(Params p = Params{1});\n')
        entries = cuda_kernel_entries(source)
        self.assertEqual(['compute'], [e.name for e in entries])
        changed = cuda_kernel_entries(source.replace('use(p, s, n)', 'use_new(p, s, n)'))
        self.assertNotEqual(entries[0].body_sha256, changed[0].body_sha256)

    def test_cuda_alias_braced_defaults_and_nested_constructors(self):
        source = ('#define GPU_ENTRY __global__ __launch_bounds__(128)\n'
                  'GPU_ENTRY void compute(Params p = Params{nested(1, 2)}) { use(p); }\n'
                  'GPU_ENTRY void prototype(Params p = {});\n')
        entries = cuda_kernel_entries(source)
        self.assertEqual(['compute'], [e.name for e in entries])
        self.assertEqual(2, entries[0].line)

    def test_cuda_default_initializer_snippet_keeps_only_selected_complete_body(self):
        from kernel_entries import kernel_snippet
        source = ('template<typename Alpha>\n'
                  '__global__ void compute(Alpha alpha = Alpha{1}) { use(alpha); }\n'
                  '__global__ void unreviewed() { other(); }\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.cu'
            path.write_text(source)
            entry = cuda_kernel_entries(source)[0]
            self.assertEqual(source[:source.index('__global__ void unreviewed')],
                             kernel_snippet(path, entry.name, entry.line))
            with self.assertRaises(ValueError):
                kernel_snippet(path, 'wrong_name', entry.line)
            with self.assertRaises(ValueError):
                kernel_snippet(path, entry.name, 1)

    def test_cuda_conditional_signatures_and_unknown_launch_macros_retain_coverage(self):
        source = '''__global__ void
#if defined(FAST)
__launch_bounds__(512, 4) quantize(
#else
quantize(
#endif
int* x) { x[0] = 1; }
__global__ CUSTOM_LAUNCH_BOUNDS(128) void sample(int* x) { x[0] = 2; }
'''
        entries = cuda_kernel_entries(source)
        self.assertEqual(['quantize', 'sample'], [e.name for e in entries])
        from kernel_entries import kernel_snippet
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.cu'
            path.write_text(source)
            self.assertEqual(source[:source.index('__global__ CUSTOM')],
                             kernel_snippet(path, entries[0].name, entries[0].line))

    def test_python_multiple_entries_and_comment(self):
        entries = python_kernel_entries(CODE + '\n# @triton.jit\n@cute.kernel\ndef second(x):\n    return x\n')
        self.assertEqual(['reduce_kernel', 'second'], [e.name for e in entries])

    def test_invalid_python_is_not_a_silent_success(self):
        from repo_import_support import prepared, commit, import_repo, load_config, read
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            (repo / 'broken.py').write_text('def invalid(:')
            commit(repo)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            self.assertEqual('broken.py', read(run / 'inventory.yaml')['failures'][0]['path'])

    def test_cuda_definition_not_declaration_or_comment(self):
        entries = cuda_kernel_entries('// __global__ void fake() {}\n__global__ void declared();\n__global__ void real(int* x) { x[0] = 1; }')
        self.assertEqual(['real'], [e.name for e in entries])

    def test_cutlass_global_definitions_launch_bounds_and_exact_snippets(self):
        from kernel_entries import kernel_snippet
        source = ('#define CUTLASS_GLOBAL __global__ static\n'
                  '// CUTLASS_GLOBAL void fake() {}\n'
                  'CUTLASS_GLOBAL void declared();\n'
                  'template<int N>\nCUTLASS_GLOBAL void __launch_bounds__(N, 1)\n'
                  'first(int* x) { x[0] = N; }\n'
                  'CUTLASS_GLOBAL __launch_bounds__(128, 1) void second(int* x) { x[1] = 2; }\n')
        entries = cuda_kernel_entries(source)
        self.assertEqual(['first', 'second'], [entry.name for entry in entries])
        self.assertEqual([5, 7], [entry.line for entry in entries])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.cuh'
            path.write_text(source)
            snippet = kernel_snippet(path, entries[0].name, entries[0].line)
            self.assertIn('template<int N>', snippet)
            self.assertIn('x[0] = N', snippet)
            self.assertNotIn('second', snippet)
            self.assertNotIn('#define', snippet)

    def test_global_qualifier_alias_is_not_a_host_function_definition(self):
        source = '#define CUTLASS_GLOBAL __global__\ninline void host_only(int* x) { x[0] = 1; }\n'
        self.assertEqual([], cuda_kernel_entries(source))

    def test_cuda_qualifier_macro_binds_functions_not_attributes_or_intervening_types(self):
        source = ('#define GPU_ENTRY __global__ __launch_bounds__(Trait::threads, 1) inline\n'
                  'struct Params { int count; };\n'
                  'enum class Mode { one, two };\n'
                  'inline int helper(int x) { return x + 1; }\n'
                  'template<typename Trait>\n'
                  'GPU_ENTRY void first(Params params) { compute(params); }\n'
                  'GPU_ENTRY void prototype(Params params);\n'
                  'GPU_ENTRY void second(Params params) { other(params); }\n')
        entries = cuda_kernel_entries(source)
        self.assertEqual(['first', 'second'], [entry.name for entry in entries])
        self.assertEqual([6, 8], [entry.line for entry in entries])
        changed = cuda_kernel_entries(source.replace('compute(params)', 'compute_new(params)'))
        self.assertNotEqual(entries[0].body_sha256, changed[0].body_sha256)
        self.assertEqual(entries[1].body_sha256, changed[1].body_sha256)

    def test_cuda_unused_or_unknown_qualifier_macros_never_capture_host_helpers(self):
        source = ('#define UNUSED __global__ __launch_bounds__(128, 2)\n'
                  '#define UNKNOWN __global__ unsupported_qualifier(128)\n'
                  'struct alignas(16) Params { int value; };\n'
                  'inline void host_only() { work(); }\n')
        self.assertEqual([], cuda_kernel_entries(source))

    def test_cuda_qualifier_alias_lifetime_and_balanced_signature(self):
        source = ('#define GPU_ENTRY __global__ __cluster_dims__(2, 1, 1)\n'
                  'GPU_ENTRY void first(int value = nested(1, call(2))) { use(value); }\n'
                  '#undef GPU_ENTRY\n'
                  'GPU_ENTRY void ignored() { unrelated(); }\n'
                  '#define GPU_ENTRY __global__ __launch_bounds__(64)\n'
                  'GPU_ENTRY void second(int value) __attribute__((unused)) { use(value + 1); }\n')
        entries = cuda_kernel_entries(source)
        self.assertEqual(['first', 'second'], [entry.name for entry in entries])
        self.assertEqual([2, 6], [entry.line for entry in entries])

    def test_cuda_qualifier_alias_snippet_includes_definition_and_only_selected_body(self):
        from kernel_entries import kernel_snippet
        definition = '#define GPU_ENTRY __global__ __launch_bounds__(128, 2)\n'
        selected = 'template<int N>\nGPU_ENTRY void selected(int* x) { x[0] = N; }\n'
        source = definition + 'struct Params { int count; };\n' + selected + 'GPU_ENTRY void other(int* x) { x[1] = 2; }\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.cuh'
            path.write_text(source)
            entry = cuda_kernel_entries(source)[0]
            self.assertEqual(definition + '\n' + selected, kernel_snippet(path, entry.name, entry.line))

    def test_selected_snippet_never_publishes_unreviewed_definition(self):
        from kernel_entries import kernel_snippet
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.py'
            path.write_text(CODE + '\n@triton.jit\ndef unreviewed(x):\n    return x * 2\n')
            entry = python_kernel_entries(path.read_text())[0]
            snippet = kernel_snippet(path, entry.name, entry.line)
            self.assertIn('reduce_kernel', snippet)
            self.assertNotIn('unreviewed', snippet)

    def test_cuda_strings_cannot_create_entries_or_close_bodies(self):
        from kernel_entries import kernel_snippet
        source = 'const char* fake = "__global__ void fake() {}";\n__global__ void real(int* x) { const char* s = "}"; x[0] = 1; }\n'
        entries = cuda_kernel_entries(source)
        self.assertEqual(['real'], [e.name for e in entries])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.cu'
            path.write_text(source)
            self.assertIn('x[0] = 1;', kernel_snippet(path, entries[0].name, entries[0].line))

    def test_cuda_conditional_braces_keep_all_arms_without_publishing_other_entries(self):
        from kernel_entries import kernel_snippet
        source = '''__global__ void conditional(int* x) {
#if defined(FAST)
    if (threadIdx.x < 32) {
#elif defined(MEDIUM)
    if (threadIdx.x < 16) {
#else
    if (threadIdx.x < 8) {
#endif
#ifdef ADD_OFFSET
        { x[0] += 2; }
#endif
        x[0] += 1;
    }
}
__global__ void unreviewed(int* x) { x[0] = 3; }
'''
        entries = cuda_kernel_entries(source)
        self.assertEqual(['conditional', 'unreviewed'], [e.name for e in entries])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.cu'
            path.write_text(source)
            snippet = kernel_snippet(path, entries[0].name, entries[0].line)
            self.assertEqual(source[:source.index('__global__ void unreviewed')], snippet)
            self.assertIn('#elif defined(MEDIUM)', snippet)
            self.assertIn('x[0] += 1;', snippet)
            changed = cuda_kernel_entries(source.replace('threadIdx.x < 8', 'threadIdx.x < 4'))
            self.assertNotEqual(entries[0].body_sha256, changed[0].body_sha256)

    def test_cuda_declaration_prefix_macros_expose_each_real_invocation(self):
        from kernel_entries import kernel_snippet
        prefix = ('#define DEFINE_FORWARD(kernelName, ...) \\\n'
                  'template<typename Traits, __VA_ARGS__> \\\n'
                  '__global__ void kernelName(const Params params)\n')
        invocations = [
            'DEFINE_FORWARD(forward_main, bool Causal) { compute_main(params); }',
            'DEFINE_FORWARD(forward_split, bool Split) { compute_split(params); }',
            'DEFINE_FORWARD(forward_combine, int Parts) { compute_combine(params); }',
        ]
        source = prefix + '\n' + '\n\n'.join(invocations) + '\n'
        entries = cuda_kernel_entries(source)
        self.assertEqual(['forward_main', 'forward_split', 'forward_combine'], [e.name for e in entries])
        self.assertEqual([5, 7, 9], [e.line for e in entries])
        self.assertEqual(3, len({e.body_sha256 for e in entries}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'launch.h'
            path.write_text(source)
            for index, entry in enumerate(entries):
                snippet = kernel_snippet(path, entry.name, entry.line)
                self.assertEqual(prefix + '\n' + invocations[index] + '\n', snippet)
                for other in entries:
                    if other.name != entry.name:
                        self.assertNotIn(other.name, snippet)
            changed = cuda_kernel_entries(source.replace('compute_split(params)', 'compute_split_new(params)'))
            self.assertEqual(entries[0].body_sha256, changed[0].body_sha256)
            self.assertNotEqual(entries[1].body_sha256, changed[1].body_sha256)
            self.assertEqual(entries[2].body_sha256, changed[2].body_sha256)

    def test_cuda_prefix_invocation_balances_nested_arguments_and_preserves_template(self):
        from kernel_entries import kernel_snippet
        prefix = ('#define DECLARE_KERNEL(name, ...) \\\n'
                  'CUTLASS_GLOBAL void name(decltype(make(1, (2 + 3))) value)\n')
        invocation = ('template<int N>\n'
                      'DECLARE_KERNEL(nested, TYPE(Shape(2, (N + 4)))) /* brace follows */ {\n'
                      '    launch_helper(value, nested_call(1, 2));\n}\n')
        source = prefix + '\n' + invocation + 'DECLARE_KERNEL(other, int) { other_body(); }\n'
        entries = cuda_kernel_entries(source)
        self.assertEqual(['nested', 'other'], [e.name for e in entries])
        self.assertEqual(5, entries[0].line)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'launch.cuh'
            path.write_text(source)
            self.assertEqual(prefix + '\n' + invocation,
                             kernel_snippet(path, entries[0].name, entries[0].line))

    def test_cuda_prefix_scaffolds_prototypes_comments_strings_and_nonliteral_names_are_not_entries(self):
        source = ('#define PREFIX(name, ...) \\\n'
                  '__global__ void name(int* x)\n'
                  '#define PROTOTYPE(name) __global__ void name();\n'
                  '#define WRONG_FIRST(unused, actualName) __global__ void actualName(int* x)\n'
                  '#define FORWARD_MACRO() PREFIX(nested_macro, int) { phantom(); }\n'
                  '// PREFIX(commented, int) { phantom(); }\n'
                  'const char* s = "PREFIX(stringified, int) { phantom(); }";\n'
                  'PREFIX(declared_only, int);\n'
                  'PROTOTYPE(also_declared);\n'
                  'WRONG_FIRST(ignored, unsupported_name) { unsupported(); }\n'
                  'PREFIX("not_an_identifier", int) { phantom(); }\n'
                  'PREFIX((parenthesized_name), int) { phantom(); }\n'
                  'PREFIX(qualified::name, int) { phantom(); }\n'
                  'PREFIX(concatenated ## name, int) { phantom(); }\n'
                  'PREFIX(actual_name, call(1, nested(2, 3))) { x[0] = 1; }\n')
        entries = cuda_kernel_entries(source)
        self.assertEqual(['actual_name'], [e.name for e in entries])
        self.assertEqual(15, entries[0].line)

    def test_cuda_prefix_recognition_preserves_macro_owned_and_ordinary_cuda_entries(self):
        source = ('#define PREFIX(name) __global__ void name(int* x)\n'
                  '#define OWNED(name) \\\n'
                  '__global__ void name(int* x) { x[0] = 4; }\n'
                  'PREFIX(from_prefix) { x[0] = 1; }\n'
                  '__global__ void ordinary(int* x) { x[0] = 2; }\n'
                  'CUTLASS_GLOBAL void cutlass_entry(int* x) { x[0] = 3; }\n')
        old_source = source[source.index('#define OWNED'):source.index('PREFIX(from_prefix)')] + source[source.index('__global__ void ordinary'):]
        previous = {e.name: e for e in cuda_kernel_entries(old_source)}
        current = {e.name: e for e in cuda_kernel_entries(source)}
        self.assertEqual({'name', 'from_prefix', 'ordinary', 'cutlass_entry'}, set(current))
        for name, entry in previous.items():
            self.assertEqual(entry.semantic_key, current[name].semantic_key)
        self.assertEqual([3, 4, 5, 6], [e.line for e in cuda_kernel_entries(source)])

    def test_cuda_prefix_undef_and_redefinition_bound_association(self):
        from kernel_entries import kernel_snippet
        first = '#define PREFIX(name) __global__ void name(First value)\n'
        second = '#define PREFIX(name) CUTLASS_GLOBAL void name(Second value)\n'
        source = (first + 'PREFIX(before_undef) { first_body(); }\n'
                  '#undef PREFIX\n'
                  'PREFIX(no_definition) { phantom(); }\n' + second +
                  'PREFIX(after_redefinition) { second_body(); }\n')
        entries = cuda_kernel_entries(source)
        self.assertEqual(['before_undef', 'after_redefinition'], [e.name for e in entries])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'launch.h'
            path.write_text(source)
            self.assertEqual(second + '\nPREFIX(after_redefinition) { second_body(); }\n',
                             kernel_snippet(path, entries[1].name, entries[1].line))

    def test_repo_inventory_records_literal_prefix_names_and_physical_invocation_lines(self):
        from repo_import_support import prepared, commit, import_repo, load_config, read
        from kernel_entries import kernel_snippet
        source = ('#define DEFINE_KERNEL(name, ...) \\\n'
                  'template<typename Traits, __VA_ARGS__> \\\n'
                  '__global__ void name(int* x)\n'
                  '\nDEFINE_KERNEL(main_kernel, int N) { x[0] = N; }\n'
                  '\nDEFINE_KERNEL(split_kernel, bool Split) { x[0] = Split; }\n'
                  '\nDEFINE_KERNEL(combine_kernel, int Parts) { x[0] += Parts; }\n')
        with tempfile.TemporaryDirectory() as directory:
            wiki, repo, _, _ = prepared(Path(directory))
            (repo / 'launch.cuh').write_text(source)
            commit(repo)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', load_config(wiki))
            inventory = read(run / 'inventory.yaml')
            candidates = [c for c in inventory['candidates'] if c['path'] == 'launch.cuh']
            self.assertEqual(['main_kernel', 'split_kernel', 'combine_kernel'], [c['entry'] for c in candidates])
            self.assertEqual([5, 7, 9], [c['line'] for c in candidates])
            self.assertEqual(3, len({c['fingerprints']['entry_sha256'] for c in candidates}))
            checkout = Path(read(run / 'snapshot.yaml')['checkout'])
            for candidate in candidates:
                snippet = kernel_snippet(checkout / candidate['path'], candidate['entry'], candidate['line'])
                self.assertTrue(snippet.startswith(source[:source.index('\nDEFINE_KERNEL')]))
                self.assertIn(candidate['entry'], snippet)
                self.assertEqual(1, snippet.count('DEFINE_KERNEL(') - 1)
