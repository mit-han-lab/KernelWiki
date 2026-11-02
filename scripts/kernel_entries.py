#!/usr/bin/env python3

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CUDA_EXTENSIONS = {".cu", ".cuh", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inl"}


@dataclass(frozen=True)
class KernelEntry:
    name: str
    kind: str
    body_sha256: str
    line: int

    @property
    def semantic_key(self) -> tuple[str, str, str]:
        return self.kind, self.name, self.body_sha256

    def manifest_record(self) -> dict[str, str]:
        return {
            "name": self.name,
            "kind": self.kind,
            "body_sha256": self.body_sha256,
        }


def decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def python_kernel_entries(text: str) -> list[KernelEntry]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    entries = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = {decorator_name(item) for item in node.decorator_list}
        if any(name.endswith("cute.kernel") for name in decorators):
            kind = "cute_kernel"
        elif any(name.endswith("triton.jit") for name in decorators):
            kind = "triton_jit"
        else:
            continue
        body = ast.dump(
            ast.Module(body=node.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        )
        entry = KernelEntry(
            name=node.name,
            kind=kind,
            body_sha256=hashlib.sha256(body.encode()).hexdigest(),
            line=min(
                [item.lineno for item in node.decorator_list if hasattr(item, "lineno")]
                or [node.lineno]
            ),
        )
        if entry.semantic_key not in seen:
            seen.add(entry.semantic_key)
            entries.append(entry)
    return sorted(entries, key=lambda item: (item.line, item.name, item.body_sha256))


def mask_cpp(text: str) -> str:
    """Hide comments/literals without changing source offsets or line numbers."""
    token = re.compile(
        r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"'
        r'|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        flags=re.DOTALL,
    )
    return token.sub(lambda match: ''.join('\n' if c == '\n' else ' ' for c in match.group()), text)


def cpp_body_end(syntax_text: str, start: int) -> Optional[int]:
    """Find an extent without preprocessing or executing upstream code.

    Preserve existing extents. If flat brace counting fails, merge equally
    balanced conditional arms instead of adding mutually exclusive braces.
    Ambiguous arm depths remain unsupported rather than guessing an extent.
    """
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(syntax_text)):
        depth += (syntax_text[index] == '{') - (syntax_text[index] == '}')
        if depth == 0:
            return index + 1
    depth = 0
    frames = []
    for token in re.finditer(r'^[ \t]*#[^\n]*|[{}]', syntax_text[start:], re.MULTILINE):
        value = token.group()
        if value.lstrip().startswith('#'):
            directive = re.match(r'\s*#\s*(\w+)', value)
            kind = directive.group(1) if directive else ''
            if kind in {'if', 'ifdef', 'ifndef'}:
                frames.append({'entry': depth, 'ends': [], 'has_else': False})
            elif kind in {'elif', 'elifdef', 'elifndef', 'else'} and frames:
                frame = frames[-1]
                if frame['has_else']:
                    return None
                frame['ends'].append(depth)
                depth = frame['entry']
                frame['has_else'] = kind == 'else'
            elif kind == 'endif' and frames:
                frame = frames.pop()
                ends = frame['ends'] + [depth]
                if not frame['has_else']:
                    ends.append(frame['entry'])
                if len(set(ends)) != 1:
                    return None
                depth = ends[0]
            continue
        depth += 1 if value == '{' else -1
        if depth == 0 and not frames:
            return start + token.end()
        if depth < 0:
            return None
    return None


CUDA_GLOBAL_MARKER = re.compile(r"\b(?:__global__|CUTLASS_GLOBAL)\b")


def cpp_directives(masked):
    """Physical extents of directives, including backslash-continued lines."""
    for match in re.finditer(r'(?m)^[ \t]*#', masked):
        # Continued directive lines cannot themselves start a new directive.
        start = match.start()
        previous = masked.rfind('\n', 0, max(0, start - 1)) + 1
        if start and masked[previous:start].rstrip(' \t\r\n').endswith('\\'):
            continue
        end = masked.find('\n', start)
        end = len(masked) if end < 0 else end + 1
        while masked[start:end].rstrip('\r\n').endswith('\\') and end < len(masked):
            following = masked.find('\n', end)
            end = len(masked) if following < 0 else following + 1
        yield start, end


def cpp_parenthesis_end(masked, start, stop=None):
    """Return the offset after a balanced parenthesized token sequence."""
    stop = len(masked) if stop is None else stop
    depth = 0
    for position in range(start, stop):
        depth += (masked[position] == '(') - (masked[position] == ')')
        if depth == 0:
            return position + 1
    return None


def cuda_definition(masked, start, stop=None, *, single_name=False):
    """Find a function body after balanced parameter/attribute sequences.

    Braced default arguments belong to the parameter list, not to the body.
    Skip complete parenthesized sequences so constructors in defaults cannot
    replace the function name. Unknown launch-attribute macros before the actual
    name remain supported for direct CUDA qualifiers; aliases stay conservative.
    """
    stop = len(masked) if stop is None else stop
    cursor, names = start, []
    attributes = {'__launch_bounds__', '__cluster_dims__', '__attribute__', '__declspec',
                  'alignas', 'decltype', 'noexcept'}
    while cursor < stop:
        token = re.search(r'([A-Za-z_]\w*)\s*\(|[;{]', masked[cursor:stop])
        if token is None:
            break
        position = cursor + token.start()
        if token.group(1) is None:
            if token.group() != '{' or not names or (single_name and len(names) != 1):
                break
            end = cpp_body_end(masked, position)
            return (names[-1], position, end) if end is not None else None
        end = cpp_parenthesis_end(masked, cursor + token.end() - 1, stop)
        if end is None:
            break
        if token.group(1) not in attributes:
            names.append(token.group(1))
        cursor = end
    if not single_name:
        # Preserve existing lexical coverage of declarations whose #if arms
        # repeat the opening function parenthesis. This does not choose an arm
        # or preprocess source. Alias recognition must not use this fallback.
        terminator = re.search(r'[;{]', masked[start:stop])
        if terminator and terminator.group() == '{':
            body_start = start + terminator.start()
            header = masked[start:body_start]
            if re.search(r'^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b', header, re.MULTILINE):
                calls = re.findall(r'([A-Za-z_]\w*)\s*\(', header)
                end = cpp_body_end(masked, body_start)
                if calls and end is not None:
                    return calls[-1], body_start, end
    return None


def cuda_prefix_macros(masked):
    """Recognize only same-file, first-name-parameter declaration prefixes.

    This is lexical recognition, not preprocessor expansion. Body-owning macros
    remain ordinary CUDA entries. Every bodyless CUDA function-like scaffold is
    hidden from ordinary extraction, including unsupported parameter layouts.
    """
    directives = list(cpp_directives(masked))
    declarations, prefixes = [], []
    for start, end in directives:
        logical = re.sub(r'\\\r?\n', ' ', masked[start:end])
        match = re.match(r'[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)\(([^)]*)\)(.*)', logical, re.DOTALL)
        if not match:
            continue
        name, parameters, replacement = match.groups()
        marker = CUDA_GLOBAL_MARKER.search(replacement)
        if marker is None or re.search(r'[{};]', replacement):
            continue
        declarations.append((start, end))
        first = parameters.split(',')[0].strip()
        if not re.fullmatch(r'[A-Za-z_]\w*', first) or '#' in replacement:
            continue
        signatures = re.finditer(r'\b' + re.escape(first) + r'\s*\(', replacement[marker.end():])
        is_function_name = False
        for signature in signatures:
            signature_end = cpp_parenthesis_end(replacement, marker.end() + signature.end() - 1)
            if signature_end is not None and not replacement[signature_end:].strip():
                is_function_name = True
                break
        if not is_function_name:
            continue
        # A later redefinition/undef ends the lexical extent; no branch selection
        # or included/header-defined macro identity is inferred.
        stop = len(masked)
        for later_start, later_end in directives:
            if later_start < end:
                continue
            if re.match(r'[ \t]*#[ \t]*(?:define|undef)[ \t]+' + re.escape(name) + r'\b',
                        masked[later_start:later_end]):
                stop = later_start
                break
        prefixes.append((name, start, end, stop))
    return directives, declarations, prefixes


def cuda_prefix_invocations(masked, directives, prefixes):
    """Locate literal-name invocations followed directly by a balanced body."""
    for macro, definition_start, definition_end, stop in prefixes:
        pattern = re.compile(r'\b' + re.escape(macro) + r'\s*\(')
        for match in pattern.finditer(masked, definition_end, stop):
            if any(start <= match.start() < end for start, end in directives):
                continue
            first = re.match(r'\s*([A-Za-z_]\w*)\s*(?=,|\))', masked[match.end():])
            if first is None:
                continue
            position = cpp_parenthesis_end(masked, match.end() - 1, stop)
            if position is None:
                continue
            while position < stop and masked[position].isspace():
                position += 1
            if position >= stop or masked[position] != '{':
                continue
            body_end = cpp_body_end(masked, position)
            if body_end is not None:
                yield first.group(1), match.start(), position, body_end, definition_start, definition_end


def cuda_qualifier_macros(masked):
    """Recognize same-file object-like CUDA declaration qualifiers lexically."""
    directives = list(cpp_directives(masked))
    declarations, aliases = [], []
    for start, end in directives:
        logical = re.sub(r'\\\r?\n', ' ', masked[start:end])
        match = re.match(r'[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)([ \t]+)(.*)', logical, re.DOTALL)
        if not match:
            continue
        name, _, replacement = match.groups()
        if not CUDA_GLOBAL_MARKER.search(replacement) or re.search(r'[{};#]', replacement):
            continue
        # Even unsupported bodyless aliases must not capture a later struct or
        # host helper as though that were the macro's kernel body.
        declarations.append((start, end))
        qualifier = re.compile(r'\s*(?:__global__|CUTLASS_GLOBAL|static|inline|__forceinline__|void)\b')
        attribute = re.compile(r'\s*(?:__launch_bounds__|__cluster_dims__|__attribute__|__declspec|alignas)\s*(?=\()')
        position = 0
        while position < len(replacement) and replacement[position:].strip():
            token = qualifier.match(replacement, position)
            if token:
                position = token.end()
                continue
            token = attribute.match(replacement, position)
            if token:
                position = cpp_parenthesis_end(replacement, token.end())
                if position is not None:
                    continue
            break
        else:
            # CUTLASS_GLOBAL is already a built-in marker; preserve its existing
            # definition locations and snippet representation.
            if name == 'CUTLASS_GLOBAL':
                continue
            stop = len(masked)
            for later_start, later_end in directives:
                if later_start >= end and re.match(
                    r'[ \t]*#[ \t]*(?:define|undef)[ \t]+' + re.escape(name) + r'\b',
                    masked[later_start:later_end],
                ):
                    stop = later_start
                    break
            aliases.append((name, start, end, stop))
    return declarations, aliases


def cuda_qualifier_invocations(masked, directives, aliases):
    """Bind a qualifier use to one named function, excluding attribute calls."""
    for macro, definition_start, definition_end, stop in aliases:
        for match in re.finditer(r'\b' + re.escape(macro) + r'\b', masked[definition_end:stop]):
            position = definition_end + match.start()
            if any(start <= position < end for start, end in directives):
                continue
            result = cuda_definition(masked, definition_end + match.end(), stop, single_name=True)
            if result is not None:
                name, body_start, body_end = result
                yield name, position, body_start, body_end, definition_start, definition_end


def cuda_syntax(text):
    """Hide qualifier aliases and declaration scaffolds with stable offsets."""
    masked = mask_cpp(text)
    alias = re.compile(r"(?m)^[ \t]*#[ \t]*define[ \t]+\w+[ \t]+(?:__global__|CUTLASS_GLOBAL)(?:[ \t]+(?:static|inline))*[ \t]*$")
    masked = alias.sub(lambda match: ' ' * len(match.group()), masked)
    _, declarations, _ = cuda_prefix_macros(masked)
    qualifier_declarations, _ = cuda_qualifier_macros(masked)
    declarations += qualifier_declarations
    for start, end in declarations:
        masked = masked[:start] + ''.join('\n' if c == '\n' else ' ' for c in masked[start:end]) + masked[end:]
    return masked


def cuda_kernel_entries(text: str) -> list[KernelEntry]:
    syntax_text = cuda_syntax(text)
    entries = []
    seen = set()
    masked = mask_cpp(text)
    directives, _, prefixes = cuda_prefix_macros(masked)
    _, aliases = cuda_qualifier_macros(masked)
    definitions = [(name, position, body_start, body_end)
                   for name, position, body_start, body_end, _, _
                   in cuda_prefix_invocations(masked, directives, prefixes)]
    definitions.extend((name, position, body_start, body_end)
                       for name, position, body_start, body_end, _, _
                       in cuda_qualifier_invocations(masked, directives, aliases))
    for marker in CUDA_GLOBAL_MARKER.finditer(syntax_text):
        result = cuda_definition(syntax_text, marker.end())
        if result is not None:
            name, body_start, body_end = result
            definitions.append((name, marker.start(), body_start, body_end))
    for name, position, body_start, body_end in sorted(definitions, key=lambda row: row[1]):
        body = text[body_start:body_end]
        body = re.sub(r"//.*?$|/\*.*?\*/", "", body, flags=re.MULTILINE | re.DOTALL)
        body = re.sub(r"\s+", "", body)
        if body == "{}":
            continue
        entry = KernelEntry(
            name=name,
            kind="cuda_global",
            body_sha256=hashlib.sha256(body.encode()).hexdigest(),
            line=text[:position].count("\n") + 1,
        )
        if entry.semantic_key not in seen:
            seen.add(entry.semantic_key)
            entries.append(entry)
    return sorted(entries, key=lambda item: (item.line, item.name, item.body_sha256))


def kernel_entries(path: Path, text: Optional[str] = None) -> list[KernelEntry]:
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        return python_kernel_entries(text)
    if path.suffix in CUDA_EXTENSIONS:
        return cuda_kernel_entries(text)
    return []


def kernel_snippet(path: Path, name: str, line: int) -> str:
    """Render only the selected definition, never other candidates in its file."""
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines(keepends=True)
    if path.suffix == '.py':
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                first = min([d.lineno for d in node.decorator_list] or [node.lineno])
                if first == line:
                    return ''.join(lines[first - 1:node.end_lineno])
    else:
        # Preserve offsets while hiding comments and strings for brace matching.
        masked = cuda_syntax(text)
        offset = sum(len(item) for item in lines[:line - 1])
        source_masked = mask_cpp(text)
        directives, _, prefixes = cuda_prefix_macros(source_masked)
        _, aliases = cuda_qualifier_macros(source_masked)
        macro_definitions = [*cuda_prefix_invocations(source_masked, directives, prefixes),
                             *cuda_qualifier_invocations(source_masked, directives, aliases)]
        for entry_name, position, _, end, definition_start, definition_end in macro_definitions:
            if entry_name != name or text[:position].count('\n') + 1 != line:
                continue
            prefix_start = offset
            boundary = max(masked.rfind('}', 0, position), masked.rfind(';', 0, position))
            template = masked.rfind('template', boundary + 1, position)
            if template >= 0:
                prefix_start = template
            return text[definition_start:definition_end] + '\n' + text[prefix_start:end] + '\n'

        match = CUDA_GLOBAL_MARKER.search(masked, offset)
        if match is not None and text[:match.start()].count('\n') + 1 == line:
            marker = match.start()
            result = cuda_definition(masked, match.end())
            if result is not None and result[0] == name:
                _, _, end = result
                prefix_start = offset
                boundary = max(masked.rfind('}', 0, marker), masked.rfind(';', 0, marker))
                template = masked.rfind('template', boundary + 1, marker)
                if template >= 0:
                    prefix_start = template
                return text[prefix_start:end] + '\n'
    raise ValueError(f'Cannot render selected entry {path}::{name}')
