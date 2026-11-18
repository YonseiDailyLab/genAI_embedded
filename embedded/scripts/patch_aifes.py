Import("env")

from pathlib import Path
import re


def patch_q7_quant_dump(file_path: Path) -> bool:
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    marker = 'printf("TEST2\\n");'
    if marker not in text:
        return False

    start = text.find(marker)
    # Include the preceding newline print if present.
    start_printf_nl = text.rfind('printf("\\n");', 0, start)
    if start_printf_nl != -1:
        start = start_printf_nl

    end = text.find("//Free Q7 data memory", start)
    if end == -1:
        return False

    patched = text[:start] + "    // (patched) removed verbose debug dump\n\n" + text[end:]
    if patched == text:
        return False

    file_path.write_text(patched, encoding="utf-8")
    return True


def patch_q7_quant_leak(file_path: Path) -> bool:
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    start_marker = "int8_t AIFES_E_quantisation_fnn_f32_to_q7"
    end_marker = "int8_t AIFES_E_inference_fnn_q7"

    start = text.find(start_marker)
    end = text.find(end_marker, start + 1)
    if start == -1 or end == -1:
        return False

    func = text[start:end]
    if "memory_ptr_q7" in func:
        return False

    anchor = "memory_size = aialgo_sizeof_inference_memory(&model_q7);"
    anchor_pos = func.find(anchor)
    if anchor_pos == -1:
        return False

    malloc_pos = func.find("memory_ptr = malloc(memory_size);", anchor_pos)
    if malloc_pos == -1:
        return False

    # Preserve indentation.
    line_start = func.rfind("\n", 0, malloc_pos) + 1
    indent_match = re.match(r"[ \t]*", func[line_start:malloc_pos])
    indent = indent_match.group(0) if indent_match else ""

    func = func[:malloc_pos] + f"{indent}void *memory_ptr_q7 = malloc(memory_size);" + func[malloc_pos + len("memory_ptr = malloc(memory_size);") :]

    # Patch the NULL check that follows the second malloc.
    null_check_pos = func.find("if(memory_ptr == NULL)", malloc_pos)
    if null_check_pos != -1:
        func = func[:null_check_pos] + "if(memory_ptr_q7 == NULL)" + func[null_check_pos + len("if(memory_ptr == NULL)") :]

    # Patch scheduling call for q7 model.
    func = func.replace(
        "aialgo_schedule_inference_memory(&model_q7, memory_ptr, memory_size);",
        "aialgo_schedule_inference_memory(&model_q7, memory_ptr_q7, memory_size);",
    )

    # Free both allocations (fix leak).
    free_pos = func.rfind("free(memory_ptr);")
    if free_pos == -1:
        return False
    if "free(memory_ptr_q7);" not in func[free_pos:]:
        after = free_pos + len("free(memory_ptr);")
        func = func[:after] + f"\n{indent}free(memory_ptr_q7);" + func[after:]

    patched = text[:start] + func + text[end:]
    if patched == text:
        return False

    file_path.write_text(patched, encoding="utf-8")
    return True


# Newer espressif32 toolchains promote several -Wmaybe-uninitialized warnings
# in the third-party AIfES sources to errors, and neither project build_flags
# nor env CCFLAGS reach these library compile units. Disable the warning at the
# source level by prepending a pragma to every AIfES .c file (idempotent).
_MU_PRAGMA = '#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"\n'


def disable_maybe_uninitialized(src_root: Path) -> int:
    count = 0
    for c_file in src_root.rglob("*.c"):
        text = c_file.read_text(encoding="utf-8", errors="ignore")
        if _MU_PRAGMA in text:
            continue
        c_file.write_text(_MU_PRAGMA + text, encoding="utf-8")
        count += 1
    return count


project_dir = Path(env["PROJECT_DIR"])
libdeps_dir = project_dir / ".pio" / "libdeps"

if libdeps_dir.exists():
    for env_dir in libdeps_dir.iterdir():
        aifes_root = env_dir / "AIfES for Arduino" / "src"
        if not aifes_root.exists():
            continue

        candidate = aifes_root / "basic" / "express" / "aifes_express_q7_fnn.c"
        if candidate.exists():
            if patch_q7_quant_dump(candidate):
                print(f"[patch_aifes] Patched debug dump: {candidate}")
            if patch_q7_quant_leak(candidate):
                print(f"[patch_aifes] Patched quantization leak: {candidate}")

        n = disable_maybe_uninitialized(aifes_root)
        if n:
            print(f"[patch_aifes] Disabled -Wmaybe-uninitialized in {n} AIfES source file(s)")
