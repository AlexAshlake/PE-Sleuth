# classify_from_c_code.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import re
import json
import time
import math
import html
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
from model_server import LLMInterface, LoRALLMInterface, ModelConfig

# =========================
# Config
# =========================
class ExperimentConfig:
    """
    Central configuration for the hierarchical semantic aggregation experiment.
    """
    # --- I/O Directories ---
    REPO_ROOT = Path(__file__).resolve().parent
    INPUT_DIR:  Path = REPO_ROOT / "Classify" / "Classify_Input_C_Code"
    OUTPUT_DIR: Path = REPO_ROOT / "Classify" / "Classify_Output_All"

    # --- Sub-directory Names (do not rename existing ones) ---
    DIR_METADATA = "1a_parsed_metadata"
    DIR_CALL_GRAPHS = "1b_call_graphs"
    DIR_CHUNKS = "2_smart_chunks"
    DIR_VISUALIZATIONS = "3_visualizations"
    DIR_CHUNK_SUMMARIES = "4_chunk_summaries"
    DIR_PROGRAM_SUMMARIES = "5_program_summaries"
    DIR_FINAL_IR = "6_final_ir"
    DIR_CLASSIFICATION = "7_classification_results"
    DIR_CLASSIFICATION_RATIONALE = "8_classification_rationale"

    # --- Step Execution Switches ---
    RUN_STEP_1A_METADATA_EXTRACTION: bool = True
    RUN_STEP_1B_CALL_GRAPH_GENERATION: bool = True
    RUN_STEP_2_SMART_CHUNKING: bool = True
    RUN_STEP_3_VISUALIZATIONS: bool = True
    RUN_STEP_4_CHUNK_SUMMARIZATION: bool = True
    RUN_STEP_5_PROGRAM_SUMMARIZATION: bool = True
    RUN_STEP_6_IR_CONSTRUCTION: bool = True
    RUN_STEP_7_CLASSIFICATION: bool = True
    RUN_STEP_8_CLASSIFICATION_RATIONALE: bool = True

    # --- Step 7 behavior knobs ---
    # If True, Step 7 uses LoRA fine-tuned model (via model_server.LoRALLMInterface).
    # If False, Step 7 uses the same base model as Steps 4/5.
    USE_LORA_FOR_CLASSIFICATION: bool = True

    # --- Ablation toggles ---
    RUN_ABLATION_E_META: bool = True
    RUN_ABLATION_E_SUMMARY: bool = True

    # --- Overwrite behavior (checkpoint-friendly) ---
    OVERWRITE_EXISTING: bool = False

    # --- Save LLM inputs for auditing ---
    SAVE_LLM_INPUTS: bool = False

    # --- Smart Chunking Parameters ---
    CHUNK_TOKEN_SIZE: int = 30 * 1024

    # --- Metadata hard budget ---
    MAX_METADATA_BYTES: int = 4096
    MAX_TOP_STRINGS: int = 120
    MAX_STRING_LEN: int = 200
    MAX_GLOBAL_DECLS: int = 80
    MAX_API_IMPORTS: int = 120

    # --- Logging ---
    LOG_FILE_NAME: str = "experiment.log"
    LOG_LEVEL = logging.INFO

    # --- LLM Prompts ---
    PROMPT_CHUNK_SUMMARY: str = (
        "Analyze the following C code fragment and provide its key behaviors. "
        "Your response should be concise and focused on describing the key behaviors."
        "\n\n--- CODE ---\n{code_chunk}"
    )
    PROMPT_PROGRAM_SUMMARY: str = (
        "You are a senior malware analyst. Your task is to analyze a list of behavioral summaries from a program's code fragments and determine if it is ransomware or benign.\n\n"
        "Your response MUST start with either `[RANSOMWARE]` or `[BENIGN]`. Follow the tag with a concise justification.\n\n"
        "CRITICAL INSTRUCTION: Scan all summaries for high-risk ransomware indicators. The presence of even a few strong indicators (like file encryption, shadow copy deletion, ransom notes) in some fragments is far more important than the absence of them in many other fragments. Do not be misled by the volume of benign-looking, low-level operations.\n\n"
        "Classify as [RANSOMWARE] if you find credible evidence of actions such as:\n"
        "- Widespread file encryption or preparation for it (key generation, directory traversal).\n"
        "- Deletion of backups or volume shadow copies.\n"
        "- Creation of a ransom note.\n"
        "- Terminating processes/services to enable encryption.\n\n"
        "--- BEHAVIORAL SUMMARIES ---\n{combined_summaries}"
    )
    PROMPT_CLASSIFICATION: str = (
        "You are classifying code as ransomware vs. benign using the structured JSON (global_metadata + program_summary).\n\n"
        "STRICT OUTPUT: Return EXACTLY ONE LABEL in square brackets: [RANSOMWARE] or [BENIGN].\n"
        "HARD RULE: If program_summary.text contains the literal tag [RANSOMWARE] OR if any of these metadata indicators is true — "
        "\"vssadmin\", \"delete shadows\", \".onion\", \"decrypt\", \"ransom\", \"CryptAcquireContext\" — then output [RANSOMWARE].\n"
        "Otherwise, decide holistically using the available signals.\n\n"
        "--- INTERMEDIATE REPRESENTATION (JSON) ---\n{ir_json_block}"
    )
    PROMPT_CLASSIFICATION_RATIONALE: str = (
        "You are a senior malware analyst. The program has already been classified as {label}.\n"
        "Using ONLY the structured IR (global_metadata + program_summary) below, explain concisely why this classification is appropriate.\n"
        "Focus on concrete evidence: specific strings, APIs, behaviors, and their connections to ransomware/benign patterns.\n"
        "Avoid speculation. Return a short paragraph followed by 3–6 bullet points.\n\n"
        "--- IR(JSON) ---\n{ir_json_block}"
    )


# =========================
# Utilities
# =========================
def out_dir(name: str) -> Path:
    p = ExperimentConfig.OUTPUT_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def llm_input_dir() -> Path:
    p = ExperimentConfig.OUTPUT_DIR / "llm_input"
    p.mkdir(parents=True, exist_ok=True)
    return p

def llm_input_path_for(json_path: Path) -> Path:
    name = json_path.name
    if name.endswith(".json"):
        name = name[:-5] + ".txt"
    else:
        name = name + ".txt"
    return llm_input_dir() / name


def setup_logging():
    ExperimentConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ExperimentConfig.OUTPUT_DIR / ExperimentConfig.LOG_FILE_NAME
    logging.basicConfig(
        level=ExperimentConfig.LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    )
    logging.info("Logging initialized.")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))


def write_json(p: Path, obj: Dict[str, Any]):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html(p: Path, html_str: str):
    p.write_text(html_str, encoding="utf-8")


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def out_paths(sample_path: Path) -> Dict[str, Path]:
    stem = sample_path.stem
    return {
        "metadata": out_dir(ExperimentConfig.DIR_METADATA) / f"{stem}_metadata.json",
        "callgraph": out_dir(ExperimentConfig.DIR_CALL_GRAPHS) / f"{stem}_callgraph.json",
        "chunks": out_dir(ExperimentConfig.DIR_CHUNKS) / f"{stem}_chunks.json",
        "viz_html": out_dir(ExperimentConfig.DIR_VISUALIZATIONS) / f"{stem}_callgraph_chunks.html",
        "chunk_summaries": out_dir(ExperimentConfig.DIR_CHUNK_SUMMARIES) / f"{stem}_chunk_summaries.json",
        "program_summary": out_dir(ExperimentConfig.DIR_PROGRAM_SUMMARIES) / f"{stem}_program_summary.json",
        "ir": out_dir(ExperimentConfig.DIR_FINAL_IR) / f"{stem}_ir.json",
        "classification": out_dir(ExperimentConfig.DIR_CLASSIFICATION) / f"{stem}_classification.json",
        "rationale": out_dir(ExperimentConfig.DIR_CLASSIFICATION_RATIONALE) / f"{stem}_rationale.json",
    }

# =========================
# Token Counter
# =========================
class TokenCounter:
    def __init__(self, llm: Optional[LLMInterface]):
        self.llm = llm
        self.tok = None
        if llm is not None:
            try:
                self.tok = llm.get_tokenizer()
            except Exception as e:
                logging.warning(f"Tokenizer not available from LLM: {e}")

    def count(self, text: str) -> int:
        if self.tok:
            try:
                ids = self.tok.encode(text, add_special_tokens=False)
                return len(ids)
            except Exception:
                pass
        return max(1, int(len(text) / 3.8))

# =========================
# LLM Helpers
# =========================
def ask_llm(llm: LLMInterface, prompt: str, max_retries: int = 3, backoff: float = 0.7) -> str:
    last = None
    for i in range(max_retries):
        try:
            return llm.generate(prompt=prompt)
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"LLM generation failed after retries: {last}")


# =========================
# Step 1A: Metadata Extraction
# =========================
STRING_RE = re.compile(
    r'(?P<w>L")(?P<wstr>(?:[^"\\]|\\.)*?)"|(?P<n>")(?P<nstr>(?:[^"\\]|\\.)*?)"',
    re.MULTILINE
)
EXTERN_LINE_RE = re.compile(r'^\s*//\s*extern\s+.+$', re.MULTILINE)
GLOBAL_DECL_RE = re.compile(
    r'^\s*(?:const\s+)?(?:unsigned\s+|signed\s+)?(?:char|short|int|long|__int\d+|_DWORD|_BYTE|_WORD|_QWORD|BYTE|WORD|DWORD|LPCSTR|LPCWSTR|H[A-Z][A-Za-z0-9_]*|struct\s+\w+)\s+([A-Za-z_][\w\[\]{}]*)\s*(?:=\s*[^;]+)?;',
    re.MULTILINE
)

def _shorten(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    keep = max(4, max_len // 2 - 2)
    return s[:keep] + " … " + s[-keep:]

def extract_strings(code: str) -> List[str]:
    res = []
    for m in STRING_RE.finditer(code):
        s = m.group('wstr') if m.group('w') else m.group('nstr')
        if not s:
            continue
        ss = s.strip()
        if len(ss) < 3:
            continue
        res.append(ss)
    seen = set()
    dedup = []
    for s in res:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup

def rank_strings(strings: List[str]) -> List[str]:
    suspects = re.compile(
        r'\.onion|vssadmin|delete\s+shadows|shadow\s+copy|tor|decrypt|bitcoin|btc|ransom|key\s+pair|Crypt|Acquire|Encrypt|Decrypt|'
        r'CreateFileW|WriteFile|ReadFile|SetFileAttributesW|MoveFileExW|FindFirstFileW|FindNextFileW|GetLogicalDrives|GetVolume|'
        r'ShellExecuteW|cmd\.exe|sql\.exe|\\AppData\\|\\Windows\\System32\\|NetShareEnum|WNet|CriticalSection|Mutex|Semaphore|'
        r'Reg(Open|Set|Create|Close)Key|Service|OpenProcess|TerminateProcess',
        re.IGNORECASE
    )
    ranked = []
    for s in strings:
        s = _shorten(s, ExperimentConfig.MAX_STRING_LEN)
        score = 0.0
        score += min(len(s) / 20.0, 3.0)
        if re.search(r'[\\/]', s): score += 1.0
        if re.search(r'\.[a-zA-Z0-9]{2,6}\b', s): score += 0.8
        if re.search(suspects, s): score += 2.5
        ranked.append((s, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:ExperimentConfig.MAX_TOP_STRINGS]]

def extract_extern_imports(code: str) -> List[str]:
    imports: Set[str] = set()
    for line in EXTERN_LINE_RE.findall(code):
        for m in re.finditer(r'\*\s*([A-Za-z_][\w:]*)\s*\)\s*\(', line):
            imports.add(m.group(1))
        for m in re.finditer(r'([A-Za-z_][\w:]*)\s*\(', line):
            imports.add(m.group(1))
    lst = sorted(imports)
    return [_shorten(s, 120) for s in lst[:ExperimentConfig.MAX_API_IMPORTS]]

def _json_size(obj: Dict[str, Any]) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

def compress_metadata_to_budget(meta: Dict[str, Any], tc: TokenCounter,
                                max_tokens: int = 1000, max_bytes: int = 4096) -> Dict[str, Any]:
    m = json.loads(json.dumps(meta, ensure_ascii=False))
    def tok_len() -> int:
        return tc.count(json.dumps(m, ensure_ascii=False))
    if tok_len() > max_tokens:
        if m.get("global_declarations"):
            m["global_declarations"] = [_shorten(x, 120) for x in m["global_declarations"][:ExperimentConfig.MAX_GLOBAL_DECLS]]
    if tok_len() > max_tokens:
        if m.get("functions"):
            m["functions"] = [{"name": f.get("name"), "approx_tokens": f.get("tokens", 0)} for f in m["functions"]]
            m["functions"] = m["functions"][:50]
    if tok_len() > max_tokens:
        if m.get("api_imports"):
            m["api_imports"] = m["api_imports"][:60]
    if tok_len() > max_tokens:
        if m.get("top_strings"):
            m["top_strings"] = m["top_strings"][:80]
    if tok_len() > max_tokens:
        m = {
            "file_size_bytes": m.get("file_size_bytes"),
            "suspicious_indicators": m.get("suspicious_indicators", {}),
            "api_imports": m.get("api_imports", [])[:40],
            "top_strings": m.get("top_strings", [])[:60],
            "global_declarations": m.get("global_declarations", [])[:30],
        }

    if "extraction_stats" in m and _json_size(m) > max_bytes:
        m.pop("extraction_stats", None)

    def shrink_list_field(key: str, step: int):
        if key in m and isinstance(m[key], list) and m[key]:
            m[key] = m[key][:max(0, len(m[key]) - step)]

    while _json_size(m) > max_bytes and any(isinstance(m.get(k), list) and m[k] for k in ("top_strings","api_imports","global_declarations")):
        shrink_list_field("top_strings", 10)
        if _json_size(m) <= max_bytes: break
        shrink_list_field("api_imports", 10)
        if _json_size(m) <= max_bytes: break
        shrink_list_field("global_declarations", 10)

    if _json_size(m) > max_bytes and m.get("top_strings"):
        max_len = ExperimentConfig.MAX_STRING_LEN
        while _json_size(m) > max_bytes and max_len > 40:
            max_len = int(max_len * 0.8)
            m["top_strings"] = [_shorten(s, max_len) for s in m["top_strings"]]

    while _json_size(m) > max_bytes:
        shrink_list_field("top_strings", 5)
        shrink_list_field("api_imports", 5)
        shrink_list_field("global_declarations", 5)
        if not any(isinstance(m.get(k), list) and m[k] for k in ("top_strings","api_imports","global_declarations")):
            break

    if _json_size(m) > max_bytes:
        m = {
            "file_size_bytes": meta.get("file_size_bytes"),
            "suspicious_indicators": meta.get("suspicious_indicators", {}),
            "top_strings": [ _shorten(s, 60) for s in (meta.get("top_strings", [])[:20]) ],
            "api_imports": (meta.get("api_imports", [])[:15]),
        }
    return m

def step_1a_parse_metadata(sample_path: Path, code: str, tc: TokenCounter) -> Dict[str, Any]:
    strings = extract_strings(code)
    top_strings = rank_strings(strings)
    api_imports = extract_extern_imports(code)
    global_decls = [ _shorten(m.group(0).strip(), 120) for m in GLOBAL_DECL_RE.finditer(code) ][:ExperimentConfig.MAX_GLOBAL_DECLS]

    suspicious = {}
    for pat in ["vssadmin", "delete shadows", ".onion", "decrypt", "ransom", "CryptAcquireContext", "CreateFileW",
                "WriteFile", "RegSetValue", "OpenService", "TerminateProcess", "NetShareEnum", "MountPoint", "Mutex"]:
        suspicious[pat] = bool(re.search(pat, code, re.IGNORECASE))

    # Raw extraction (never returned as-is)
    meta_raw = {
        "source_path": str(sample_path),
        "file_size_bytes": sample_path.stat().st_size,
        "extraction_stats": {
            "n_strings_total": len(strings),
            "n_strings_kept": len(top_strings),
            "n_api_imports": len(api_imports),
            "n_global_declarations": len(global_decls),
        },
        "top_strings": top_strings,
        "api_imports": api_imports,
        "global_declarations": global_decls,
        "suspicious_indicators": suspicious,
    }

    # Compress to budget first
    m = compress_metadata_to_budget(meta_raw, tc, max_tokens=1000, max_bytes=ExperimentConfig.MAX_METADATA_BYTES)

    # 1) Drop source_path to avoid label leakage and ensure shape differs from meta_raw.
    m.pop("source_path", None)
    # 2) Ensure structural keys exist even for tiny files (avoid writing back meta_raw-like objects).
    m.setdefault("top_strings", [])
    m.setdefault("api_imports", [])
    m.setdefault("global_declarations", [])
    m.setdefault("suspicious_indicators", suspicious)
    m.setdefault("file_size_bytes", sample_path.stat().st_size)
    # 3) extraction_stats is diagnostic; drop if empty or tiny sample with zeros.
    if not m.get("extraction_stats") or not any(m["extraction_stats"].values()):
        m.pop("extraction_stats", None)

    return m


# =========================
# Step 1B: Call Graph (robust)
# =========================
IDA_SPLITTER_RE = re.compile(r'^\s*//-+\s*\(([0-9A-Fa-fx]+)\)\s*-+.*$', re.MULTILINE)

def strip_strings_and_comments(code: str) -> str:
    code = re.sub(r'L?"(?:\\.|[^"\\])*"', '""', code)
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.DOTALL)
    code = re.sub(r'//.*?$', ' ', code, flags=re.MULTILINE)
    return code

def _extract_func_name_from_signature(sig_line: str) -> str:
    left = sig_line.split('(')[0].strip()
    tokens = [t for t in re.split(r'\s+', left) if t]
    cand = tokens[-1] if tokens else left
    cand = cand.rstrip('*&')
    cand = re.sub(r'@<[^>]+>$', '', cand)  # 去掉 @<eax> 等尾缀
    m = re.search(r'([A-Za-z_~][\w:~<>]*)$', cand)
    return m.group(1) if m else (cand or "func")

def parse_functions_ida_style(code: str, tc: TokenCounter) -> List[Dict[str, Any]]:
    lines = code.splitlines()
    split_lines = [i for i, ln in enumerate(lines) if IDA_SPLITTER_RE.match(ln)]
    segments: List[Tuple[int, int]] = []
    for i, s in enumerate(split_lines):
        e = (split_lines[i+1] - 1) if i + 1 < len(split_lines) else (len(lines) - 1)
        sign_idx = None
        for k in range(s + 1, min(s + 5, e + 1)):
            if '(' in lines[k]:
                sign_idx = k
                break
        if sign_idx is None:
            continue
        segments.append((sign_idx, e))

    funcs: List[Dict[str, Any]] = []
    for idx, (s, e) in enumerate(segments):
        header = lines[s]
        disp_name = _extract_func_name_from_signature(header)
        body = "\n".join(lines[s:e+1])
        funcs.append({
            "id": idx,
            "display_name": disp_name,
            "unique": f"{disp_name}#L{s+1}",
            "line_start": s + 1,
            "line_end": e + 1,
            "code": body,
            "tokens": tc.count(body),
            "content_sha1": sha1_text(body),
        })
    return funcs

def build_call_graph(code: str, tc: TokenCounter) -> Dict[str, Any]:
    nodes = parse_functions_ida_style(code, tc)

    internal_names: Dict[str, List[int]] = {}
    for n in nodes:
        internal_names.setdefault(n["display_name"], []).append(n["id"])

    names_sorted = sorted(internal_names.keys(), key=len, reverse=True)
    if not names_sorted:
        return {"graph_type": "call_graph", "node_count": 0, "edge_count": 0, "nodes": [], "edges": []}

    escaped = [re.escape(x) for x in names_sorted]
    call_pat = re.compile(r'(?<![A-Za-z0-9_:])(?:' + "|".join(escaped) + r')\s*\(')

    edges: Set[Tuple[int, int]] = set()
    for n in nodes:
        src = n["id"]
        body = strip_strings_and_comments(n["code"])
        for m in call_pat.finditer(body):
            snippet = body[m.start():m.end()]
            for name in names_sorted:
                if snippet.startswith(name):
                    for dst in internal_names[name]:
                        if dst != src:
                            edges.add((src, dst))
                    break

    graph = {
        "graph_type": "call_graph",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": [{"src": s, "dst": t} for (s, t) in sorted(edges)],
    }
    return graph


# =========================
# Step 2: Smart Chunking
# =========================
def split_long_function(name: str, code: str, tc: TokenCounter, max_tokens: int) -> List[Dict[str, Any]]:
    toks = tc.count(code)
    if toks <= max_tokens:
        return [{"name": name, "code": code, "tokens": toks, "sha1": sha1_text(code)}]

    parts: List[str] = []
    depth = 0
    cur = []
    for ch in code:
        cur.append(ch)
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth = max(0, depth - 1)
            if depth == 0:
                parts.append("".join(cur)); cur = []
    if cur: parts.append("".join(cur))
    if not parts: parts = [code]

    chunks: List[Dict[str, Any]] = []
    for i, p in enumerate(parts, 1):
        ptok = tc.count(p)
        if ptok <= max_tokens:
            chunks.append({"name": f"{name}#part{i}", "code": p, "tokens": ptok, "sha1": sha1_text(p)})
        else:
            approx_chars = int(max_tokens * 3.8)
            start = 0
            j = 1
            while start < len(p):
                sub = p[start:start+approx_chars]
                if not sub: break
                chunks.append({
                    "name": f"{name}#part{i}.{j}",
                    "code": sub,
                    "tokens": tc.count(sub),
                    "sha1": sha1_text(sub)
                })
                start += approx_chars
                j += 1
    return chunks

def connected_components(n: int, edges: List[Dict[str, int]]) -> List[List[int]]:
    g = [[] for _ in range(n)]
    for e in edges:
        s, t = e["src"], e["dst"]
        g[s].append(t); g[t].append(s)
    vis = [False]*n
    comps: List[List[int]] = []
    for i in range(n):
        if vis[i]: continue
        q = [i]; vis[i] = True; comp = []
        while q:
            u = q.pop(); comp.append(u)
            for v in g[u]:
                if not vis[v]: vis[v] = True; q.append(v)
        comps.append(sorted(comp))
    return comps

def step_2_smart_chunk(graph: Dict[str, Any], tc: TokenCounter, max_tokens: int) -> Dict[str, Any]:
    nodes = graph["nodes"]; edges = graph["edges"]; n = len(nodes)
    comps = connected_components(n, edges)

    blocks: List[Dict[str, Any]] = []
    seen_sha: Set[str] = set()

    def flush(cur):
        if not cur["functions"]: return
        cur["code"] = "\n\n/* --- BLOCK SEPARATOR --- */\n\n".join(f["code"] for f in cur["functions"])
        cur["block_tokens"] = tc.count(cur["code"])
        blocks.append({
            "block_id": cur["block_id"],
            "function_count": len(cur["functions"]),
            "block_tokens": cur["block_tokens"],
            "functions": cur["functions"],
            "code": cur["code"],
        })

    for comp in comps:
        def pri(idx: int) -> Tuple[int, int]:
            nm = nodes[idx]["display_name"].lower()
            p = 1
            if any(x in nm for x in ("wmain", "main", "winmain", "start")): p = 0
            return (p, -nodes[idx]["tokens"])

        order = sorted(comp, key=pri)
        cur = {"block_id": f"block_{len(blocks)+1}", "functions": [], "block_tokens": 0, "code": ""}

        for idx in order:
            f = nodes[idx]
            fname, fcode, ftok, fsha = f["display_name"], f["code"], f["tokens"], f["content_sha1"]

            if ftok > max_tokens:
                parts = split_long_function(fname, fcode, tc, max_tokens)
                for p in parts:
                    if p["sha1"] in seen_sha: continue
                    if cur["block_tokens"] + p["tokens"] <= max_tokens:
                        cur["functions"].append({"name": p["name"], "code": p["code"], "tokens": p["tokens"], "sha1": p["sha1"]})
                        cur["block_tokens"] += p["tokens"]; seen_sha.add(p["sha1"])
                    else:
                        flush(cur)
                        cur = {"block_id": f"block_{len(blocks)+1}", "functions": [], "block_tokens": 0, "code": ""}
                        cur["functions"].append({"name": p["name"], "code": p["code"], "tokens": p["tokens"], "sha1": p["sha1"]})
                        cur["block_tokens"] = p["tokens"]; seen_sha.add(p["sha1"])
                continue

            if fsha in seen_sha: continue
            if cur["block_tokens"] + ftok <= max_tokens:
                cur["functions"].append({"name": fname, "code": fcode, "tokens": ftok, "sha1": fsha})
                cur["block_tokens"] += ftok; seen_sha.add(fsha)
            else:
                flush(cur)
                cur = {"block_id": f"block_{len(blocks)+1}", "functions": [], "block_tokens": 0, "code": ""}
                cur["functions"].append({"name": fname, "code": fcode, "tokens": ftok, "sha1": fsha})
                cur["block_tokens"] = ftok; seen_sha.add(fsha)
        flush(cur)

    changed = True
    while changed and len(blocks) > 1:
        changed = False; i = 0
        while i < len(blocks) - 1:
            a, b = blocks[i], blocks[i+1]
            if a["block_tokens"] + b["block_tokens"] <= int(max_tokens * 0.98):
                merged_funcs = a["functions"] + b["functions"]
                merged_code = "\n\n/* --- BLOCK SEPARATOR --- */\n\n".join(f["code"] for f in merged_funcs)
                blocks[i] = {
                    "block_id": a["block_id"],
                    "function_count": len(merged_funcs),
                    "block_tokens": tc.count(merged_code),
                    "functions": merged_funcs,
                    "code": merged_code
                }
                blocks.pop(i+1); changed = True
            else:
                i += 1

    return {
        "sample_blocks": blocks,
        "total_blocks": len(blocks),
        "max_block_tokens": max((b["block_tokens"] for b in blocks), default=0),
    }


# =========================
# Step 3: Visualization
# =========================
def render_visualization_html(sample_id: str, graph: Dict[str, Any], chunks: Dict[str, Any]) -> str:
    # Map functions to their block ids & collect block token stats
    f2block: Dict[str, str] = {}
    block_tokens: Dict[str, int] = {}
    for b in chunks.get("sample_blocks", []):
        block_tokens[b["block_id"]] = b["block_tokens"]
        for f in b.get("functions", []):
            f2block[f["name"]] = b["block_id"]

    # Build node objects for the front-end
    node_objs = []
    for n in graph.get("nodes", []):
        nm = n["display_name"]
        bl = None
        # best-effort match to a block (keep existing behavior)
        for k in f2block:
            if k.startswith(nm):
                bl = f2block[k]
                break
        node_objs.append({
            "id": n["id"],
            "name": nm,
            "tokens": int(n.get("tokens", 0)),
            "block": bl or "UNASSIGNED",
            "block_tokens": int(block_tokens.get(bl or "UNASSIGNED", 0)),
        })

    edges = graph.get("edges", [])
    html_str = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Call Graph & Chunks — {html.escape(sample_id)}</title>
<style>
  :root {{
    --legend-font: 18px;
    --label-font: 18px;
  }}

  html, body {{
    margin: 0; padding: 0; background: #ffffff; color: #111;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  }}
  h2 {{
    margin: 16px; font-weight: 600;
  }}
  #graph {{
    margin: 0 16px 16px 16px; border: 1px solid #e5e5e5; border-radius: 10px;
  }}

  .node circle {{ stroke: #333; stroke-width: 1px; }}
  .link {{ stroke: #999; stroke-opacity: 0.6; }}

  .label {{
    font-size: var(--label-font);
    font-weight: 500;
    fill: #111;
    pointer-events: none; /* do not affect dragging */
    paint-order: stroke;         /* halo behind fill */
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    stroke: #fff;                /* halo color */
    stroke-width: 3px;           /* halo thickness */
    stroke-linejoin: round;
  }}
  .label.dimmed {{ opacity: 0.25; }}

  .tooltip {{
    position: absolute; background: rgba(0,0,0,0.75); color: #fff; padding: 8px 10px; border-radius: 6px;
    pointer-events: none; font-size: 12px; line-height: 1.4;
  }}

  .legend {{
    position: fixed; right: 16px; top: 16px; background: #f8f8f8; border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px;
    max-width: 360px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    font-size: var(--legend-font);
  }}
  .legend h4 {{ margin: 8px 0 10px 0; font-size: var(--legend-font); line-height: 1.2; }}
  .legend .row {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; }}
  .legend .blocks .row span.swatch {{
    display:inline-block;width:14px;height:14px;margin-right:8px;border:1px solid #333;border-radius:3px;
  }}
  .controls label {{ user-select: none; cursor: pointer; }}

  .pill {{
    display:inline-block; padding: 2px 8px; border-radius: 999px; background:#fff; border:1px solid #ddd; font-size: 12px;
  }}
</style>
</head>
<body>
  <h2>Call Graph & Chunking — {html.escape(sample_id)}</h2>
  <div id="graph"></div>
  <div id="legend" class="legend"></div>
  <div class="tooltip" id="tooltip" style="opacity:0"></div>

  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
  // ---- data ----
  const nodes = {json.dumps(node_objs)};
  const links = {json.dumps([{"src": e["src"], "dst": e["dst"]} for e in edges])}
                 .map(d => ({{source: d.src, target: d.dst}}));

  const blocks = Array.from(new Set(nodes.map(n => n.block)));
  const color = d3.scaleOrdinal().domain(blocks).range(d3.schemeCategory10.concat(d3.schemeTableau10));

  const width = Math.max(960, window.innerWidth - 32);
  const height = Math.max(640, window.innerHeight - 32);

  const svg = d3.select("#graph").append("svg")
    .attr("width", width).attr("height", height);

  // Zoomable container
  const container = svg.append("g");
  let currentTransform = d3.zoomIdentity;

  const tooltip = d3.select("#tooltip");

  // ---- Connected components for rigid group dragging in freeze mode ----
  function computeComponents(nodes, links) {{
    const idx = new Map(nodes.map((n,i)=>[n.id, i]));
    const g = Array(nodes.length).fill(0).map(()=>[]);
    links.forEach(l=>{{
      const s = idx.get(l.source), t = idx.get(l.target);
      if (s!=null && t!=null) {{ g[s].push(t); g[t].push(s); }}
    }});
    const compId = Array(nodes.length).fill(-1);
    let cid = 0;
    for (let i=0;i<nodes.length;i++) {{
      if (compId[i] !== -1) continue;
      const q=[i]; compId[i]=cid;
      while(q.length) {{
        const u=q.pop();
        for (const v of g[u]) if (compId[v]===-1) {{ compId[v]=cid; q.push(v); }}
      }}
      cid++;
    }}
    const map = new Map();
    nodes.forEach((n,i)=> map.set(n.id, compId[i]));
    return map;
  }}
  const compMap = computeComponents(nodes, links);
  nodes.forEach(n => n.comp = compMap.get(n.id));

  // ---- Simulation (can be frozen) ----
  let freeze = false;

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(60).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-150))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collision", d3.forceCollide().radius(d => 10 + Math.log(1 + d.tokens/200)).iterations(2));

  const link = container.append("g").attr("class", "links")
    .selectAll("line").data(links).enter().append("line").attr("class", "link");

  const node = container.append("g").attr("class", "nodes")
    .selectAll("g").data(nodes).enter().append("g")
    .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));

  node.append("circle")
    .attr("r", d => 6 + Math.min(18, Math.log(1 + d.tokens/100)))
    .attr("fill", d => color(d.block));

  node.append("title").text(d => d.name);

  node.on("mouseover", (event, d) => {{
    tooltip.style("opacity", 1)
      .html(`<b>${{d.name}}</b><br/>tokens: ${{d.tokens}}<br/>block: <b>${{d.block}}</b><br/>block tokens: ${{d.block_tokens}}`);
  }}).on("mousemove", (event) => {{
    tooltip.style("left", (event.pageX + 10) + "px").style("top", (event.pageY + 10) + "px");
  }}).on("mouseout", () => tooltip.style("opacity", 0));

  // ---- Labels (font 18px) with halo & declutter ----
  const labels = node.append("text")
    .attr("class", "label")
    .text(d => d.name)
    .attr("x", 10).attr("y", 5);

  // Measure label intrinsic size (local coords) once; recompute on demand if needed
  function measureLabels() {{
    labels.each(function(d) {{
      try {{
        const bb = this.getBBox();
        d._lw = bb.width;  // local width
        d._lh = bb.height; // local height
      }} catch (e) {{
        d._lw = d._lw || 40;
        d._lh = d._lh || 12;
      }}
    }});
  }}
  measureLabels();

  // Greedy declutter in screen space: keep more important labels (by tokens) and dim overlapping ones.
  function updateLabelDeclutter() {{
    const k = currentTransform.k;
    // importance: tokens desc, fallback by name length desc
    const items = labels.nodes().map((el, i) => {{
      const d = d3.select(el).datum();
      const cx = d.x + 10;        // label anchor (local)
      const cy = d.y + 5;
      const p = currentTransform.apply([cx, cy]); // to screen space
      const w = (d._lw || 40) * k;
      const h = (d._lh || 12) * k;
      const r = Math.hypot(w, h) * 0.5; // circle approx
      return {{el, d, x:p[0], y:p[1], r, imp: (d.tokens || 0) * 100 + (d.name?.length || 0)}};
    }});
    items.sort((a,b) => b.imp - a.imp);

    const kept = [];
    const nodeCount = items.length;
    const overlapFactor = 0.92; // <1 means stricter (more dimming)

    for (let i=0; i<nodeCount; i++) {{
      const it = items[i];
      let overlaps = false;
      for (let j=0; j<kept.length; j++) {{
        const kpt = kept[j];
        const dx = it.x - kpt.x, dy = it.y - kpt.y;
        const dist2 = dx*dx + dy*dy;
        const sumr = (it.r + kpt.r) * overlapFactor;
        if (dist2 < sumr*sumr) {{ overlaps = true; break; }}
      }}
      if (!overlaps) kept.push(it);
      d3.select(it.el).classed("dimmed", overlaps);
    }}
  }}

  // Shared render used by ticks and freeze-drag
  function render() {{
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("transform", d => `translate(${{d.x}}, ${{d.y}})`);
    updateLabelDeclutter();
  }}

  simulation.on("tick", render);

  // ---- Zoom / Pan ----
  const zoom = d3.zoom().scaleExtent([0.1, 8]).on("zoom", (event) => {{
    currentTransform = event.transform;
    container.attr("transform", currentTransform);
    // On zoom scale, recompute declutter (sizes change with k)
    updateLabelDeclutter();
  }});
  svg.call(zoom);

  // ---- Drag (freeze supports rigid group move of a connected component) ----
  function dragstarted(event, d) {{
    if (freeze) {{
      d._lastX = d.x; d._lastY = d.y;
      d._group = nodes.filter(n => n.comp === d.comp);
    }} else {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }}
  }}
  function dragged(event, d) {{
    if (freeze) {{
      const dx = event.x - d._lastX, dy = event.y - d._lastY;
      d._lastX = event.x; d._lastY = event.y;
      for (const n of d._group) {{ n.x += dx; n.y += dy; }}
      render(); // keep link shapes/label declutter updated while dragging
    }} else {{
      d.fx = event.x; d.fy = event.y;
    }}
  }}
  function dragended(event, d) {{
    if (freeze) {{
      // nothing; positions already updated
    }} else {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }}
  }}

  // ---- Legend & Controls (font 18px) ----
  const legend = d3.select("#legend");

  legend.append("h4").text("Controls");
  const controls = legend.append("div").attr("class", "controls");
  controls.append("div").attr("class","row")
    .html(`<label><input type="checkbox" id="toggleLabels" checked> Show function names</label>`);
  controls.append("div").attr("class","row")
    .html(`<label><input type="checkbox" id="toggleFreeze"> Freeze layout (disable forces)</label>`);

  legend.append("h4").text("Blocks");
  const blockList = legend.append("div").attr("class","blocks");
  blocks.forEach(b => {{
    const item = blockList.append("div").attr("class","row");
    item.html(`<span class="swatch" style="background:${{color(b)}};"></span><span>${{b}}</span>`);
  }});

  // Graph element legend (under block legend)
  legend.append("h4").text("Graph legend");
  const ge = legend.append("div").attr("class","graph-legend");
  ge.append("div").attr("class","row").html(`
    <svg width="28" height="18" style="flex:0 0 auto;"><circle cx="9" cy="9" r="6" fill="#000"/></svg>
    <span>Function (node)</span>
  `);
  ge.append("div").attr("class","row").html(`
    <svg width="28" height="18" style="flex:0 0 auto;"><line x1="3" y1="9" x2="25" y2="9" stroke="#333" stroke-width="2"/></svg>
    <span>Call (edge)</span>
  `);

  // ---- Controls wiring ----
  document.getElementById("toggleLabels").addEventListener("change", (e) => {{
    const display = e.target.checked ? null : "none";
    labels.style("display", display);
    // when turning on, recompute metrics & declutter
    if (e.target.checked) {{
      measureLabels();
      updateLabelDeclutter();
    }}
  }});

  document.getElementById("toggleFreeze").addEventListener("change", (e) => {{
    freeze = !!e.target.checked;
    if (freeze) {{
      simulation.stop(); // keep current link shapes; no forces
    }} else {{
      simulation.alpha(0.5).restart(); // resume layout
    }}
  }});

  // Initial declutter after first positions settle a bit
  setTimeout(() => updateLabelDeclutter(), 150);

  // On window resize, keep canvas size but avoid recomputing layout (positions remain);
  // you can adapt if you want responsive sizing.
  </script>
</body>
</html>"""
    return html_str



# =========================
# Step 4 & 5: Summarization
# =========================
def summarize_blocks(llm: LLMInterface, chunks: Dict[str, Any], save_inputs_to: Optional[Path] = None) -> List[Dict[str, Any]]:
    prompts_log: List[str] = []
    out = []
    for blk in tqdm(chunks.get("sample_blocks", []), desc="LLM chunk summaries"):
        if blk.get("functions"):
            pure_code = "\n\n".join(f["code"] for f in blk["functions"])
        else:
            pure_code = blk.get("code", "").replace("/* --- BLOCK SEPARATOR --- */", "\n\n")
        prompt = ExperimentConfig.PROMPT_CHUNK_SUMMARY.format(code_chunk=pure_code)
        prompts_log.append(f"### {blk['block_id']}\n{prompt}\n")
        summary = ask_llm(llm, prompt)
        out.append({
            "block_id": blk["block_id"],
            "block_tokens": blk["block_tokens"],
            "summary": summary.strip()
        })

    if save_inputs_to is not None:
        try:
            save_inputs_to.parent.mkdir(parents=True, exist_ok=True)
            save_inputs_to.write_text("\n\n".join(prompts_log), encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to save LLM inputs for chunk summaries: {e}")

    return out

def summarize_program(llm: LLMInterface, block_summaries: List[Dict[str, Any]], save_input_to: Optional[Path] = None) -> str:
    joined = "\n\n---\n\n".join(f"[{bs['block_id']}] {bs['summary']}" for bs in block_summaries)
    prompt = ExperimentConfig.PROMPT_PROGRAM_SUMMARY.format(combined_summaries=joined)
    if save_input_to is not None:
        try:
            save_input_to.parent.mkdir(parents=True, exist_ok=True)
            save_input_to.write_text(prompt, encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to save LLM input for program summary: {e}")
    return ask_llm(llm, prompt).strip()


# =========================
# Step 6: IR Construction
# =========================
def build_final_ir(global_meta: Dict[str, Any], program_summary_text: str) -> Dict[str, Any]:
    return {
        "global_metadata": global_meta,
        "program_summary": {"text": program_summary_text}
    }


# =========================
# Step 7: Classification helpers (unchanged prompt builder)
# =========================
def sanitize_ir_for_llm(ir_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in ir_obj.items() if k in {"global_metadata", "program_summary"}}

def build_classification_prompt(ir_obj: Dict[str, Any]) -> str:
    safe_ir = sanitize_ir_for_llm(ir_obj)
    ir_json_block = json.dumps(safe_ir, ensure_ascii=False, indent=2)
    return ExperimentConfig.PROMPT_CLASSIFICATION.format(ir_json_block=ir_json_block)

def parse_classification_output(out: str) -> Tuple[str, str]:
    label = "UNKNOWN"
    justification = out.strip()
    m = re.match(r'^\s*\[(RANSOMWARE|BENIGN)\]\s*(.*)$', out.strip(), re.IGNORECASE | re.DOTALL)
    if m:
        label = m.group(1).upper()
        justification = m.group(2).strip()
    return label, justification


# =========================
# Step 8: Label Rationale (NEW)
# =========================
def build_rationale_prompt(ir_obj: Dict[str, Any], label: str) -> str:
    safe_ir = sanitize_ir_for_llm(ir_obj)
    ir_json_block = json.dumps(safe_ir, ensure_ascii=False, indent=2)
    return ExperimentConfig.PROMPT_CLASSIFICATION_RATIONALE.format(
        label=f"[{label}]",
        ir_json_block=ir_json_block
    )


# =========================
# BFS Orchestrator
# =========================
def main():
    setup_logging()

    input_dir = ExperimentConfig.INPUT_DIR
    if not input_dir.exists():
        logging.error(f"INPUT_DIR not found: {input_dir}")
        return

    files = [p for p in input_dir.rglob("*.c") if p.is_file()]
    files.sort(key=lambda p: natural_key(p.name))
    if not files:
        logging.warning(f"No .c files found under {input_dir}")
        return

    base_llm = None
    tc = TokenCounter(None)

    try:
        # Always load base model first so earlier stages use its tokenizer.
        base_llm = LLMInterface(ModelConfig())
        tc = TokenCounter(base_llm)  # use real tokenizer for token budgeting
    except Exception as e:
        logging.exception(f"Failed to initialize base model: {e}")
        return

    # -----------------------
    # Step 1A
    # -----------------------
    if ExperimentConfig.RUN_STEP_1A_METADATA_EXTRACTION:
        for p in tqdm(files, desc="Step 1A: metadata"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["metadata"].exists():
                continue
            try:
                code = read_text(p)
                meta = step_1a_parse_metadata(p, code, tc)
                meta_bytes = len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
                if meta_bytes > ExperimentConfig.MAX_METADATA_BYTES:
                    logging.warning(f"[{p.stem}] metadata {meta_bytes}B over budget; shrinking.")
                    meta = compress_metadata_to_budget(meta, tc, max_tokens=1000, max_bytes=ExperimentConfig.MAX_METADATA_BYTES)
                    meta.pop("source_path", None)
                write_json(paths["metadata"], meta)
            except Exception as e:
                logging.exception(f"Step 1A failed for {p.name}: {e}")

    # -----------------------
    # Step 1B: call graph
    # -----------------------
    if ExperimentConfig.RUN_STEP_1B_CALL_GRAPH_GENERATION:
        for p in tqdm(files, desc="Step 1B: call graph"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["callgraph"].exists():
                continue
            try:
                code = read_text(p)
                cg = build_call_graph(code, tc)
                write_json(paths["callgraph"], cg)
            except Exception as e:
                logging.exception(f"Step 1B failed for {p.name}: {e}")

    # -----------------------
    # Step 2: smart chunks
    # -----------------------
    if ExperimentConfig.RUN_STEP_2_SMART_CHUNKING:
        for p in tqdm(files, desc="Step 2: smart chunking"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["chunks"].exists():
                continue
            try:
                cg = read_json(paths["callgraph"])
                chunks = step_2_smart_chunk(cg, tc, ExperimentConfig.CHUNK_TOKEN_SIZE)
                write_json(paths["chunks"], chunks)
            except Exception as e:
                logging.exception(f"Step 2 failed for {p.name}: {e}")

    # -----------------------
    # Step 3: visualization
    # -----------------------
    if ExperimentConfig.RUN_STEP_3_VISUALIZATIONS:
        for p in tqdm(files, desc="Step 3: visualization"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["viz_html"].exists():
                continue
            try:
                cg = read_json(paths["callgraph"])
                chunks = read_json(paths["chunks"])
                html_str = render_visualization_html(p.stem, cg, chunks)
                write_html(paths["viz_html"], html_str)
            except Exception as e:
                logging.exception(f"Step 3 failed for {p.name}: {e}")

    # -----------------------
    # Step 4: chunk summaries
    # -----------------------
    if ExperimentConfig.RUN_STEP_4_CHUNK_SUMMARIZATION:
        if base_llm is None:
            logging.error("Base LLM required for Step 4.")
            return
        for p in tqdm(files, desc="Step 4: chunk summaries"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["chunk_summaries"].exists():
                continue
            try:
                chunks = read_json(paths["chunks"])
                save_txt = llm_input_path_for(paths["chunk_summaries"]) if ExperimentConfig.SAVE_LLM_INPUTS else None
                block_summaries = summarize_blocks(base_llm, chunks, save_inputs_to=save_txt)
                write_json(paths["chunk_summaries"], {"block_summaries": block_summaries})
            except Exception as e:
                logging.exception(f"Step 4 failed for {p.name}: {e}")

    # -----------------------
    # Step 5: program summary
    # -----------------------
    if ExperimentConfig.RUN_STEP_5_PROGRAM_SUMMARIZATION:
        if base_llm is None:
            logging.error("Base LLM required for Step 5.")
            return
        for p in tqdm(files, desc="Step 5: program summary"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["program_summary"].exists():
                continue
            try:
                block_summaries = read_json(paths["chunk_summaries"]).get("block_summaries", [])
                save_txt = llm_input_path_for(paths["program_summary"]) if ExperimentConfig.SAVE_LLM_INPUTS else None
                program_summary = summarize_program(base_llm, block_summaries, save_input_to=save_txt)
                write_json(paths["program_summary"], {"program_summary": program_summary})
            except Exception as e:
                logging.exception(f"Step 5 failed for {p.name}: {e}")

    # -----------------------
    # Step 6: IR construction
    # -----------------------
    if ExperimentConfig.RUN_STEP_6_IR_CONSTRUCTION:
        for p in tqdm(files, desc="Step 6: IR construction"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["ir"].exists():
                continue
            try:
                meta = read_json(paths["metadata"])
                program_summary = read_json(paths["program_summary"]).get("program_summary", "")
                ir_obj = build_final_ir(meta, program_summary)
                write_json(paths["ir"], ir_obj)
            except Exception as e:
                logging.exception(f"Step 6 failed for {p.name}: {e}")

    # -----------------------
    # Step 7: classification
    # -----------------------
    if ExperimentConfig.RUN_STEP_7_CLASSIFICATION:
        # Decide which model to use for Step 7
        use_lora = ExperimentConfig.USE_LORA_FOR_CLASSIFICATION

        if use_lora:
            # Unload BASE to save VRAM before loading LoRA
            try:
                if base_llm is not None:
                    base_llm.unload()
                    base_llm = None
            except Exception as e:
                logging.warning(f"Failed to unload base before LoRA: {e}")

            # Load LoRA model only for Step 7
            try:
                clf_llm = LoRALLMInterface(ModelConfig())
            except Exception as e:
                logging.exception(f"Failed to initialize LoRA model for classification: {e}")
                return
        else:
            # Use already-loaded BASE model
            clf_llm = base_llm
            if clf_llm is None:
                logging.error("Base model not available for classification.")
                return

        for p in tqdm(files, desc="Step 7: classification"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["classification"].exists():
                continue
            try:
                ir_obj = read_json(paths["ir"])
                prompts_log: List[str] = []
                results = {"variants": {}}

                # Full IR
                prompt_full = build_classification_prompt(ir_obj)
                out_full = ask_llm(clf_llm, prompt_full)
                lab, just = parse_classification_output(out_full)
                results["variants"]["full"] = {"prediction": lab, "justification": just, "raw_output": out_full}
                prompts_log.append(f"### FULL\n{prompt_full}\n")

                # Ablations (use the same classifier)
                if ExperimentConfig.RUN_ABLATION_E_META:
                    e_meta = {"program_summary": ir_obj.get("program_summary")}
                    prompt_e_meta = build_classification_prompt(e_meta)
                    out_e_meta = ask_llm(clf_llm, prompt_e_meta)
                    lab_m, just_m = parse_classification_output(out_e_meta)
                    results["variants"]["E-Meta"] = {"prediction": lab_m, "justification": just_m, "raw_output": out_e_meta}
                    prompts_log.append(f"### E-META\n{prompt_e_meta}\n")

                if ExperimentConfig.RUN_ABLATION_E_SUMMARY:
                    e_sum = {"global_metadata": ir_obj.get("global_metadata")}
                    prompt_e_sum = build_classification_prompt(e_sum)
                    out_e_sum = ask_llm(clf_llm, prompt_e_sum)
                    lab_s, just_s = parse_classification_output(out_e_sum)
                    results["variants"]["E-Summary"] = {"prediction": lab_s, "justification": just_s, "raw_output": out_e_sum}
                    prompts_log.append(f"### E-SUMMARY\n{prompt_e_sum}\n")

                write_json(paths["classification"], results)

                if ExperimentConfig.SAVE_LLM_INPUTS:
                    try:
                        save_txt = llm_input_path_for(paths["classification"])
                        save_txt.parent.mkdir(parents=True, exist_ok=True)
                        save_txt.write_text("\n\n".join(prompts_log), encoding="utf-8")
                    except Exception as e:
                        logging.warning(f"Failed to save LLM inputs for classification: {e}")

            except Exception as e:
                logging.exception(f"Step 7 failed for {p.name}: {e}")

        # If we loaded LoRA, unload it after step 7
        if use_lora:
            try:
                clf_llm.unload()
            except Exception as e:
                logging.warning(f"Failed to unload LoRA after classification: {e}")

    # -----------------------
    # Step 8: rationale
    # -----------------------
    if ExperimentConfig.RUN_STEP_8_CLASSIFICATION_RATIONALE:
        # Ensure BASE is loaded now (this may be the third load in LoRA case)
        if base_llm is None:
            try:
                base_llm = LLMInterface(ModelConfig())
            except Exception as e:
                logging.exception(f"Failed to (re)load base model for rationale: {e}")
                return

        for p in tqdm(files, desc="Step 8: rationale"):
            paths = out_paths(p)
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["rationale"].exists():
                continue
            try:
                ir_obj = read_json(paths["ir"])
                clf = read_json(paths["classification"])
                full = clf.get("variants", {}).get("full", {})
                label = full.get("prediction", "UNKNOWN")

                prompt = build_rationale_prompt(ir_obj, label)
                if ExperimentConfig.SAVE_LLM_INPUTS:
                    try:
                        save_txt = llm_input_path_for(paths["rationale"])
                        save_txt.parent.mkdir(parents=True, exist_ok=True)
                        save_txt.write_text(prompt, encoding="utf-8")
                    except Exception as e:
                        logging.warning(f"Failed to save LLM input for rationale: {e}")

                rationale = ask_llm(base_llm, prompt).strip()
                write_json(paths["rationale"], {"label": label, "rationale": rationale})
            except Exception as e:
                logging.exception(f"Step 8 failed for {p.name}: {e}")

    logging.info("All done (BFS pipeline with single active model).")


if __name__ == "__main__":
    main()
