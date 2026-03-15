"""Agent tools for HDL analysis."""
import re
from typing import Optional

def syntax_checker(code: str) -> dict:
    """Check HDL code for common syntax issues."""
    errors = []
    lines = code.strip().split("\n")
    module_count = code.count("module ")
    endmodule_count = code.count("endmodule")
    if module_count != endmodule_count:
        errors.append(f"Mismatched module/endmodule: {module_count} vs {endmodule_count}")

    begin_count = code.count("begin")
    end_count = len(re.findall(r"\bend\b", code)) - endmodule_count
    if begin_count != end_count:
        errors.append(f"Mismatched begin/end blocks: {begin_count} vs {end_count}")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
            if any(kw in stripped for kw in ["assign", "wire", "reg", "input", "output"]):
                if not stripped.endswith(";") and not stripped.endswith(","):
                    errors.append(f"Line {i+1}: Possible missing semicolon")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "lines_checked": len(lines),
        "module_count": module_count,
    }

def port_analyzer(code: str) -> dict:
    """Analyze module ports and signal widths."""
    inputs = re.findall(r"input\s+(?:\[([\d:]+)\]\s*)?(\w+)", code)
    outputs = re.findall(r"output\s+(?:\[([\d:]+)\]\s*)?(\w+)", code)
    return {
        "inputs": [{"name": name, "width": width or "1"} for width, name in inputs],
        "outputs": [{"name": name, "width": width or "1"} for width, name in outputs],
        "total_ports": len(inputs) + len(outputs),
    }

def timing_estimator(code: str) -> dict:
    """Estimate critical path depth based on combinational logic."""
    assign_chains = len(re.findall(r"assign\b", code))
    always_comb = len(re.findall(r"always_comb|always\s*@\s*\(\*\)", code))
    mux_depth = len(re.findall(r"\?.*:", code))
    estimated_levels = assign_chains + mux_depth + always_comb * 2
    return {
        "estimated_logic_levels": estimated_levels,
        "combinational_blocks": always_comb,
        "mux_count": mux_depth,
        "risk": "high" if estimated_levels > 20 else "medium" if estimated_levels > 10 else "low",
    }
