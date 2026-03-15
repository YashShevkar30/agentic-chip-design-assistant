"""Agentic HDL interpreter for chip design analysis."""
import re
import logging
from typing import Optional
from src.retrieval.vector_store import FAISSStore
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

class HDLInterpreterAgent:
    """AI agent that interprets HDL code and suggests optimizations."""

    OPTIMIZATION_RULES = {
        "blocking_assignment": {
            "pattern": r"\b\w+\s*=\s*",
            "suggestion": "Consider using non-blocking assignments (<=) in sequential always blocks",
            "severity": "warning",
        },
        "incomplete_sensitivity": {
            "pattern": r"always\s*@\s*\(",
            "suggestion": "Use always @(*) or always_comb for combinational logic",
            "severity": "info",
        },
        "large_mux": {
            "pattern": r"case\s*\(",
            "suggestion": "Large case statements may benefit from priority encoding",
            "severity": "optimization",
        },
        "clock_domain": {
            "pattern": r"posedge\s+\w+",
            "suggestion": "Verify clock domain crossings with synchronizers",
            "severity": "critical",
        },
    }

    def __init__(self, vector_store: Optional[FAISSStore] = None, reranker: Optional[CrossEncoderReranker] = None):
        self.vector_store = vector_store
        self.reranker = reranker or CrossEncoderReranker()

    def analyze(self, hdl_code: str) -> dict:
        issues = self._detect_issues(hdl_code)
        modules = self._parse_modules(hdl_code)
        complexity = self._estimate_complexity(hdl_code)
        suggestions = self._generate_suggestions(hdl_code, issues)

        return {
            "modules_found": modules,
            "issues": issues,
            "complexity_score": complexity,
            "suggestions": suggestions,
            "summary": f"Found {len(modules)} module(s) with {len(issues)} potential issue(s)",
        }

    def _detect_issues(self, code: str) -> list[dict]:
        issues = []
        for rule_name, rule in self.OPTIMIZATION_RULES.items():
            matches = re.findall(rule["pattern"], code)
            if matches:
                issues.append({
                    "rule": rule_name,
                    "count": len(matches),
                    "severity": rule["severity"],
                    "suggestion": rule["suggestion"],
                })
        return issues

    def _parse_modules(self, code: str) -> list[dict]:
        modules = []
        module_pattern = r"module\s+(\w+)\s*(?:#\s*\(([^)]*)\))?\s*\(([^)]*)\)"
        for match in re.finditer(module_pattern, code):
            name = match.group(1)
            params = match.group(2) or ""
            ports = match.group(3) or ""
            port_list = [p.strip() for p in ports.split(",") if p.strip()]
            modules.append({
                "name": name,
                "port_count": len(port_list),
                "has_parameters": bool(params),
                "ports": port_list[:10],
            })
        return modules

    def _estimate_complexity(self, code: str) -> dict:
        lines = code.strip().split("\n")
        return {
            "total_lines": len(lines),
            "always_blocks": len(re.findall(r"always\s", code)),
            "assign_statements": len(re.findall(r"\bassign\b", code)),
            "register_count": len(re.findall(r"\breg\b", code)),
            "wire_count": len(re.findall(r"\bwire\b", code)),
            "instantiations": len(re.findall(r"\b\w+\s+\w+\s*\(", code)),
        }

    def _generate_suggestions(self, code: str, issues: list) -> list[str]:
        suggestions = []
        for issue in issues:
            if issue["severity"] == "critical":
                suggestions.insert(0, f"[CRITICAL] {issue['suggestion']}")
            elif issue["severity"] == "warning":
                suggestions.append(f"[WARNING] {issue['suggestion']}")
            else:
                suggestions.append(f"[INFO] {issue['suggestion']}")
        if not suggestions:
            suggestions.append("No significant issues detected. Code follows best practices.")
        return suggestions
