"""Tests for HDL interpreter agent."""
import pytest
from src.agent.hdl_interpreter import HDLInterpreterAgent
from src.agent.tools import syntax_checker, port_analyzer, timing_estimator

SAMPLE_VERILOG = """
module counter #(parameter WIDTH = 8) (
    input wire clk,
    input wire rst,
    output reg [WIDTH-1:0] count
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 0;
        else
            count <= count + 1;
    end
endmodule
"""

def test_analyze_modules():
    agent = HDLInterpreterAgent()
    result = agent.analyze(SAMPLE_VERILOG)
    assert len(result["modules_found"]) >= 1
    assert result["modules_found"][0]["name"] == "counter"

def test_detect_clock_domain():
    agent = HDLInterpreterAgent()
    result = agent.analyze(SAMPLE_VERILOG)
    clock_issues = [i for i in result["issues"] if i["rule"] == "clock_domain"]
    assert len(clock_issues) >= 1

def test_syntax_checker():
    result = syntax_checker(SAMPLE_VERILOG)
    assert result["module_count"] == 1

def test_port_analyzer():
    result = port_analyzer(SAMPLE_VERILOG)
    assert result["total_ports"] >= 2

def test_timing_estimator():
    result = timing_estimator(SAMPLE_VERILOG)
    assert "estimated_logic_levels" in result
    assert result["risk"] in ["low", "medium", "high"]
