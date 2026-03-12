from src.scaling.speculative_decoding import generate_fast

class HDLInterpreterAgent:
    def analyze(self, hdl_code):
        return generate_fast(hdl_code)
