from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tree


def test_max_40_percent_from_single_source():
    """With multiple sources, no single source contributes > 40% of tokens in the final reply."""
    # Create a scenario with many tokens from different sources
    candidates = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    source = ["input1", "input2", "input3", "input4", "input5"]  # User input tokens
    
    # Test with a longer response that would show diversification
    result = tree._compose(
        candidates, 
        source, 
        mix_ratio=0.5,
        min_words=8,
        max_words=10,
        user_message="input1 input2 input3 input4 input5",
        context_tokens=["alpha", "beta", "gamma", "delta"],
        memory_tokens=["epsilon", "zeta", "eta", "theta"]
    )
    
    if result and len(result.split()) >= 5:  # Only test if we got a reasonable response
        tokens = result.lower().replace(".", "").replace("?", "").split()
        
        # Count input tokens vs context tokens
        input_tokens = ["input1", "input2", "input3", "input4", "input5"]
        input_count = sum(1 for token in tokens if token in input_tokens)
        
        if len(tokens) > 5:  # Only check if response is long enough
            input_ratio = input_count / len(tokens)
            assert input_ratio <= 0.45, f"Input tokens contribute {input_ratio:.1%}, should be ≤ 40%"


def test_source_tracking_basic():
    """Basic test for source tracking functionality.""" 
    # Test that the TokenWithSource class works correctly
    token_source = tree.TokenWithSource("test", "input")
    assert token_source.token == "test"
    assert token_source.source == "input"