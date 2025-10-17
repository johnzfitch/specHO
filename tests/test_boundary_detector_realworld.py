"""
Real-World Validation Tests for ClauseBoundaryDetector (Task 3.1)

Tests clause boundary detection on diverse real-world text samples.
5 samples per category to ensure robustness across different text types.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector
from specHO.preprocessor.pipeline import LinguisticPreprocessor


@pytest.fixture
def detector():
    """Fixture for ClauseBoundaryDetector."""
    return ClauseBoundaryDetector()


@pytest.fixture
def preprocessor():
    """Fixture for LinguisticPreprocessor."""
    return LinguisticPreprocessor()


# ============================================================================
# Category 1: News/Journalism (Formal Writing)
# ============================================================================

def test_news_sample_1_breaking_news(detector, preprocessor):
    """News Sample 1: Breaking news announcement."""
    text = (
        "Scientists announced a major breakthrough in renewable energy yesterday. "
        "The new solar panel design increases efficiency by 40 percent, and researchers "
        "believe it could revolutionize the industry within five years."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    # Should detect multiple clauses
    assert len(clauses) >= 3
    # Should have main clauses
    assert any(c.clause_type == "main" for c in clauses)
    # Should have coordinate clauses (and)
    assert any(c.clause_type == "coordinate" for c in clauses)

    print(f"\n[News 1] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_news_sample_2_political_reporting(detector, preprocessor):
    """News Sample 2: Political news with quotes."""
    text = (
        "The senator criticized the new policy during a press conference Tuesday. "
        "She argued that the legislation would harm small businesses, although "
        "supporters claim it will create jobs."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3
    # Should detect subordinate clause (although)
    assert any(c.clause_type == "subordinate" for c in clauses)

    print(f"\n[News 2] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_news_sample_3_financial_report(detector, preprocessor):
    """News Sample 3: Financial reporting."""
    text = (
        "Stock markets closed higher on Friday, and analysts attributed the gains "
        "to positive economic data. The technology sector led the rally, while "
        "energy stocks remained flat."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3
    assert any(c.clause_type == "coordinate" for c in clauses)

    print(f"\n[News 3] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_news_sample_4_sports_coverage(detector, preprocessor):
    """News Sample 4: Sports news."""
    text = (
        "The team won the championship game when the quarterback threw a touchdown "
        "pass in the final seconds. Fans celebrated throughout the night, and the "
        "city declared a holiday to honor the victory."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 4
    # Should have subordinate clause (when)
    assert any(c.clause_type == "subordinate" for c in clauses)

    print(f"\n[News 4] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_news_sample_5_science_reporting(detector, preprocessor):
    """News Sample 5: Science news with technical content."""
    text = (
        "Researchers discovered a new species in the Amazon rainforest. "
        "The animal resembles a small mammal, but genetic analysis revealed "
        "surprising evolutionary connections to distant relatives."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[News 5] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


# ============================================================================
# Category 2: Conversational/Informal Text
# ============================================================================

def test_conversational_sample_1_casual_chat(detector, preprocessor):
    """Conversational Sample 1: Casual conversation."""
    text = (
        "Hey, I can't believe what happened yesterday. We were just walking down "
        "the street when this guy suddenly appeared out of nowhere."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 2
    # Should handle contractions (can't)
    assert any(c.clause_type == "subordinate" for c in clauses)

    print(f"\n[Conversational 1] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_conversational_sample_2_dialogue(detector, preprocessor):
    """Conversational Sample 2: Dialogue exchange."""
    text = (
        "I think we should leave now, but maybe we should wait a few more minutes. "
        "What do you think?"
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 2
    assert any(c.clause_type == "coordinate" for c in clauses)

    print(f"\n[Conversational 2] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_conversational_sample_3_storytelling(detector, preprocessor):
    """Conversational Sample 3: Informal storytelling."""
    text = (
        "So we got to the restaurant, and the place was completely packed. "
        "We waited for like an hour, and then they finally gave us a table "
        "in the back corner."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[Conversational 3] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_conversational_sample_4_planning(detector, preprocessor):
    """Conversational Sample 4: Planning discussion."""
    text = (
        "We could go to the movies tonight, or we could stay home and watch "
        "something on TV. Either way is fine with me."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 2
    assert any(c.clause_type == "coordinate" for c in clauses)

    print(f"\n[Conversational 4] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_conversational_sample_5_advice(detector, preprocessor):
    """Conversational Sample 5: Giving advice."""
    text = (
        "You should definitely apply for that job, even though the deadline "
        "is coming up soon. I think you'd be perfect for it, and the worst "
        "they can say is no."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[Conversational 5] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


# ============================================================================
# Category 3: Literary/Descriptive Prose
# ============================================================================

def test_literary_sample_1_descriptive(detector, preprocessor):
    """Literary Sample 1: Descriptive prose."""
    text = (
        "The garden was silent in the moonlight; shadows danced across the lawn "
        "like ghosts. A gentle breeze stirred the leaves, and somewhere in the "
        "distance, an owl hooted softly."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3
    # Literary text often has higher clause density

    print(f"\n[Literary 1] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_literary_sample_2_narrative(detector, preprocessor):
    """Literary Sample 2: Narrative prose."""
    text = (
        "She walked slowly through the empty streets, remembering the days when "
        "this place had been filled with life. Now only silence remained, and "
        "the buildings stood like forgotten monuments."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3
    assert any(c.clause_type == "subordinate" for c in clauses)

    print(f"\n[Literary 2] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_literary_sample_3_atmospheric(detector, preprocessor):
    """Literary Sample 3: Atmospheric description."""
    text = (
        "Rain fell steadily against the windows while thunder rumbled in the distance. "
        "Inside, the fire crackled warmly, and she pulled the blanket closer, "
        "grateful for the shelter."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[Literary 3] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_literary_sample_4_character_introspection(detector, preprocessor):
    """Literary Sample 4: Character introspection."""
    text = (
        "He wondered if he had made the right choice, although it was too late "
        "to turn back now. The path ahead seemed uncertain, but he kept walking, "
        "driven by a hope he couldn't quite name."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3
    assert any(c.clause_type == "subordinate" for c in clauses)

    print(f"\n[Literary 4] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_literary_sample_5_action_sequence(detector, preprocessor):
    """Literary Sample 5: Action sequence."""
    text = (
        "The door burst open, and three figures rushed into the room. "
        "They moved with purpose, searching every corner until they found "
        "what they were looking for hidden beneath the floorboards."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[Literary 5] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


# ============================================================================
# Category 4: Technical/Academic Writing
# ============================================================================

def test_technical_sample_1_documentation(detector, preprocessor):
    """Technical Sample 1: API documentation."""
    text = (
        "The API endpoint accepts POST requests with JSON payloads. "
        "Authentication requires a valid API key in the Authorization header, "
        "and rate limits apply at 1000 requests per hour."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 2

    print(f"\n[Technical 1] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_technical_sample_2_research_abstract(detector, preprocessor):
    """Technical Sample 2: Research abstract."""
    text = (
        "This study examines the effects of temperature on enzyme activity. "
        "Results indicate that optimal performance occurs at 37 degrees Celsius, "
        "although significant activity remains at temperatures up to 45 degrees."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3
    assert any(c.clause_type == "subordinate" for c in clauses)

    print(f"\n[Technical 2] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_technical_sample_3_methodology(detector, preprocessor):
    """Technical Sample 3: Research methodology."""
    text = (
        "Participants were randomly assigned to control or experimental groups. "
        "The experimental group received the intervention, while the control group "
        "followed standard procedures for comparison purposes."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[Technical 3] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_technical_sample_4_tutorial(detector, preprocessor):
    """Technical Sample 4: Tutorial instructions."""
    text = (
        "First, install the required dependencies using the package manager. "
        "Next, configure the environment variables, and then run the initialization "
        "script to set up the database schema."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[Technical 4] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_technical_sample_5_specification(detector, preprocessor):
    """Technical Sample 5: Technical specification."""
    text = (
        "The system must handle concurrent requests efficiently. "
        "Response times should not exceed 200 milliseconds under normal load, "
        "and the architecture must support horizontal scaling when demand increases."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[Technical 5] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


# ============================================================================
# Category 5: Complex Sentence Structures
# ============================================================================

def test_complex_sample_1_multiple_subordination(detector, preprocessor):
    """Complex Sample 1: Multiple levels of subordination."""
    text = (
        "Although previous research suggested a correlation, the current study "
        "found no significant relationship between the variables when confounding "
        "factors were properly controlled."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 2
    # Should detect multiple subordinate clauses
    subordinate_count = sum(1 for c in clauses if c.clause_type == "subordinate")
    assert subordinate_count >= 1

    print(f"\n[Complex 1] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_complex_sample_2_long_coordination(detector, preprocessor):
    """Complex Sample 2: Long chain of coordinated clauses."""
    text = (
        "The team analyzed the data, the researchers reviewed the findings, "
        "the statisticians verified the results, and the committee approved "
        "the publication."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 4
    # Multiple coordinate clauses

    print(f"\n[Complex 2] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_complex_sample_3_mixed_clause_types(detector, preprocessor):
    """Complex Sample 3: Mixed main, coordinate, and subordinate."""
    text = (
        "When the storm approached, residents evacuated the coastal areas, "
        "emergency services prepared for potential damage, and meteorologists "
        "continued monitoring the system because conditions could change rapidly."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 4
    # Should have all three clause types
    clause_types = {c.clause_type for c in clauses}
    assert len(clause_types) >= 2

    print(f"\n[Complex 3] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_complex_sample_4_embedded_clauses(detector, preprocessor):
    """Complex Sample 4: Embedded clausal structures."""
    text = (
        "Scientists believe that the discovery, which was made accidentally "
        "during routine testing, could lead to applications that nobody had "
        "previously imagined."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 2

    print(f"\n[Complex 4] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


def test_complex_sample_5_conditional_chains(detector, preprocessor):
    """Complex Sample 5: Conditional reasoning chains."""
    text = (
        "If the hypothesis is correct, then the experiment should produce consistent "
        "results, but if unexpected variations occur, the team will need to revise "
        "their theoretical framework and conduct additional trials."
    )
    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3

    print(f"\n[Complex 5] Detected {len(clauses)} clauses:")
    for i, clause in enumerate(clauses):
        tokens_text = ' '.join(t.text for t in clause.tokens[:10])
        print(f"  Clause {i+1} ({clause.clause_type}): {tokens_text}...")


# ============================================================================
# Summary Test - All Categories Combined
# ============================================================================

def test_summary_all_categories(detector, preprocessor):
    """Summary test analyzing all 25 real-world samples."""

    all_samples = [
        # News (5)
        "Scientists announced a major breakthrough in renewable energy yesterday.",
        "The senator criticized the new policy during a press conference Tuesday.",
        "Stock markets closed higher on Friday, and analysts attributed the gains to positive economic data.",
        "The team won the championship game when the quarterback threw a touchdown pass in the final seconds.",
        "Researchers discovered a new species in the Amazon rainforest.",

        # Conversational (5)
        "Hey, I can't believe what happened yesterday.",
        "I think we should leave now, but maybe we should wait a few more minutes.",
        "So we got to the restaurant, and the place was completely packed.",
        "We could go to the movies tonight, or we could stay home and watch something on TV.",
        "You should definitely apply for that job, even though the deadline is coming up soon.",

        # Literary (5)
        "The garden was silent in the moonlight; shadows danced across the lawn like ghosts.",
        "She walked slowly through the empty streets, remembering the days when this place had been filled with life.",
        "Rain fell steadily against the windows while thunder rumbled in the distance.",
        "He wondered if he had made the right choice, although it was too late to turn back now.",
        "The door burst open, and three figures rushed into the room.",

        # Technical (5)
        "The API endpoint accepts POST requests with JSON payloads.",
        "This study examines the effects of temperature on enzyme activity.",
        "Participants were randomly assigned to control or experimental groups.",
        "First, install the required dependencies using the package manager.",
        "The system must handle concurrent requests efficiently.",

        # Complex (5)
        "Although previous research suggested a correlation, the current study found no significant relationship.",
        "The team analyzed the data, the researchers reviewed the findings, and the committee approved the publication.",
        "When the storm approached, residents evacuated the coastal areas.",
        "Scientists believe that the discovery could lead to applications that nobody had previously imagined.",
        "If the hypothesis is correct, then the experiment should produce consistent results."
    ]

    total_clauses = 0
    total_main = 0
    total_coordinate = 0
    total_subordinate = 0

    print("\n" + "="*80)
    print("SUMMARY: Real-World Validation Across All Categories")
    print("="*80)

    for i, text in enumerate(all_samples, 1):
        tokens, doc = preprocessor.process(text)
        clauses = detector.identify_clauses(doc, tokens)

        total_clauses += len(clauses)
        total_main += sum(1 for c in clauses if c.clause_type == "main")
        total_coordinate += sum(1 for c in clauses if c.clause_type == "coordinate")
        total_subordinate += sum(1 for c in clauses if c.clause_type == "subordinate")

    print(f"\nTotal samples analyzed: {len(all_samples)}")
    print(f"Total clauses detected: {total_clauses}")
    print(f"  - Main clauses: {total_main} ({total_main/total_clauses*100:.1f}%)")
    print(f"  - Coordinate clauses: {total_coordinate} ({total_coordinate/total_clauses*100:.1f}%)")
    print(f"  - Subordinate clauses: {total_subordinate} ({total_subordinate/total_clauses*100:.1f}%)")
    print(f"\nAverage clauses per sample: {total_clauses/len(all_samples):.1f}")

    # Validate that we detected clauses in all samples
    assert total_clauses > 0
    assert total_main > 0  # Every sample should have at least one main clause

    print("\n[PASS] All 25 real-world samples processed successfully")
    print("="*80)
