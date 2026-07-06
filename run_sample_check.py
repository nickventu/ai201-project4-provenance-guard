"""
Run the full detection pipeline (both signals + aggregation) against six
test samples spanning the confidence range: two extreme (unambiguous)
cases plus the original four clearly-AI/clearly-human/borderline cases.
Each result is written to the audit log alongside its individual signal
scores, per the checkpoint requirement.

Usage:
    set GROQ_API_KEY=your_key_here      (Windows cmd)
    python run_sample_check.py
"""

from pipeline.signals import stylometric, llm_classifier
from pipeline.aggregate import aggregate
from pipeline.label import label_for
from storage import audit_log

SAMPLES = {
    "EXTREME: unambiguous AI corporate-speak (expect: high confidence AI)": (
        "It is important to note that leveraging synergies across cross-functional "
        "teams can significantly enhance overall organizational efficiency. "
        "Furthermore, it is essential to consider that stakeholders must remain "
        "aligned with strategic objectives in order to drive sustainable growth. "
        "Ultimately, by fostering a culture of continuous improvement, organizations "
        "can better position themselves to navigate an increasingly complex and "
        "dynamic business landscape while simultaneously maximizing value for all "
        "relevant stakeholders."
    ),
    "EXTREME: unambiguous idiosyncratic human story (expect: high confidence human)": (
        "My uncle Ray used to keep a shoebox of rubber bands under the sink for "
        "absolutely no reason anyone could ever explain, and when he died last "
        "spring we found seventeen more boxes in the garage, all labeled by decade "
        "like some kind of deranged elastic archaeology. I laughed so hard at the "
        "funeral reception that Aunt Pat swatted my arm. Grief's weird like that. "
        "You don't get to pick what breaks you open."
    ),
    "ai_generated (expect: high)": (
        "Artificial intelligence represents a transformative paradigm shift in modern society. "
        "It is important to note that while the benefits of AI are numerous, it is equally "
        "essential to consider the ethical implications. Furthermore, stakeholders across "
        "various sectors must collaborate to ensure responsible deployment."
    ),
    "human_casual (expect: low)": (
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in it and "
        "i was thirsty for like three hours after. my friend got the spicy version and "
        "said it was better. probably won't go back unless someone drags me there"
    ),
    "human_formal (expect: mid-high, borderline)": (
        "The relationship between monetary policy and asset price inflation has been "
        "extensively studied in the literature. Central banks face a fundamental tension "
        "between their mandate for price stability and the unintended consequences of "
        "prolonged low interest rates on equity and real estate valuations."
    ),
    "ai_lightly_edited (expect: mid-range, borderline)": (
        "I've been thinking a lot about remote work lately. There are genuine tradeoffs — "
        "flexibility and no commute on one side, isolation and blurred work-life boundaries "
        "on the other. Studies show productivity varies widely by individual and role type."
    ),
}


def main():
    for name, text in SAMPLES.items():
        print(f"--- {name} ---")
        try:
            result = aggregate(text)
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
            continue

        content_id = audit_log.compute_content_id(text)
        labeled = label_for(result["raw_score"], result["confidence"])
        print(f"  content_id:    {content_id}")
        print(f"  stylometric:   {result['signals']['stylometric']:.3f}")
        print(f"  llm_classifier:{result['signals']['llm_classifier']:.3f}")
        print(f"  raw_score:     {result['raw_score']:.3f}")
        print(f"  confidence:    {result['confidence']:.3f}")
        print(f"  label:         {labeled.label}")

        entry = audit_log.log_result(
            content_id=content_id,
            signals=result["signals"],
            raw_score=result["raw_score"],
            confidence=result["confidence"],
            label=labeled.label,
        )
        print(f"  logged at:     {entry['created_at']}")
        print()

    print("=== full audit log (most recent first) ===")
    for entry in audit_log.get_all():
        print(
            f"  {entry['content_id']}  raw_score={entry['raw_score']:.3f}  "
            f"confidence={entry['confidence']:.3f}  signals={entry['signals']}"
        )


if __name__ == "__main__":
    main()

