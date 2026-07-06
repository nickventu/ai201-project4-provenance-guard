"""
Run the full detection pipeline (both signals + aggregation) against the
four hand-picked test samples: clearly-AI, clearly-human, borderline-formal-
human, and borderline-lightly-edited-AI.

Usage:
    set GROQ_API_KEY=your_key_here      (Windows cmd)
    python run_sample_check.py
"""

from pipeline.signals import stylometric, llm_classifier
from pipeline.aggregate import aggregate, CalibrationModel

SAMPLES = {
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
    calibration = CalibrationModel.load()  # will be unfitted unless you've run .fit()
    print(f"calibration fitted: {calibration.fitted}\n")

    for name, text in SAMPLES.items():
        print(f"--- {name} ---")
        try:
            result = aggregate(text, calibration=calibration)
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
            continue

        print(f"  stylometric:   {result['signals']['stylometric']:.3f}")
        print(f"  llm_classifier:{result['signals']['llm_classifier']:.3f}")
        print(f"  raw_score:     {result['raw_score']:.3f}")
        print(f"  confidence:    {result['confidence']:.3f}")
        print(f"  calibrated:    {result['calibrated']}")
        print()


if __name__ == "__main__":
    main()
