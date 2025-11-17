#!/usr/bin/env python3
# ==========================================
# Text Summarizer
# Last experimented: 2025-11-17
# Platform: M1 Mac
# ==========================================
#
# Uses HuggingFace 'transformers' library.
#
# Summarizes multiple text samples.
#   Each gets:
#       - Short summary
#       - Bullet digest
#       - Title proposal
#
# Setup:
#   1. Create and activate a virtual environment
#        python3 -m venv venv
#        source venv/bin/activate
#
#   2. Install dependencies (choose one)
#        a) Installing manually:
#              pip install torch transformers
#
#        b) Using requirements.txt:
#              pip install -r requirements.txt
#
#   3. Run the script:
#        python summarizer_en.py
#
# Notes:
#   • The first run will download the model.
#   • Subsequent runs are fast thanks to local caching.
#
# License:
#   MIT License
#
# Enjoy coding! 🛸
# ==========================================

from transformers import pipeline


def summarize(summarizer, text, max_len, min_len):
    return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0][
        "summary_text"
    ].strip()


def main():

    model_name = "facebook/bart-large-cnn"

    # --- Load summarization model ---
    summarizer = pipeline("summarization", model=model_name)

    # --- Sample text to summarize ---
    text = """
    Artificial intelligence research has accelerated rapidly over the past decade.
    New architectures, massive datasets, and faster hardware have enabled models
    that can perform language translation, generate images, summarize research,
    and even write software. While these tools offer enormous potential, they also
    raise questions about privacy, labor, and long-term societal impact. As AI
    systems become more capable, researchers emphasize the importance of careful
    evaluation, transparency, and responsible deployment. Understanding both the
    benefits and limitations will shape how we integrate AI into daily life.
    """

    # --- Summaries ---
    title = summarize(summarizer, text, 20, 5)
    short = summarize(summarizer, text, 110, 60)

    # --- Print clean output ---
    print("=== TITLE ===")
    print(title.strip(), "\n")

    print("=== SUMMARY ===")
    print(short.strip(), "\n")

    print("\n==========================")


if __name__ == "__main__":
    main()
