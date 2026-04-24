from transformers import pipeline



qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

def summarize_chunks(chunks):
    summaries = []

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=120,
            min_length=40,
            do_sample=False
        )
        summaries.append(result[0]['summary_text'])

    return " ".join(summaries)


def final_summary(text):
    result = summarizer(
        text,
        max_length=200,
        min_length=80,
        do_sample=False
    )
    return result[0]['summary_text']