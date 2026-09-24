import polars as pl

def main():
    print("Hello from llm-faithfulness!")

    df = pl.read_ndjson('hf://datasets/shuyuej/MedQA-USMLE-Benchmark/MedQA_USMLE_test.jsonl')

    print(f"{df.item(0, 'options')}")


if __name__ == "__main__":
    main()
