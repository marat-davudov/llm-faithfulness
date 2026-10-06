import argparse
import logging

from llm_faithfulness.utils.logging_utils import get_logger

logger = get_logger("medqa_dataset_generator")

BEDROCK_MODEL_IDS = "amazon.nova-pro-v1:0"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", help="Logging level", action="store_true", default=False)
    parser.add_argument(
        "--force",
        help="Force overwrite of the output file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output",
        help="Path to the output file",
        default="out/dataset.json",
    )
    parser.add_argument(
        "--samples",
        help="Number of samples to generate",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--model-ids",
        help="Comma-separated list of model IDs to use",
        type=lambda s: s.split(","),
        default=BEDROCK_MODEL_IDS.split(","),
    )

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.v else logging.INFO)

    logger.info("Info")


if __name__ == "__main__":
    main()


# {
#     "prompt": "What is high intensity interval training?",
#     "referenceResponse": "High-Intensity Interval Training (HIIT) is a cardiovascular exercise approach that involves short, intense bursts of exercise followed by brief recovery or rest periods.",
#     "category": "Fitness",
#      "modelResponses": [
#         {
#             "response": "High intensity interval training (HIIT) is a workout strategy that alternates between short bursts of intense, maximum-effort exercise and brief recovery periods, designed to maximize calorie burn and improve cardiovascular fitness.",
#             "modelIdentifier": "my_model"
#         }
#     ]
# }
