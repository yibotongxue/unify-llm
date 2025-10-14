from ..utils.config import *
from ..utils.type_utils import InferenceInput
from .factory import InferenceFactory


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file-path",
        type=str,
        required=True,
        help="The path to the config file",
    )
    args, unparsed_args = parser.parse_known_args()

    cfgs = load_config(args.config_file_path)
    update_config_with_unparsed_args(unparsed_args=unparsed_args, cfgs=cfgs)

    inference = InferenceFactory.get_inference_instance(**cfgs)

    inference_input = [
        InferenceInput.from_prompts(
            prompt="中国的首都是哪里？",
            system_prompt="你是一个人工智能助手",
        ),
        InferenceInput.from_prompts(
            prompt="Where is the capital of China?",
            system_prompt="You are an AI assistant",
        ),
        InferenceInput.from_prompts(
            prompt="中国有多少个省份？分别是哪些？",
            system_prompt="你是一个人工智能助手",
        ),
        InferenceInput.from_prompts(
            prompt="How many provinces are there in China? What are they?",
            system_prompt="You are an AI assistant",
        ),
    ]

    outputs = inference._generate(
        inference_input,
        enable_tqdm=True,
        tqdm_args={"desc": "Generating responses using api"},
    )

    for inference_output in outputs:
        print(inference_output.response)


main()
