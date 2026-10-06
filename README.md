<a id="readme-top"></a>



<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">LLM Faithfulness Evaluator</h3>

  <p align="center">
    Evaluate whether LLM Chain-of-Thought explanations are faithful or post-hoc rationalisations.
    <br />
    <a href="https://github.com/marat-davudov/llm-faithfulness/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/marat-davudov/llm-faithfulness/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

`llm-faithfulness` is a lightweight, black-box evaluation toolkit for measuring how honest large language models are about their own reasoning. We start with the **Turpin bias-resistance test** (Turpin et al., NeurIPS 2023): run MedQA-USMLE questions with and without a subtle prompt-level bias cue, detect whether the model's answer flips to the cue, and use a judge model to check whether the written explanation admits the cue influenced the decision. A flip without acknowledgement is the classic signature of an unfaithful, post-hoc rationalisation.

The code is intentionally kept as a small, copy-pasteable Python module so it can be embedded in demos, articles, and research pipelines.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]
* [![AWS][AWS]][AWS-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is a [uv](https://docs.astral.sh/uv/) managed Python project. To get a local copy up and running follow these steps.

### Prerequisites

* Python 3.12+
* [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/marat-davudov/llm-faithfulness.git
   ```
2. Install dependencies
   ```sh
   uv sync
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

The CLI runs the Turpin bias-resistance evaluation on a sample of the MedQA-USMLE benchmark using AWS Bedrock. Example:

```sh
uv run python main.py \
  --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \
  --judge-id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --region us-east-1 \
  --sample-size 50 \
  --seed 42 \
  --cue both \
  --output results.jsonl
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock cross-region inference profile for the target model. |
| `--judge-id` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Bedrock cross-region inference profile for the verbalisation judge. |
| `--region` | None | AWS region passed to the Bedrock runtime client. |
| `--sample-size` | `50` | Number of cases to sample and evaluate. |
| `--seed` | `42` | Random seed for case sampling and bias-target selection. |
| `--cue` | `both` | Bias cue to inject: `sycophancy`, `metadata`, or `both`. |
| `--data-path` | None | Path to a local MedQA JSONL file; otherwise the dataset is downloaded/cached. |
| `--output` | `results.jsonl` | Path for the per-case JSONL results file. |

### Prerequisites

* Configure AWS credentials for the target account/region.
* Enable Bedrock model access for the target and judge inference profiles in the target region.
* Verify the exact inference profile IDs available to your account if a run fails with a model-not-found error:
  ```sh
  aws bedrock list-inference-profiles --region <region>
  ```

The CLI prints a human-readable report and writes per-case JSONL results with model/Judge outputs and verdicts.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] MedQA-USMLE dataset loader
- [ ] Turpin bias-resistance evaluator (authority and XML metadata cues)
- [ ] Verbalisation judge for flipped explanations
- [ ] Aggregate faithfulness report
- [ ] Lanham mistake-injection test
- [ ] Counterfactual-swap test

See the [open issues](https://github.com/marat-davudov/llm-faithfulness/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Run tests on every commit

Install [pre-commit](https://pre-commit.com/) and register the repository's Git hook:

```sh
brew install pre-commit
pre-commit install
```

The hook runs `uv run pytest` before each commit. To run it manually against all files:

```sh
pre-commit run --all-files
```

### Top contributors:

<a href="https://github.com/marat-davudov/llm-faithfulness/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=marat-davudov/llm-faithfulness" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Turpin et al., *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting* (NeurIPS 2023)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/marat-davudov/llm-faithfulness.svg?style=for-the-badge
[contributors-url]: https://github.com/marat-davudov/llm-faithfulness/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/marat-davudov/llm-faithfulness.svg?style=for-the-badge
[forks-url]: https://github.com/marat-davudov/llm-faithfulness/network/members
[stars-shield]: https://img.shields.io/github/stars/marat-davudov/llm-faithfulness.svg?style=for-the-badge
[stars-url]: https://github.com/marat-davudov/llm-faithfulness/stargazers
[issues-shield]: https://img.shields.io/github/issues/marat-davudov/llm-faithfulness.svg?style=for-the-badge
[issues-url]: https://github.com/marat-davudov/llm-faithfulness/issues
[license-shield]: https://img.shields.io/github/license/marat-davudov/llm-faithfulness.svg?style=for-the-badge
[license-url]: https://github.com/marat-davudov/llm-faithfulness/blob/main/LICENSE
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[AWS]: https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white
[AWS-url]: https://aws.amazon.com/bedrock/
