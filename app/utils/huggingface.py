"""
Downloads models from Hugging Face to models/username_modelname.
"""

import argparse
import base64
import datetime
import hashlib
import json
from os import environ
from pathlib import Path
from re import Pattern, compile
from typing import Optional
from logging import INFO, basicConfig, getLogger
from requests import Response, Session

try:
    from tqdm import tqdm
    from tqdm.contrib.concurrent import thread_map

except ImportError:
    tqdm = thread_map = None

basicConfig(level=INFO, format="[%(asctime)s] - %(levelname)s - %(message)s")
logger = getLogger(__name__)
logger.setLevel(INFO)


class HuggingfaceDownloader:
    _base_url: str = "https://huggingface.co"
    _branch_pattern: Pattern = compile(r"^[a-zA-Z0-9._-]+$")
    _pytorch_pattern: Pattern = compile(r"(pytorch|adapter)_model.*\.bin")
    _safetensors_pattern: Pattern = compile(r".*\.safetensors")
    _pt_pattern: Pattern = compile(r".*\.pt")
    _ggml_pattern: Pattern = compile(r".*ggml.*\.bin")
    _tokenizer_pattern: Pattern = compile(r"(tokenizer|ice).*\.model")
    _text_pattern: Pattern = compile(r".*\.(txt|json|py|md)")
    output_folder: Optional[Path] = None

    def __init__(self):
        self.session = Session()
        user: Optional[str] = environ.get("HF_USER")
        password: Optional[str] = environ.get("HF_PASS")
        if user and password:
            self.session.auth = (user, password)

    @classmethod
    def from_repository(
        cls,
        model: str,  # e.g. "facebook/opt-1.3b"
        branch: str = "main",
        threads: int = 1,
        base_folder: Optional[str] = None,
        clean: bool = False,
        check: bool = False,
        text_only: bool = False,
    ) -> "HuggingfaceDownloader":
        model, branch = model.lower(), branch.lower()

        self = cls()
        # Cleaning up the model/branch names
        try:
            model, branch = self.sanitize_model_and_branch_names(model, branch)
        except ValueError as err_branch:
            logger.error(err_branch)
            raise err_branch

        # Getting the download links from Hugging Face
        links, sha256, is_lora = self.get_download_links_from_huggingface(
            model, branch, text_only=text_only
        )
        logger.info("Links:" + "".join([f"\n- {link}" for link in links]))
        logger.info(
            "SHA256:"
            + "".join([f"\n- {fname}: {fhash}" for fname, fhash in sha256]),
        )
        logger.info(f"Is LoRA: {is_lora}")

        # Getting the output folder
        self.output_folder = self.get_output_folder(
            model, branch, is_lora, base_folder=base_folder
        )
        logger.info(f"Output folder: {self.output_folder}")

        if check:
            # Check previously downloaded files
            self.check_model_files_by_sha256(
                sha256=sha256, output_folder=self.output_folder
            )
        else:
            # Download files
            self.download_model_files(
                model,
                branch,
                links,
                sha256,
                self.output_folder,
                threads=threads,
            )
        return self

    def sanitize_model_and_branch_names(
        self, model: str, branch: Optional[str] = None
    ) -> tuple[str, str]:
        if model.endswith("/"):
            model = model[:-1]

        if branch is None:
            branch = "main"
        else:
            if not self._branch_pattern.match(branch):
                raise ValueError(
                    "Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed."
                )

        return model, branch

    def get_download_links_from_huggingface(
        self, model: str, branch: str, text_only: bool = False
    ) -> tuple[list[str], list[list[str]], bool]:
        page: str = f"/api/models/{model}/tree/{branch}"
        cursor: bytes = b""

        links: list[str] = []
        sha256: list[list[str]] = []
        classifications: list[str] = []
        has_pytorch: bool = False
        has_pt: bool = False
        # has_ggml = False
        has_safetensors: bool = False
        is_lora: bool = False
        while True:
            url: str = f"{self._base_url}{page}" + (
                f"?cursor={cursor.decode()}" if cursor else ""
            )
            response: Response = self.session.get(url, timeout=20)
            response.raise_for_status()
            content: bytes = response.content

            json_decoded: dict = json.loads(content)
            if not json_decoded:
                break

            for json_idx in range(len(json_decoded)):
                fname: str = json_decoded[json_idx]["path"]
                if fname.endswith(
                    ("adapter_config.json", "adapter_model.bin")
                ):
                    is_lora = True

                (
                    is_pytorch,
                    is_safetensors,
                    is_pt,
                    is_ggml,
                    is_tokenizer,
                    is_text,
                ) = (
                    self._pytorch_pattern.match(fname),
                    self._safetensors_pattern.match(fname),
                    self._pt_pattern.match(fname),
                    self._ggml_pattern.match(fname),
                    self._tokenizer_pattern.match(fname),
                    self._text_pattern.match(fname),
                )

                if is_text is None:
                    is_text = is_tokenizer
                if any(
                    (
                        is_pytorch,
                        is_safetensors,
                        is_pt,
                        is_ggml,
                        is_tokenizer,
                        is_text,
                    )
                ):
                    if "lfs" in json_decoded[json_idx]:
                        sha256.append(
                            [fname, json_decoded[json_idx]["lfs"]["oid"]]
                        )

                    if is_text:
                        links.append(
                            f"https://huggingface.co/{model}/resolve/{branch}/{fname}"
                        )
                        classifications.append("text")
                        continue

                    if not text_only:
                        links.append(
                            f"https://huggingface.co/{model}/resolve/{branch}/{fname}"
                        )
                        if is_safetensors:
                            has_safetensors = True
                            classifications.append("safetensors")
                        elif is_pytorch:
                            has_pytorch = True
                            classifications.append("pytorch")
                        elif is_pt:
                            has_pt = True
                            classifications.append("pt")
                        elif is_ggml:
                            # has_ggml = True
                            classifications.append("ggml")

            cursor = base64.b64encode(
                (
                    base64.b64encode(
                        f'{{"file_name":"{json_decoded[-1]["path"]}"}}'.encode()
                    )
                    + b":50"
                )
            ).replace(b"=", b"%3D")

        # If both pytorch and safetensors are available, download safetensors only
        if (has_pytorch or has_pt) and has_safetensors:
            for json_idx in range(len(classifications) - 1, -1, -1):
                if classifications[json_idx] in ("pytorch", "pt"):
                    links.pop(json_idx)

        return links, sha256, is_lora

    def get_output_folder(
        self,
        model: str,
        branch: str,
        is_lora: bool,
        base_folder: Optional[str] = None,
    ) -> Path:
        if base_folder is None:
            base_folder = "models" if not is_lora else "loras"

        output_folder: str = f"{'_'.join(model.split('/')[-2:])}"
        if branch != "main":
            output_folder += f"_{branch}"

        return Path(base_folder.lower()) / Path(output_folder.lower())

    def get_single_file(
        self, url: str, output_folder: Path, start_from_scratch: bool = False
    ) -> None:
        filename = Path(url.rsplit("/", 1)[1])
        output_path = output_folder / filename
        headers: dict[str, str] = {}
        mode: str = "wb"
        if output_path.exists() and not start_from_scratch:
            # Check if the file has already been downloaded completely
            response: Response = self.session.get(url, stream=True, timeout=20)
            total_size: int = int(response.headers.get("content-length", 0))
            if output_path.stat().st_size >= total_size:
                logger.info(f"{filename} already exists. Skipping...")
                return

            # Otherwise, resume the download from where it left off
            logger.info(
                f"Resuming download of {filename} "
                f"from {output_path.stat().st_size / 1024**2} MB to {total_size / 1024**2} MB"
            )
            headers = {"Range": f"bytes={output_path.stat().st_size}-"}
            mode = "ab"

        with self.session.get(
            url, stream=True, headers=headers, timeout=20
        ) as response:
            response.raise_for_status()  # Do not continue the download if the request was unsuccessful
            total_size = int(response.headers.get("content-length", 0))
            block_size: int = 1024 * 1024  # 1MB
            with open(output_path, mode) as f:
                t = (
                    tqdm(
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        bar_format="{l_bar}{bar}| {n_fmt:6}/{total_fmt:6} {rate_fmt:6}",
                    )
                    if tqdm is not None
                    else None
                )
                count: int = 0
                for data in response.iter_content(block_size):
                    if t is not None:
                        t.update(len(data))
                    f.write(data)
                    if total_size != 0 and self.progress_bar is not None:
                        count += len(data)
                        self.progress_bar(
                            float(count) / float(total_size),
                            f"Downloading {filename}",
                        )
                # tqdm 객체가 있으면 close 메서드를 호출합니다.
                if t is not None:
                    t.close()

    def start_download_threads(
        self,
        file_list: list[str],
        output_folder: Path,
        start_from_scratch: bool = False,
        threads: int = 1,
    ) -> None:
        # use_tqdm 변수에 따라 thread_map 함수를 사용하거나 사용하지 않습니다.
        if thread_map is not None:
            thread_map(
                lambda url: self.get_single_file(
                    url,
                    output_folder=output_folder,
                    start_from_scratch=start_from_scratch,
                ),
                file_list,
                max_workers=min(threads, len(file_list)),
                disable=True,
            )
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(
                max_workers=min(threads, len(file_list))
            ) as executor:
                [
                    future.result()
                    for future in as_completed(
                        [
                            executor.submit(
                                self.get_single_file,
                                url,
                                output_folder=output_folder,
                                start_from_scratch=start_from_scratch,
                            )
                            for url in file_list
                        ]
                    )
                ]

    def download_model_files(
        self,
        model: str,
        branch: str,
        links: list[str],
        sha256: list[list[str]],
        output_folder: Path,
        progress_bar=None,
        start_from_scratch: bool = False,
        threads: int = 1,
    ) -> None:
        self.progress_bar = progress_bar

        # Creating the folder and writing the metadata
        output_folder.mkdir(parents=True, exist_ok=True)
        metadata: str = (
            f"url: {self._base_url}/{model}\n"
            f"branch: {branch}\n"
            f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        )
        logger.info(f"Downloading with metadata:\n{metadata}")

        sha256_str: str = "\n".join(
            [f"    {item[1]} {item[0]}" for item in sha256]
        )
        if sha256_str:
            metadata += f"sha256sum:\n{sha256_str}"

        metadata += "\n"
        (output_folder / "huggingface-metadata.txt").write_text(metadata)

        # Downloading the files
        logger.info(f"Downloading the model to {output_folder}...")
        self.start_download_threads(
            links,
            output_folder,
            start_from_scratch=start_from_scratch,
            threads=threads,
        )

    def check_model_files_by_sha256(
        self,
        sha256: list[list[str]],
        output_folder: Path,
    ) -> bool:
        # Validate the checksums
        is_validated: bool = True
        for single_sha256 in sha256:
            fname, fhash = single_sha256
            fpath = output_folder / Path(fname)

            if not fpath.exists():
                logger.info(f"The following file is missing: {fpath}")
                is_validated = False
                continue

            with open(output_folder / Path(fname), "rb") as f:
                real_hash = hashlib.sha256(f.read()).hexdigest()
                if real_hash != fhash:
                    logger.info(f"Checksum failed: {fname}  {fhash}")
                    is_validated = False
                else:
                    logger.info(f"Checksum validated: {fname}  {fhash}")

        if is_validated:
            logger.info("[+] Validated checksums of all model files!")
        else:
            logger.error(
                "[-] Invalid checksums. Rerun downloader with the --clean flag."
            )
        return is_validated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        default=None,
        nargs="?",
        help="The model you'd like to download. e.g. facebook/opt-1.3b",
    )

    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Name of the Git branch to download from.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of files to download simultaneously.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only download text files (txt/json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The folder where the model should be saved.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Does not resume the previous download.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validates the checksums of model files.",
    )
    args = parser.parse_args()

    if args.model is None:
        parser.error(
            "Error: Please specify the model you'd like to download "
            "(e.g. 'python download-model.py facebook/opt-1.3b')."
        )

    HuggingfaceDownloader.from_repository(
        model=args.model,
        branch=args.branch,
        threads=args.threads,
        base_folder=args.output,
        clean=args.clean,
        check=args.check,
        text_only=args.text_only,
    )
