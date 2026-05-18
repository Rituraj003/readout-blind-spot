from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable

from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _auto_load_dotenv() -> None:
    project_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / ".env", project_root / ".env"]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_env_file(resolved)


def _resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            if key != "HF_TOKEN":
                os.environ.setdefault("HF_TOKEN", value)
            return value
    return None


class BaseTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self.itos = vocab
        self.stoi = {token: index for index, token in enumerate(vocab)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(self.stoi.get(token, self.unk_id) for token in self.tokenize(text))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int], stop_at_eos: bool = True, skip_special: bool = True) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if stop_at_eos and token_id == self.eos_id:
                break
            token = self.itos[token_id]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return self.detokenize(pieces)

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def detokenize(self, pieces: list[str]) -> str:
        return "".join(pieces)

    def fit_length(
        self,
        token_ids: list[int],
        length: int,
        *,
        pad_side: str,
        truncate_side: str,
        add_eos: bool = False,
    ) -> list[int]:
        working = list(token_ids)
        if add_eos and (not working or working[-1] != self.eos_id):
            working.append(self.eos_id)
        if len(working) > length:
            if add_eos and truncate_side == "right" and length > 0:
                working = working[: length - 1] + [self.eos_id]
            elif add_eos and truncate_side == "left" and length > 0:
                if length == 1:
                    working = [self.eos_id]
                else:
                    working = [self.eos_id] + working[-(length - 1) :]
            elif truncate_side == "left":
                working = working[-length:]
            elif not add_eos:
                working = working[:length]
        padding = [self.pad_id] * max(length - len(working), 0)
        if pad_side == "left":
            return padding + working
        return working + padding

    def __len__(self) -> int:
        return len(self.itos)


class CharacterTokenizer(BaseTokenizer):
    kind = "char"

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharacterTokenizer":
        symbols = sorted({char for text in texts for char in text})
        vocab = SPECIAL_TOKENS + [symbol for symbol in symbols if symbol not in SPECIAL_TOKENS]
        return cls(vocab)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "CharacterTokenizer":
        return cls(list(state["itos"]))

    def to_state(self) -> dict[str, Any]:
        return {"kind": self.kind, "itos": self.itos}

    def tokenize(self, text: str) -> list[str]:
        return list(text)


class RegexTokenizer(BaseTokenizer):
    kind = "regex"

    def __init__(self, vocab: list[str], pattern: str) -> None:
        super().__init__(vocab)
        self.pattern = pattern
        self._regex = re.compile(pattern)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        pattern: str,
        *,
        max_vocab_size: int | None = None,
        min_token_freq: int = 1,
    ) -> "RegexTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(cls.tokenize_static(text, pattern))
        items = [
            (token, count)
            for token, count in counts.items()
            if count >= min_token_freq and token not in SPECIAL_TOKENS
        ]
        items.sort(key=lambda item: (-item[1], item[0]))
        if max_vocab_size is not None:
            limit = max(0, max_vocab_size - len(SPECIAL_TOKENS))
            items = items[:limit]
        symbols = [token for token, _ in items]
        vocab = SPECIAL_TOKENS + [symbol for symbol in symbols if symbol not in SPECIAL_TOKENS]
        return cls(vocab, pattern)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RegexTokenizer":
        return cls(list(state["itos"]), str(state["pattern"]))

    @staticmethod
    def tokenize_static(text: str, pattern: str) -> list[str]:
        return re.findall(pattern, text)

    def to_state(self) -> dict[str, Any]:
        return {"kind": self.kind, "itos": self.itos, "pattern": self.pattern}

    def tokenize(self, text: str) -> list[str]:
        return self._regex.findall(text)


def tokenizer_from_state(state: dict[str, Any]) -> BaseTokenizer:
    kind = state.get("kind", "char")
    if kind == "char":
        return CharacterTokenizer.from_state(state)
    if kind == "regex":
        return RegexTokenizer.from_state(state)
    if kind == "sat":
        from data.sat import SatTokenizer
        return SatTokenizer.from_state(state)
    if kind == "addition":
        from data.addition import AdditionTokenizer
        return AdditionTokenizer.from_state(state)
    raise ValueError(f"unknown tokenizer kind in checkpoint: {kind}")


def build_tokenizer(texts: list[str], config: dict[str, Any]) -> BaseTokenizer:
    tokenizer_config = config.get("tokenizer", {})
    kind = tokenizer_config.get("kind", "char")
    if kind == "char":
        return CharacterTokenizer.from_texts(texts)
    if kind == "regex":
        pattern = tokenizer_config.get("pattern", r"\s+|[A-Za-z0-9_]+|[^\w\s]")
        return RegexTokenizer.from_texts(
            texts,
            pattern,
            max_vocab_size=tokenizer_config.get("max_vocab_size"),
            min_token_freq=tokenizer_config.get("min_token_freq", 1),
        )
    raise ValueError(f"unknown tokenizer kind: {kind}")


@dataclass(frozen=True)
class SyntheticEntityRecord:
    prefix_text: str
    suffix_text: str
    metadata: dict[str, str]


def _indefinite_article(noun: str) -> str:
    return "an" if noun[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def build_synthetic_entity_records(num_samples: int, seed: int) -> list[SyntheticEntityRecord]:
    names = [
        "Aria",
        "Milo",
        "Nina",
        "Theo",
        "Iris",
        "Jules",
        "Omar",
        "Lena",
        "Kira",
        "Zane",
    ]
    cities = [
        "Kyoto",
        "Lima",
        "Oslo",
        "Accra",
        "Seoul",
        "Rome",
        "Doha",
        "Austin",
        "Hanoi",
        "Jaipur",
    ]
    colors = [
        "amber",
        "teal",
        "crimson",
        "ivory",
        "violet",
        "silver",
        "gold",
        "navy",
        "coral",
        "olive",
    ]
    pets = [
        "otter",
        "fox",
        "owl",
        "cat",
        "dog",
        "rabbit",
        "iguana",
        "alpaca",
        "emu",
        "yak",
    ]

    rng = random.Random(seed)
    records: list[SyntheticEntityRecord] = []
    for _ in range(num_samples):
        name = rng.choice(names)
        city = rng.choice(cities)
        color = rng.choice(colors)
        pet = rng.choice(pets)
        article = _indefinite_article(pet)
        prefix_text = (
            f"Facts: name={name}; city={city}; color={color}; pet={pet}.\n"
            "Task: write one recap sentence using the same facts.\n"
            "Recap:"
        )
        suffix_text = f" {name} lives in {city}, likes {color}, and keeps {article} {pet}."
        records.append(
            SyntheticEntityRecord(
                prefix_text=prefix_text,
                suffix_text=suffix_text,
                metadata={"name": name, "city": city, "color": color, "pet": pet},
            )
        )
    return records


class SyntheticEntityDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[SyntheticEntityRecord],
        tokenizer: BaseTokenizer,
        prefix_length: int,
        suffix_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length

        prefix_lengths = [len(tokenizer.encode(record.prefix_text)) for record in records]
        suffix_lengths = [len(tokenizer.encode(record.suffix_text, add_eos=True)) for record in records]
        max_prefix_length = max(prefix_lengths, default=0)
        max_suffix_length = max(suffix_lengths, default=0)
        if max_prefix_length > prefix_length:
            raise ValueError(
                "prefix_length is too small for synthetic_entities: "
                f"configured={prefix_length}, required>={max_prefix_length}"
            )
        if max_suffix_length > suffix_length:
            raise ValueError(
                "suffix_length is too small for synthetic_entities: "
                f"configured={suffix_length}, required>={max_suffix_length}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        prefix_tokens = self.tokenizer.fit_length(
            self.tokenizer.encode(record.prefix_text),
            self.prefix_length,
            pad_side="left",
            truncate_side="left",
        )
        suffix_tokens = self.tokenizer.fit_length(
            self.tokenizer.encode(record.suffix_text),
            self.suffix_length,
            pad_side="right",
            truncate_side="right",
            add_eos=True,
        )
        return {
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "metadata": record.metadata,
            "prefix_text": record.prefix_text,
            "suffix_text": record.suffix_text,
        }


class TextWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        text: str | None,
        tokenizer: BaseTokenizer | None,
        prefix_length: int,
        suffix_length: int,
        stride: int,
        *,
        token_ids: list[int] | None = None,
        start_positions: range | list[int] | None = None,
    ) -> None:
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.total_length = prefix_length + suffix_length

        if token_ids is None:
            if text is None:
                raise ValueError("text must be provided when token_ids are not supplied")
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when token_ids are not supplied")
            token_ids = tokenizer.encode(text, add_eos=True)
        self.token_ids = token_ids

        if len(self.token_ids) < self.total_length:
            raise ValueError("text corpus is shorter than prefix_length + suffix_length")

        if start_positions is None:
            max_start = len(self.token_ids) - self.total_length
            start_positions = range(0, max_start + 1, stride)
        self.start_positions = start_positions

    def __len__(self) -> int:
        return len(self.start_positions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = self.start_positions[index]
        window = self.token_ids[start : start + self.total_length]
        return {
            "prefix_tokens": window[: self.prefix_length],
            "suffix_tokens": window[self.prefix_length :],
        }

    def subset(self, start_index: int, end_index: int) -> "TextWindowDataset":
        return TextWindowDataset(
            text=None,
            tokenizer=None,
            prefix_length=self.prefix_length,
            suffix_length=self.suffix_length,
            stride=1,
            token_ids=self.token_ids,
            start_positions=self.start_positions[start_index:end_index],
        )


def load_hf_text_split(config: dict[str, Any], split: str) -> list[str]:
    _auto_load_dotenv()
    token = _resolve_hf_token()
    cache_root = config.get("hf_cache_dir")
    if cache_root is not None:
        resolved = str(Path(cache_root).resolve())
        os.environ["HF_HOME"] = resolved
        os.environ["HF_HUB_CACHE"] = str(Path(resolved) / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(resolved) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(resolved) / "datasets")
        os.environ["HF_XET_CACHE"] = str(Path(resolved) / "xet")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for hf_text. Run `uv sync` to install project dependencies."
        ) from exc

    load_kwargs: dict[str, Any] = {
        "path": config["hf_dataset"],
        "name": config.get("hf_name"),
        "split": split,
        "cache_dir": config.get("hf_cache_dir"),
    }
    if token is not None:
        load_kwargs["token"] = token

    try:
        dataset = load_dataset(**load_kwargs)
    except TypeError:
        if token is None:
            raise
        load_kwargs.pop("token", None)
        load_kwargs["use_auth_token"] = token
        dataset = load_dataset(**load_kwargs)
    text_field = config.get("hf_text_field", "text")
    drop_empty = config.get("drop_empty", True)
    strip = config.get("strip_text", False)

    texts: list[str] = []
    for value in dataset[text_field]:
        text = str(value)
        if strip:
            text = text.strip()
        if drop_empty and not text:
            continue
        texts.append(text)
    return texts


def join_texts(texts: list[str], join_with: str) -> str:
    return join_with.join(texts)


def build_datasets(config: dict[str, Any]) -> tuple[BaseTokenizer, Dataset[Any], Dataset[Any]]:
    kind = config["kind"]
    prefix_length = config["prefix_length"]
    suffix_length = config["suffix_length"]

    if kind == "synthetic_entities":
        train_records = build_synthetic_entity_records(config["train_samples"], config["train_seed"])
        val_records = build_synthetic_entity_records(config["val_samples"], config["val_seed"])
        texts = [record.prefix_text + record.suffix_text for record in train_records + val_records]
        tokenizer = build_tokenizer(texts, config)
        train_dataset = SyntheticEntityDataset(train_records, tokenizer, prefix_length, suffix_length)
        val_dataset = SyntheticEntityDataset(val_records, tokenizer, prefix_length, suffix_length)
        return tokenizer, train_dataset, val_dataset

    if kind == "text_file":
        text_path = Path(config["text_file"])
        text = text_path.read_text(encoding="utf-8")
        tokenizer = build_tokenizer([text], config)
        dataset = TextWindowDataset(
            text=text,
            tokenizer=tokenizer,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            stride=config["stride"],
        )
        split_index = max(1, int(len(dataset) * (1.0 - config["val_fraction"])))
        if split_index >= len(dataset):
            split_index = max(1, len(dataset) - 1)
        train_dataset = dataset.subset(0, split_index)
        val_dataset = dataset.subset(split_index, len(dataset))
        return tokenizer, train_dataset, val_dataset

    if kind == "addition":
        from data.addition import AdditionDataset, AdditionTokenizer

        n_digits = config["n_digits"]
        tokenizer = AdditionTokenizer.build()
        train_dataset = AdditionDataset(
            n_digits=n_digits,
            n_samples=config["train_samples"],
            base_seed=config["train_seed"],
            tokenizer=tokenizer,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
        )
        val_dataset = AdditionDataset(
            n_digits=n_digits,
            n_samples=config["val_samples"],
            base_seed=config["val_seed"],
            tokenizer=tokenizer,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
        )
        return tokenizer, train_dataset, val_dataset

    if kind == "sat":
        from data.sat import SatDataset, SatTokenizer

        n_vars = config["n_vars"]
        n_clauses = config["n_clauses"]
        k = config.get("k", 3)
        tokenizer = SatTokenizer.from_n_vars(n_vars)
        train_dataset = SatDataset(
            n_vars=n_vars,
            n_clauses=n_clauses,
            k=k,
            n_samples=config["train_samples"],
            base_seed=config["train_seed"],
            tokenizer=tokenizer,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
        )
        val_dataset = SatDataset(
            n_vars=n_vars,
            n_clauses=n_clauses,
            k=k,
            n_samples=config["val_samples"],
            base_seed=config["val_seed"],
            tokenizer=tokenizer,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
        )
        return tokenizer, train_dataset, val_dataset

    if kind == "hf_text":
        train_texts = load_hf_text_split(config, config["hf_train_split"])
        val_texts = load_hf_text_split(config, config["hf_val_split"])
        tokenizer = build_tokenizer(train_texts + val_texts, config)
        join_with = config.get("join_with", "\n\n")
        train_text = join_texts(train_texts, join_with)
        val_text = join_texts(val_texts, join_with)
        train_dataset = TextWindowDataset(
            text=train_text,
            tokenizer=tokenizer,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            stride=config["stride"],
        )
        val_dataset = TextWindowDataset(
            text=val_text,
            tokenizer=tokenizer,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
            stride=config["stride"],
        )
        return tokenizer, train_dataset, val_dataset

    raise ValueError(f"unknown data kind: {kind}")


class _ListDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
