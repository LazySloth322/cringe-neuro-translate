#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
srt_semantic_compressor.py

Постобработка SRT после speaker_pipeline.py.

Что делает:
Поддерживаемые режимы:
1. compress
   Семантически сокращает длинные реплики.

2. translate
   Переводит КАЖДУЮ реплику на естественный русский язык.

3. translate_compress
   Переводит каждую реплику на русский язык, а длинные реплики
   дополнительно семантически сокращает.

Во всех режимах сохраняются:
- таймкоды;
- порядок SRT;
- [Speaker N].

Исходный SRT никогда не изменяется: создаётся новый файл.

По умолчанию используется:
    Qwen/Qwen2.5-3B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct

Модель поддерживает русский язык и имеет размер репозитория около 3.1 GB
в исходном формате; для слабого железа есть опция 4-bit quantization.



            "Ты начинающий переводчик расшифровок речи. "
            "Твоя задача — перевести реплику на естественный русский язык. "
            "Перевод должен передавать смысл исходника, а не быть дословным. "
            "Сохраняй контекст, отрицания, причинно-следственные связи, "
            "имена собственные и технические термины. "
            "Не добавляй информацию, которой нет в исходнике. "
            "Сохраняй разговорный характер речи, если он есть. "
            "Если исходный текст уже на русском, верни его без изменений, "
            "кроме очевидных ошибок распознавания речи. "
            "Не пиши пояснений."

"""

import argparse
import gc
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList

class NoCJKLogitsProcessor(LogitsProcessor):
    """Физически запрещает модели генерировать токены, содержащие иероглифы CJK."""
    def __init__(self, tokenizer):
        print("Scanning tokenizer for CJK tokens (this may take ~10 sec)...")
        self.bad_ids = []
        # Проверяем все токены в словаре модели
        for token_id in range(tokenizer.vocab_size):
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            # Ищем китайские, японские и корейские иероглифы
            if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', token_str):
                self.bad_ids.append(token_id)
        
        self.bad_ids_tensor = torch.tensor(self.bad_ids, dtype=torch.long)
        print(f"Successfully blocked {len(self.bad_ids)} CJK tokens.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Обнуляем логиты (вероятности) для всех плохих токенов
        scores[:, self.bad_ids_tensor.to(scores.device)] = -float("inf")
        return scores



# ---------------------------------------------------------------------
# DEFAULT CONFIG
# ---------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DEFAULT_MIN_WORDS = 19
DEFAULT_TARGET_RATIO = 0.55

# Ограничиваем максимальный размер входного текста на один запрос.
# Для обычных SRT-реплик этого более чем достаточно и экономит память.
DEFAULT_MAX_INPUT_TOKENS = 900

# Максимально возможная длина ответа.
DEFAULT_MAX_NEW_TOKENS = 180


# ---------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------

TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)

SPEAKER_RE = re.compile(
    r"^\s*\[(?P<speaker>[^\]]+)\]\s*:\s*(?P<text>.*)$",
    re.DOTALL,
)


def parse_srt(path: Path):
    """
    Возвращает список cue:
    {
        "index": int,
        "start": str,
        "end": str,
        "speaker": str | None,
        "text": str,
    }
    """
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())

    cues = []

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue

        # Номер блока
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Таймкод
        time_match = TIME_RE.search(lines[1])
        if not time_match:
            continue

        start = time_match.group("start")
        end = time_match.group("end")

        text = "\n".join(lines[2:]).strip()

        speaker = None
        speaker_match = SPEAKER_RE.match(text)

        if speaker_match:
            speaker = speaker_match.group("speaker").strip()
            text = speaker_match.group("text").strip()

        cues.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
        )

    return cues


def write_srt(path: Path, cues):
    """
    Сохраняет SRT.
    Таймкоды и спикеры берутся напрямую из исходного файла.
    """
    output = []

    for i, cue in enumerate(cues, start=1):
        speaker = cue["speaker"]

        if speaker:
            text = f"[{speaker}]: {cue['text']}"
        else:
            text = cue["text"]

        output.append(
            f"{i}\n"
            f"{cue['start']} --> {cue['end']}\n"
            f"{text}\n"
        )

    path.write_text("\n".join(output), encoding="utf-8")


# ---------------------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------------------

def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def cleanup_model_output(text: str) -> str:
    """
    Удаляет случайные служебные обертки, которые маленькая модель
    иногда добавляет несмотря на инструкцию.
    """
    text = text.strip()

    # Удаляем markdown-кодовые обертки.
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Частые служебные префиксы.
    text = re.sub(
        r"^(?:Ответ|Сокращённый вариант|Сокращенный вариант|"
        r"Перевод|Перевод на русский|Русский перевод|Вот перевод|"
        r"Output|Вывод)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^(?:Русский|Russian)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Если модель зачем-то взяла ответ в кавычки целиком.
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
        
    text = normalize_spaces(text)
    
    # --- ДЕТЕКТОР ЗАЦИКЛИВАНИЯ ---
    # Если какое-то слово повторяется больше 3 раз в коротком субтитре — это бред.
    words = re.findall(r"\b\w+\b", text.lower())
    if words:
        from collections import Counter
        most_common_word, count = Counter(words).most_common(1)[0]
        # Исключаем короткие предлоги/союзы из проверки, чтобы не было ложных срабатываний
        if len(most_common_word) > 3 and count > 3:
            print(f"  [!] DETECTED REPETITION LOOP: '{most_common_word}' repeated {count} times. Discarding.")
            return "" # Возвращаем пустую строку, чтобы сработал fallback в основном коде
          
    return text


# ---------------------------------------------------------------------
# LOCAL MODEL
# ---------------------------------------------------------------------

class LocalCompressor:
    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = False,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        if load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit режим в этом скрипте рассчитан на CUDA. "
                "Запустите без --4bit при работе только на CPU."
            )

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model: {model_name}")

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
        }

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "Для --4bit установите bitsandbytes:\n"
                    "pip install -U bitsandbytes"
                ) from exc

            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            model_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        self.model.eval()
        self.logits_processor = LogitsProcessorList([NoCJKLogitsProcessor(self.tokenizer)])
        print(f"Model device: {self.model.device}")

    def _build_prompt(self, text: str, target_ratio: float) -> list:
        target_percent = int(target_ratio * 100)

        system_prompt = (
            "Ты редактор расшифровок речи. "
            "Твоя задача — сжать длинную реплику, не потеряв её общий смысл. "
            "Сохраняй язык исходного текста. "
            "Не добавляй фактов, которых нет в исходнике. "
            "Не меняй смысл отрицаний и причинно-следственных связей. "
            "Убирай повторы, слова-паразиты, лишние вводные фразы, "
            "длинные перечисления и второстепенные детали. "
            "Разрешено заметно переформулировать исходную фразу. "
            "Результат должен звучать естественно, как нормальная человеческая речь. "
            "Не пиши пояснений о своих действиях."
        )

        user_prompt = (
            f"Сократи следующую реплику примерно до {target_percent}% "
            f"от исходного объёма, сохранив главный смысл.\n\n"
            "ВАЖНО:\n"
            "- Верни ТОЛЬКО готовый сокращённый текст.\n"
            "- Не используй кавычки вокруг всего ответа.\n"
            "- Не добавляй заголовки вроде «Ответ:».\n"
            "- Не придумывай информацию.\n"
            "- Если исходник уже достаточно компактный, сократи совсем немного.\n\n"
            f"ИСХОДНИК:\n{text}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _generate(self, messages: list) -> str:
        """Общий inference-метод для всех задач."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            use_cache=True,
            logits_processor=self.logits_processor,
            stop_strings=["\n", "Input:", "Source:", "Перевод:", "Ответ:"], 
            tokenizer=self.tokenizer # Требуется для работы stop_strings
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

        result = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return cleanup_model_output(result)

    def _build_translation_prompt(self, text: str) -> list:
        system_prompt = (
            "Ты профессиональный переводчик расшифровок речи. "
            "Твоя задача — перевести реплику на естественный русский язык. "
            "Перевод должен передавать смысл исходника, а не быть дословным. "
            "Сохраняй контекст, отрицания, причинно-следственные связи, "
            "имена собственные и технические термины. "
            "Не добавляй информацию, которой нет в исходнике. "
            "Сохраняй разговорный характер речи, если он есть. "
            "Если исходный текст уже на русском, верни его без изменений, "
            "кроме очевидных грамматических ошибок, возникших из-за транскрипции. "
            "Не добавляй пояснений."
        )

        user_prompt = (
            "Переведи следующую реплику на русский язык.\n\n"
            "ВАЖНО:\n"
            "- Верни только перевод.\n"
            "- Не добавляй пояснений или комментариев.\n"
            "- Не используй заголовок «Перевод:».\n"
            "- Не заключай весь ответ в кавычки.\n\n"
            f"ИСХОДНИК:\n{text}"
        )
        

        # Приводим написанные выше \\n к реальным переводам строк.
        user_prompt = user_prompt.replace("\\n", "\n")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_translate_and_compress_prompt(
        self,
        text: str,
        target_ratio: float,
    ) -> list:
        target_percent = int(target_ratio * 100)

        system_prompt = (
            "Ты одновременно переводчик и редактор расшифровок речи. "
            "Сначала мысленно пойми смысл исходной реплики, затем передай его "
            "на естественном русском языке и убери второстепенные детали. "
            f"Итоговый текст должен занимать примерно {target_percent}% "
            "от объёма исходной реплики, когда это возможно. "
            "Главная цель — сохранить общий смысл, а не каждое слово. "
            "Не добавляй фактов, которых нет в исходнике. "
            "Не меняй отрицания, имена, числа и причинно-следственные связи. "
            "Убирай повторы, слова-паразиты, лишние вводные фразы и "
            "второстепенные подробности. "
            "Если исходник уже на русском, просто семантически сократи его. "
            "Не пиши объяснений."
        )

        user_prompt = (
            "Переведи реплику на русский и одновременно семантически сократи её.\n\n"
            "ВАЖНО:\n"
            "- Верни только финальный русский текст.\n"
            "- Не добавляй заголовки вроде «Перевод:» или «Ответ:».\n"
            "- Не придумывай информацию.\n"
            "- Не заключай весь ответ в кавычки.\n\n"
            f"ИСХОДНИК:\n{text}"
        )

        # Приводим написанные выше \\n к реальным переводам строк.
        user_prompt = user_prompt.replace("\\n", "\n")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        """Перевести одну реплику на русский."""
        text = normalize_spaces(text)
        result = self._generate(self._build_translation_prompt(text))
        if not result:
            return text
        return result

    @torch.inference_mode()
    def translate_and_compress(
        self,
        text: str,
        target_ratio: float = DEFAULT_TARGET_RATIO,
    ) -> str:
        """Перевести на русский и одновременно семантически сократить."""
        text = normalize_spaces(text)
        result = self._generate(
            self._build_translate_and_compress_prompt(text, target_ratio)
        )
        if not result:
            return text
        return result

    @torch.inference_mode()
    def compress(
        self,
        text: str,
        target_ratio: float = DEFAULT_TARGET_RATIO,
    ) -> str:
        text = normalize_spaces(text)

        result = self._generate(
            self._build_prompt(text, target_ratio)
        )

        # Защита от неудачного ответа модели.
        # Если модель вернула пусто или почти не изменила смысл/объём,
        # безопаснее оставить исходник.
        if not result:
            return text

        # Нельзя допускать случайного "раздувания" текста.
        if word_count(result) > max(word_count(text) * 1.15, word_count(text) + 8):
            return text

        return result


# ---------------------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------------------

def process_srt(
    input_path: Path,
    output_path: Path,
    model_name: str,
    min_words: int,
    target_ratio: float,
    load_in_4bit: bool,
    max_input_tokens: int,
    max_new_tokens: int,
    mode: str,
):
    cues = parse_srt(input_path)

    if not cues:
        raise RuntimeError(
            f"Не удалось найти SRT-блоки в файле: {input_path}"
        )

    print(f"Loaded cues: {len(cues)}")

    compressor = LocalCompressor(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )

    changed = 0
    skipped = 0
    translated = 0
    compressed = 0
    total_before = 0
    total_after = 0

    for position, cue in enumerate(cues, start=1):
        original = normalize_spaces(cue["text"])
        before = word_count(original)

        total_before += before

        total_after += 0

        try:
            if mode == "compress":
                # Старое поведение: короткие реплики пропускаем.
                if before < min_words:
                    result = original
                    skipped += 1
                    print(
                        f"[{position}/{len(cues)}] "
                        f"skip ({before} words)"
                    )
                else:
                    print(
                        f"[{position}/{len(cues)}] "
                        f"compressing ({before} words)..."
                    )
                    result = compressor.compress(
                        original,
                        target_ratio=target_ratio,
                    )
                    compressed += 1

            elif mode == "translate":
                # Переводятся ВСЕ реплики, включая короткие.
                translated += 1
                print(
                    f"[{position}/{len(cues)}] "
                    f"translating ({before} words)..."
                )
                result = compressor.translate(original)

            elif mode == "translate_compress":
                # Переводятся ВСЕ реплики. Длинные дополнительно сокращаются.
                translated += 1
                if before < min_words:
                    print(
                        f"[{position}/{len(cues)}] "
                        f"translating short phrase ({before} words)..."
                    )
                    result = compressor.translate(original)
                else:
                    print(
                        f"[{position}/{len(cues)}] "
                        f"translate + compress ({before} words)..."
                    )
                    result = compressor.translate_and_compress(
                        original,
                        target_ratio=target_ratio,
                    )
                    compressed += 1

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as exc:
            print(
                f"  WARNING: model error: {exc}\n"
                f"  Keeping original text."
            )
            result = original

        result = cleanup_model_output(result)
        after = word_count(result)

        cue["text"] = result
        total_after += after

        if result != original:
            changed += 1

        print(f"  {before} -> {after} words")

        # Периодическая очистка Python/CUDA кешей.
        if position % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_srt(output_path, cues)

    print("\nDone.")
    print(f"Output: {output_path}")
    print(f"Changed:     {changed}")
    print(f"Translated:  {translated}")
    print(f"Compressed:  {compressed}")
    print(f"Skipped:     {skipped}")
    print(f"Words: {total_before} -> {total_after}")

    if total_before > 0:
        ratio = total_after / total_before
        print(f"Final volume: {ratio * 100:.1f}% of original")


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Семантическое сжатие текста в SRT с локальной LLM. "
            "Таймкоды и спикеры сохраняются."
        )
    )

    parser.add_argument(
        "input_srt",
        help="Путь к исходному SRT, например output.srt",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Путь к результирующему SRT. "
            "По умолчанию: <input>_compressed.srt"
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model id. По умолчанию: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--mode",
        choices=("compress", "translate", "translate_compress"),
        default="compress",
        help=(
            "Режим обработки: compress = только сокращение; "
            "translate = перевод каждой реплики на русский; "
            "translate_compress = перевод каждой реплики + "
            "сокращение длинных реплик."
        ),
    )

    parser.add_argument(
        "--translate-ru",
        action="store_true",
        help="Псевдоним для --mode translate.",
    )

    parser.add_argument(
        "--translate-and-compress",
        action="store_true",
        help="Псевдоним для --mode translate_compress.",
    )

    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=(
            "Обрабатывать только реплики не короче N слов. "
            f"По умолчанию: {DEFAULT_MIN_WORDS}"
        ),
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_TARGET_RATIO,
        help=(
            "Желаемый объём результата от исходника. "
            f"0.55 = около 55%%. По умолчанию: {DEFAULT_TARGET_RATIO}"
        ),
    )

    parser.add_argument(
        "--4bit",
        action="store_true",
        dest="load_in_4bit",
        help="Загрузить модель в 4-bit режиме (CUDA + bitsandbytes).",
    )

    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help=f"Максимум входных токенов: {DEFAULT_MAX_INPUT_TOKENS}",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Максимум новых токенов: {DEFAULT_MAX_NEW_TOKENS}",
    )

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input_srt)

    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}")
        sys.exit(1)

    if not 0.1 <= args.ratio <= 0.95:
        print("ERROR: --ratio должен быть между 0.1 и 0.95")
        sys.exit(1)

    if args.min_words < 1:
        print("ERROR: --min-words должен быть >= 1")
        sys.exit(1)

    if args.translate_ru and args.translate_and_compress:
        print(
            "ERROR: нельзя одновременно использовать "
            "--translate-ru и --translate-and-compress."
        )
        sys.exit(1)

    if args.translate_ru:
        mode = "translate"
    elif args.translate_and_compress:
        mode = "translate_compress"
    else:
        mode = args.mode

    if args.output:
        output_path = Path(args.output)
    else:
        if mode == "translate":
            suffix = "_ru"
        elif mode == "translate_compress":
            suffix = "_ru_compressed"
        else:
            suffix = "_compressed"

        output_path = input_path.with_name(
            input_path.stem + suffix + input_path.suffix
        )

    print("=" * 70)
    print("SRT SEMANTIC COMPRESSOR")
    print("=" * 70)
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Model:       {args.model}")
    print(f"Mode:        {mode}")
    print(f"Min words:   {args.min_words}")
    print(f"Target:      {args.ratio * 100:.0f}%")
    print(f"4-bit:       {'yes' if args.load_in_4bit else 'no'}")
    print("=" * 70)

    process_srt(
        input_path=input_path,
        output_path=output_path,
        model_name=args.model,
        min_words=args.min_words,
        target_ratio=args.ratio,
        load_in_4bit=args.load_in_4bit,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        mode=mode,
    )
    subprocess.run([sys.executable, "clean.py"], check=True)


if __name__ == "__main__":
    main()
