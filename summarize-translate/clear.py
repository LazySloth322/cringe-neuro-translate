import re

def modify_srt_by_phrase(input_path, output_path, replacement_phrase, max_words=20, encoding='utf-8-sig'):
    try:
        with open(input_path, 'r', encoding=encoding) as f:
            content = f.read()
    except UnicodeDecodeError:
        print("Ошибка кодировки. Попробуйте encoding='cp1251'.")
        return

    # Заменяем "ё" на "е" и "Ё" на "Е" во всем содержимом файла
    content = content.replace('ё', 'е').replace('Ё', 'Е')
    # На всякий случай делаем то же самое для фразы-заменителя
    replacement_phrase = replacement_phrase.replace('ё', 'е').replace('Ё', 'Е')

    content = content.replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\n+', content.strip())
    speaker_pattern = re.compile(r'^(\[Speaker \d+\]:\s*)(.*)$', re.DOTALL)
    new_blocks = []
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 3:
            continue
        index = lines[0]
        timecode = lines[1]
        text = '\n'.join(lines[2:])
        
        match = speaker_pattern.match(text)
        if match:
            prefix = match.group(1)
            phrase = match.group(2)
        else:
            prefix = ""
            phrase = text
            
        words = re.findall(r'\w+', phrase)
        if len(words) > max_words:
            new_text = prefix + replacement_phrase
            print(f"Блок {index}: {len(words)} слов -> Заменено")
        else:
            new_text = text
            
        new_blocks.append(f"{index}\n{timecode}\n{new_text}")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(new_blocks) + '\n')
    print(f"\nГотово! Файл сохранен: {output_path}")

if __name__ == "__main__":
    modify_srt_by_phrase(
        input_path="output_ru_compressed.srt",
        output_path="output_ru_compressed_cleaned.srt",
        replacement_phrase="Подпишись на лалаласкул",
        max_words=20
    )