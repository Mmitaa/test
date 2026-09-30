"""Потоковое извлечение строкового поля из JSON, который модель ещё дописывает.

Модель отвечает строгим JSON (structured outputs), а поле `reply` в схеме стоит первым.
Этот парсер отдаёт текст ответа клиенту по кусочкам сразу, как только они приходят,
не дожидаясь конца JSON — так клиент видит первые слова через доли секунды.
"""

from __future__ import annotations

import json
import re

_DECODER = json.JSONDecoder(strict=False)


class JsonStringFieldStreamer:
    def __init__(self, field: str) -> None:
        self._key_re = re.compile(r'"' + re.escape(field) + r'"\s*:\s*"')
        self._buf = ""
        self._start: int | None = None  # начало значения строки в буфере
        self._pos = 0  # до этого места значение уже декодировано и отдано
        self.done = False

    @property
    def started(self) -> bool:
        return self._start is not None

    def feed(self, chunk: str) -> str:
        """Добавляет кусок JSON и возвращает новый декодированный текст поля (может быть пустым)."""
        if self.done:
            return ""
        self._buf += chunk
        if self._start is None:
            m = self._key_re.search(self._buf)
            if not m:
                return ""
            self._start = self._pos = m.end()

        buf, i, n = self._buf, self._pos, len(self._buf)
        safe = i
        while i < n:
            c = buf[i]
            if c == "\\":
                if i + 1 >= n:
                    break
                if buf[i + 1] == "u":
                    if i + 6 > n:
                        break
                    try:
                        code = int(buf[i + 2 : i + 6], 16)
                    except ValueError:
                        code = 0
                    # Суррогатная пара (эмодзи) декодируется только целиком.
                    if 0xD800 <= code <= 0xDBFF:
                        if i + 12 > n:
                            break
                        i += 12
                    else:
                        i += 6
                else:
                    i += 2
            elif c == '"':
                self.done = True
                break
            else:
                i += 1
            safe = i

        segment = buf[self._pos : safe]
        self._pos = safe
        if not segment:
            return ""
        try:
            return _DECODER.decode('"' + segment + '"')
        except json.JSONDecodeError:
            return segment
