"""Журнал очистки: что сделал каждый шаг, со сколькими строками и почему."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import pandas as pd

REMOVE = "remove"
MODIFY = "modify"
COLLAPSE = "collapse"


@dataclass(frozen=True)
class LogEntry:
    """Одна запись журнала: шаг очистки и его результат."""

    step: str
    action: str
    rule: str
    rows_before: int
    rows_after: int
    rows_affected: int
    reason: str


@dataclass
class CleaningLog:
    """Упорядоченный журнал шагов очистки."""

    entries: list[LogEntry] = field(default_factory=list)

    def add(self, entry: LogEntry) -> None:
        """Добавить запись; шаг без обоснования запрещён."""
        if not entry.reason.strip():
            raise ValueError(f"У шага {entry.step} нет обоснования")
        self.entries.append(entry)

    def to_frame(self) -> pd.DataFrame:
        """Журнал в виде таблицы."""
        columns = [item.name for item in fields(LogEntry)]
        return pd.DataFrame([asdict(entry) for entry in self.entries], columns=columns)

    def rows_removed(self) -> int:
        """Сколько строк удалено шагами типа remove."""
        return sum(e.rows_before - e.rows_after for e in self.entries if e.action == REMOVE)

    def is_continuous(self, n_raw: int, n_final: int) -> bool:
        """Каждый шаг начинается с того числа строк, на котором закончился предыдущий."""
        if not self.entries:
            return n_raw == n_final
        chained = all(prev.rows_after == cur.rows_before for prev, cur in zip(self.entries, self.entries[1:]))
        return chained and self.entries[0].rows_before == n_raw and self.entries[-1].rows_after == n_final

    def save(self, path: Path) -> None:
        """Сохранить журнал в CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)
