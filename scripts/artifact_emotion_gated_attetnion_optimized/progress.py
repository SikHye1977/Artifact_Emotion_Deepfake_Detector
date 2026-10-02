"""Flushed, throttled plain-text progress for terminals and redirected logs."""
import time


def duration(seconds):
    if seconds is None:
        return '--:--:--'
    value = max(0, int(seconds))
    return f'{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}'


def status(stage, message):
    print(f'[{stage}] {message}', flush=True)


class Progress:
    def __init__(self, stage, total, unit='samples', interval=5.0):
        self.stage, self.total, self.unit = stage, total, unit
        self.interval = interval
        self.started = self.last_print = time.monotonic()
        self.update(0, force=True)

    def update(self, done, detail='', force=False):
        now = time.monotonic()
        if not (force or done == 1 or done == self.total or now - self.last_print >= self.interval):
            return
        elapsed = now - self.started
        eta = elapsed * (self.total - done) / done if done else None
        percent = 100 * done / self.total if self.total else 100
        rate = done / elapsed if elapsed > 0 else 0
        status(self.stage, f'{done}/{self.total} {self.unit} ({percent:.1f}%) '
               f'elapsed={duration(elapsed)} ETA={duration(eta)} '
               f'rate={rate:.1f} {self.unit}/s' + (f' | {detail}' if detail else ''))
        self.last_print = now
