"""Token-bucket rate limiting for the free-tier API.

It also holds the reactive half of the same job: ``retry_with_backoff``, which
survives the failures throttling cannot prevent -- the 429 that slips through, and
the 5xx or timeout that has nothing to do with quota at all.

The Google AI Studio free tier caps us on two axes at once (spec section 6.3):
requests per minute and *input* tokens per minute. The second is the binding
one -- a couple of Critic/Report prompts can blow the 15K input-TPM ceiling
long before the RPM ceiling. So a single request must clear *both* buckets
before it goes out.

A token bucket (rather than a fixed window) is the right shape here: it permits
a short burst up to the bucket size, then throttles to the steady refill rate,
which matches how these APIs actually meter. The pipeline is sequential with
real ML compute between calls, so most of the time nothing waits at all; the
limiter only earns its keep during the LLM-dense stretches (Feature
Engineering's per-column calls, the Critic/Report pair).

Time and sleep are injected, never taken from the global clock directly. That
is what makes this testable: a test drives a year of simulated traffic through
the bucket in microseconds and asserts on the wait it *would* have imposed,
without a single real second passing.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class TokenBucket:
    """A classic token bucket.

    ``capacity`` tokens accrue at ``refill_per_sec`` up to the cap. Acquiring
    ``n`` tokens blocks until ``n`` are available. Because refill is continuous,
    a request larger than a full bucket would wait forever -- so we clamp any
    single acquire to at most ``capacity`` and document it, rather than deadlock.
    """

    capacity: float
    refill_per_sec: float
    _tokens: float = 0.0
    _last: float = 0.0
    # Injected so tests run in simulated time. Defaults are the real clock.
    time_fn: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        # Start full: the first burst should be allowed immediately, exactly as
        # a fresh minute's quota would be.
        self._tokens = self.capacity
        self._last = self.time_fn()

    def _refill(self, now: float) -> None:
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
            self._last = now

    def wait_time(self, amount: float) -> float:
        """Seconds until ``amount`` tokens would be available -- without waiting.

        Pure query, used by tests and by the combined limiter to decide which of
        its two buckets is the binding one. Does not mutate the bucket.
        """
        now = self.time_fn()
        # Snapshot the refill without committing it, so this stays side-effect
        # free: a caller can ask "how long?" without consuming the answer.
        accrued = max(0.0, now - self._last) * self.refill_per_sec
        available = min(self.capacity, self._tokens + accrued)
        want = min(amount, self.capacity)
        if available >= want:
            return 0.0
        return (want - available) / self.refill_per_sec

    def acquire(self, amount: float) -> float:
        """Consume ``amount`` tokens, sleeping if necessary. Returns time waited.

        ``amount`` is clamped to ``capacity``: a prompt larger than a whole
        minute's token budget is throttled to one-per-refill rather than
        wedging the pipeline. Sizing prompts to stay under the cap is Section
        9's job (prompt shrinking); the limiter just refuses to deadlock.
        """
        want = min(amount, self.capacity)
        now = self.time_fn()
        self._refill(now)
        if self._tokens >= want:
            self._tokens -= want
            return 0.0
        deficit = want - self._tokens
        wait = deficit / self.refill_per_sec
        self.sleep_fn(wait)
        # After sleeping, roll time forward and settle up. We set tokens to zero
        # rather than re-deriving, because we waited exactly long enough for
        # `want` and are consuming all of it.
        self._last = self.time_fn()
        self._tokens = 0.0
        return wait


class RateLimiter:
    """Guards a single model tier against both of its free-tier caps.

    One of these exists per tier (SMALL and LARGE meter on separate buckets --
    spec section 6.3). Thread-safe because a Celery worker may run with a small
    thread pool and share one limiter across concurrent calls; the lock keeps
    the two buckets consistent with each other.
    """

    def __init__(
        self,
        *,
        requests_per_minute: int,
        input_tokens_per_minute: int,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._requests = TokenBucket(
            capacity=requests_per_minute,
            refill_per_sec=requests_per_minute / 60.0,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        self._input_tokens = TokenBucket(
            capacity=input_tokens_per_minute,
            refill_per_sec=input_tokens_per_minute / 60.0,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        self._lock = threading.Lock()

    def acquire(self, estimated_input_tokens: int) -> float:
        """Block until one request of this size may go out. Returns time waited.

        Clears the request bucket first (the cheaper, coarser cap), then the
        input-token bucket. Both must be satisfied, so the total wait is at
        least the larger of the two.
        """
        with self._lock:
            waited = self._requests.acquire(1)
            waited += self._input_tokens.acquire(max(1, estimated_input_tokens))
            return waited


def retry_with_backoff[R](
    fn: Callable[[], R],
    *,
    retries: int,
    is_retryable: Callable[[Exception], bool],
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
    budget_seconds: float | None = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    rng: Callable[[], float] = random.random,
) -> R:
    """Call ``fn``, retrying with exponential backoff on failures worth retrying.

    The token bucket keeps us *under* the rate cap proactively; this is the
    safety net for everything reactive -- the 429 that slips through a burst or a
    shared quota (spec section 10 mandates exponential backoff on those), and the
    503s, gateway timeouts and dropped connections that no amount of proactive
    throttling can prevent.

    Only exceptions ``is_retryable`` recognises are retried. A 400 for a
    malformed request must fail immediately: it will fail identically five times
    and the only thing retrying buys is a slower error message.

    **Two bounds, because retries are not equally expensive.** ``retries`` caps
    the attempts, and ``budget_seconds`` caps the *elapsed time* they may add --
    measured from the first call, so it counts the failed attempts and not just
    the sleeps between them. That distinction is the whole point of the second
    bound: a rate-limit rejection returns in milliseconds and can afford five
    tries, while a request that hangs until its deadline costs a full timeout
    every time and can afford about one. One budget expresses both, and the shape
    of the failure decides how many attempts it earns.

    ``sleep_fn``, ``time_fn`` and ``rng`` are injected so the whole schedule is
    asserted in tests without a real second elapsing or a network to fail against.
    """
    started = time_fn()
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not is_retryable(exc) or attempt >= retries:
                raise
            delay = min(max_delay, base_delay * (2**attempt))
            delay += rng() * delay * 0.1  # +0-10% jitter, so retries desynchronise
            if budget_seconds is not None and (time_fn() - started) + delay >= budget_seconds:
                # Give up now rather than starting an attempt the budget cannot
                # cover. Raising the provider's own exception keeps the reason
                # for the failure intact -- the caller reports what went wrong,
                # not merely that we stopped trying.
                raise
            sleep_fn(delay)
            attempt += 1


def estimate_input_tokens(text: str) -> int:
    """Rough token count for rate-limiting purposes only.

    Deliberately crude -- ~4 characters per token is the well-worn rule of
    thumb, and it is plenty for *throttling*, where over-counting a little just
    means we stay comfortably under the cap. The numbers written to
    ``token_usage`` for cost come from the provider's real usage metadata, not
    from this; this only decides how long to wait before sending.
    """
    return max(1, len(text) // 4)
