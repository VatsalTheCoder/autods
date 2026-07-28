"""Tests for the rate limiter and backoff.

All pure -- time and sleep are injected, so a test drives minutes of simulated
traffic through the buckets without a real second passing. That is the only way
to test a rate limiter that is both fast and deterministic.
"""

from __future__ import annotations

import pytest

from app.core.llm.rate_limit import (
    RateLimiter,
    TokenBucket,
    estimate_input_tokens,
    retry_with_backoff,
)


class FakeClock:
    """A monotonic clock a test advances by hand, including when sleep is called."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestTokenBucket:
    def test_starts_full_so_first_burst_is_free(self):
        clock = FakeClock()
        bucket = TokenBucket(
            capacity=10, refill_per_sec=1, time_fn=clock.time, sleep_fn=clock.sleep
        )

        # Ten immediate acquires drain the initial capacity without waiting.
        assert sum(bucket.acquire(1) for _ in range(10)) == 0.0
        assert clock.slept == []

    def test_blocks_once_drained_then_refills(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=2, refill_per_sec=1, time_fn=clock.time, sleep_fn=clock.sleep)

        bucket.acquire(2)  # drain
        waited = bucket.acquire(1)  # must wait 1s for one token to refill

        assert waited == pytest.approx(1.0)
        assert clock.slept == [pytest.approx(1.0)]

    def test_refills_continuously_with_elapsed_time(self):
        clock = FakeClock()
        bucket = TokenBucket(
            capacity=10, refill_per_sec=2, time_fn=clock.time, sleep_fn=clock.sleep
        )
        bucket.acquire(10)  # drain

        clock.now += 3  # 3 seconds pass -> 6 tokens back

        assert bucket.acquire(6) == 0.0

    def test_oversized_request_is_clamped_not_deadlocked(self):
        """A request larger than the whole bucket must not wait forever."""
        clock = FakeClock()
        bucket = TokenBucket(capacity=5, refill_per_sec=5, time_fn=clock.time, sleep_fn=clock.sleep)
        bucket.acquire(5)  # drain

        # Asking for 100 is clamped to capacity (5): one full refill, not 20s.
        waited = bucket.acquire(100)
        assert waited == pytest.approx(1.0)

    def test_wait_time_is_side_effect_free(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=5, refill_per_sec=1, time_fn=clock.time, sleep_fn=clock.sleep)
        bucket.acquire(5)

        # Querying does not consume tokens or advance the clock.
        assert bucket.wait_time(3) == pytest.approx(3.0)
        assert bucket.wait_time(3) == pytest.approx(3.0)
        assert clock.slept == []


class TestRateLimiter:
    def test_input_token_cap_is_the_binding_constraint(self):
        """With generous RPM but tight TPM, the token bucket dominates the wait."""
        clock = FakeClock()
        limiter = RateLimiter(
            requests_per_minute=600,  # 10/s -- never the bottleneck here
            input_tokens_per_minute=600,  # 10 tokens/s
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )
        limiter.acquire(600)  # drain the token bucket

        waited = limiter.acquire(100)  # need 100 tokens back at 10/s -> 10s

        assert waited == pytest.approx(10.0)

    def test_separate_limiters_do_not_share_a_bucket(self):
        clock = FakeClock()
        small = RateLimiter(
            requests_per_minute=1,
            input_tokens_per_minute=10_000,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )
        large = RateLimiter(
            requests_per_minute=1,
            input_tokens_per_minute=10_000,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )
        # Draining small's request bucket must not affect large's.
        small.acquire(1)
        assert large.acquire(1) == 0.0


class RateLimited(Exception):
    """Stand-in for the provider's 429 error."""


class Timeout(Exception):
    """Stand-in for a request that outran its deadline -- the expensive failure."""


class TestRetryWithBackoff:
    def test_returns_immediately_on_success(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        result = retry_with_backoff(
            fn, retries=3, is_retryable=lambda e: True, sleep_fn=lambda s: None
        )
        assert result == "ok"
        assert calls["n"] == 1

    def test_backs_off_exponentially_then_succeeds(self):
        slept: list[float] = []
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RateLimited
            return "recovered"

        result = retry_with_backoff(
            fn,
            retries=5,
            is_retryable=lambda e: isinstance(e, RateLimited),
            sleep_fn=slept.append,
            base_delay=1.0,
            rng=lambda: 0.0,  # no jitter, so the schedule is exact
        )

        assert result == "recovered"
        assert slept == [1.0, 2.0]  # 1*2^0, then 1*2^1

    def test_gives_up_after_the_retry_budget(self):
        def fn():
            raise RateLimited

        with pytest.raises(RateLimited):
            retry_with_backoff(
                fn,
                retries=2,
                is_retryable=lambda e: True,
                sleep_fn=lambda s: None,
                rng=lambda: 0.0,
            )

    def test_non_rate_limit_error_is_not_retried(self):
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            raise ValueError("bad request")

        with pytest.raises(ValueError):
            retry_with_backoff(
                fn,
                retries=5,
                is_retryable=lambda e: isinstance(e, RateLimited),
                sleep_fn=lambda s: None,
            )
        assert attempts["n"] == 1  # failed once, never retried

    def test_delay_is_capped(self):
        slept: list[float] = []
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            if attempts["n"] < 6:
                raise RateLimited
            return "ok"

        retry_with_backoff(
            fn,
            retries=10,
            is_retryable=lambda e: isinstance(e, RateLimited),
            sleep_fn=slept.append,
            base_delay=1.0,
            max_delay=4.0,
            rng=lambda: 0.0,
        )
        # 1, 2, 4, then capped at 4, 4
        assert slept == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_estimate_input_tokens_is_roughly_chars_over_four():
    assert estimate_input_tokens("") == 1  # never zero, so a call always costs something
    assert estimate_input_tokens("a" * 400) == 100


class TestTheTimeBudget:
    """Retries are not equally expensive, and the count alone cannot express that.

    A rate-limit rejection comes back in milliseconds, so five of them cost
    almost nothing. A request that hangs until its deadline costs a full timeout
    *per attempt*, so the same five would stall a pipeline node for minutes. The
    budget is measured from the first call rather than summed over the sleeps,
    which is what makes it count the failed attempts too -- the expensive part.
    """

    def test_a_fast_failure_still_gets_every_retry(self):
        """Nothing changes for the cheap case: the budget is never approached."""
        clock = {"t": 0.0}
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            raise RateLimited

        with pytest.raises(RateLimited):
            retry_with_backoff(
                fn,
                retries=3,
                is_retryable=lambda e: True,
                sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s),
                time_fn=lambda: clock["t"],
                budget_seconds=90.0,
                rng=lambda: 0.0,
            )
        assert attempts["n"] == 4  # the original call plus three retries

    def test_a_slow_failure_is_cut_off_by_the_budget(self):
        """Each attempt burns 60s of wall clock -- the retry count never binds."""
        clock = {"t": 0.0}
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            clock["t"] += 60.0  # the request hangs until its timeout
            raise Timeout

        with pytest.raises(Timeout):
            retry_with_backoff(
                fn,
                retries=5,
                is_retryable=lambda e: True,
                sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s),
                time_fn=lambda: clock["t"],
                budget_seconds=90.0,
                rng=lambda: 0.0,
            )
        # One retry fits inside 90s; a second would start at 121s and is refused.
        assert attempts["n"] == 2
        assert clock["t"] < 130.0

    def test_the_budget_counts_attempts_not_just_sleeps(self):
        """The distinction the whole design rests on.

        Were the budget a sum of backoff delays, the 60s attempts above would be
        invisible to it and five of them would run -- five minutes of a node doing
        nothing, under a policy that believed it had spent three seconds.
        """
        clock = {"t": 0.0}
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            clock["t"] += 30.0
            raise Timeout

        with pytest.raises(Timeout):
            retry_with_backoff(
                fn,
                retries=10,
                is_retryable=lambda e: True,
                sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s),
                time_fn=lambda: clock["t"],
                budget_seconds=90.0,
                rng=lambda: 0.0,
            )
        # 30 + 1 + 30 + 2 + 30 = 93 > 90, so the third attempt is the last.
        assert attempts["n"] == 3

    def test_the_providers_own_error_survives_the_cutoff(self):
        """Giving up must not replace the reason with "we stopped trying"."""
        clock = {"t": 0.0}

        def fn():
            clock["t"] += 100.0
            raise Timeout("upstream did not respond")

        with pytest.raises(Timeout, match="upstream did not respond"):
            retry_with_backoff(
                fn,
                retries=5,
                is_retryable=lambda e: True,
                sleep_fn=lambda s: None,
                time_fn=lambda: clock["t"],
                budget_seconds=90.0,
                rng=lambda: 0.0,
            )

    def test_no_budget_means_the_count_is_the_only_bound(self):
        """Omitting it keeps the previous behaviour exactly."""
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            raise RateLimited

        with pytest.raises(RateLimited):
            retry_with_backoff(
                fn,
                retries=2,
                is_retryable=lambda e: True,
                sleep_fn=lambda s: None,
                rng=lambda: 0.0,
            )
        assert attempts["n"] == 3
