package pool

import (
	"errors"
	"kiro-go/config"
	"path/filepath"
	"testing"
	"time"
)

func TestOverLimitAccountsAreSkippedByDefault(t *testing.T) {
	p := &AccountPool{}
	normal := config.Account{ID: "normal"}
	overLimit := config.Account{ID: "over", UsageCurrent: 10, UsageLimit: 10}

	p.accounts = []config.Account{normal, overLimit}

	for i := 0; i < 5; i++ {
		acc := p.GetNext()
		if acc == nil {
			t.Fatalf("expected an account")
		}
		if acc.ID == "over" {
			t.Fatalf("expected over-limit account to be skipped when upstream OverageStatus is empty")
		}
	}
}

func TestOverLimitAccountsCanBeSelectedWhenUpstreamOverageEnabled(t *testing.T) {
	p := &AccountPool{}
	overLimit := config.Account{
		ID:            "over",
		UsageCurrent:  10,
		UsageLimit:    10,
		OverageStatus: "ENABLED",
	}

	p.accounts = []config.Account{overLimit}

	acc := p.GetNext()
	if acc == nil {
		t.Fatalf("expected upstream-enabled overage account to be selectable")
	}
	if acc.ID != "over" {
		t.Fatalf("expected overage account, got %q", acc.ID)
	}
}

func TestOverLimitAccountsRemainSkippedWhenUpstreamOverageDisabled(t *testing.T) {
	p := &AccountPool{}
	overLimit := config.Account{
		ID:            "over",
		UsageCurrent:  10,
		UsageLimit:    10,
		OverageStatus: "DISABLED",
	}

	p.accounts = []config.Account{overLimit}

	if acc := p.GetNext(); acc != nil {
		t.Fatalf("expected nil when upstream OverageStatus=DISABLED, got %q", acc.ID)
	}
}

func TestGetNextKeepsFiveMinuteTokenAvailable(t *testing.T) {
	p := &AccountPool{}
	account := config.Account{
		ID:          "acct-1",
		AccessToken: "access-token",
		ExpiresAt:   time.Now().Unix() + 300,
	}

	p.accounts = []config.Account{account}

	got := p.GetNext()
	if got == nil {
		t.Fatalf("expected five-minute token to be available")
	}
	if got.ID != account.ID {
		t.Fatalf("expected account %q, got %q", account.ID, got.ID)
	}
}

// ---------------------------------------------------------------------------
// IsAuthFailure
// ---------------------------------------------------------------------------

func TestIsAuthFailureRecognizes401And403(t *testing.T) {
	positives := []string{
		"HTTP 401 from server",
		"received 403 Forbidden",
		"bad credentials",
		"invalid_grant",
		"invalid_token",
		"token expired",
		"token has expired",
		"unauthorized",
	}
	for _, msg := range positives {
		if !IsAuthFailure(errors.New(msg)) {
			t.Errorf("IsAuthFailure(%q) = false, want true", msg)
		}
	}
}

func TestIsAuthFailureIgnoresFalsePositives(t *testing.T) {
	// hasStatusToken only excludes digit boundaries; e.g. "4011" contains "401"
	// but the trailing '1' is a digit so it does NOT match.
	negatives := []string{
		"status code 4011 found", // digit immediately after 401 → not a standalone token
		"error 14013 exceeded",   // digit before and after 401
		"some random error",
		"status 200 OK",
	}
	for _, msg := range negatives {
		if IsAuthFailure(errors.New(msg)) {
			t.Errorf("IsAuthFailure(%q) = true, want false", msg)
		}
	}
}

func TestIsAuthFailureNilError(t *testing.T) {
	if IsAuthFailure(nil) {
		t.Fatal("IsAuthFailure(nil) = true, want false")
	}
}

// ---------------------------------------------------------------------------
// IsSuspensionError
// ---------------------------------------------------------------------------

func TestIsSuspensionErrorDetectsKnownMessages(t *testing.T) {
	positives := []string{
		"account temporarily_suspended",
		"account temporarily suspended",
		"no available kiro profile",
		"No Available Kiro Profile", // case-insensitive
	}
	for _, msg := range positives {
		if !IsSuspensionError(errors.New(msg)) {
			t.Errorf("IsSuspensionError(%q) = false, want true", msg)
		}
	}
}

func TestIsSuspensionErrorIgnoresUnrelatedErrors(t *testing.T) {
	negatives := []string{
		"some other error",
		"unauthorized",
		"429 too many requests",
	}
	for _, msg := range negatives {
		if IsSuspensionError(errors.New(msg)) {
			t.Errorf("IsSuspensionError(%q) = true, want false", msg)
		}
	}
}

func TestIsSuspensionErrorNilError(t *testing.T) {
	if IsSuspensionError(nil) {
		t.Fatal("IsSuspensionError(nil) = true, want false")
	}
}

// ---------------------------------------------------------------------------
// GetNextForModelExcluding
// ---------------------------------------------------------------------------

func newTestPool(accounts ...config.Account) *AccountPool {
	p := &AccountPool{
		cooldowns:   make(map[string]time.Time),
		errorCounts: make(map[string]int),
		modelLists:  make(map[string]map[string]bool),
	}
	p.accounts = accounts
	return p
}

func TestGetNextForModelExcludingSkipsExcludedAccounts(t *testing.T) {
	p := newTestPool(
		config.Account{ID: "a"},
		config.Account{ID: "b"},
	)
	excluded := map[string]bool{"a": true}
	for i := 0; i < 5; i++ {
		acc := p.GetNextForModelExcluding("model", excluded)
		if acc == nil {
			t.Fatal("expected account b, got nil")
		}
		if acc.ID == "a" {
			t.Fatalf("excluded account a was returned on iteration %d", i)
		}
	}
}

func TestGetNextForModelExcludingReturnsNilWhenAllExcluded(t *testing.T) {
	p := newTestPool(config.Account{ID: "only"})
	acc := p.GetNextForModelExcluding("model", map[string]bool{"only": true})
	if acc != nil {
		t.Fatalf("expected nil when only account is excluded, got %q", acc.ID)
	}
}

func TestGetNextForModelExcludingReturnsNilOnEmptyPool(t *testing.T) {
	p := newTestPool()
	acc := p.GetNextForModelExcluding("model", map[string]bool{})
	if acc != nil {
		t.Fatalf("expected nil for empty pool, got %q", acc.ID)
	}
}

// ---------------------------------------------------------------------------
// DisableAccount
// ---------------------------------------------------------------------------

func TestDisableAccountSetsCooldown(t *testing.T) {
	// Initialize a temporary config so SetAccountBanStatus can persist safely.
	cfgFile := filepath.Join(t.TempDir(), "config.json")
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}

	p := newTestPool()
	p.DisableAccount("test-id", "test reason")

	p.mu.RLock()
	cooldown, ok := p.cooldowns["test-id"]
	p.mu.RUnlock()

	if !ok {
		t.Fatal("expected cooldown to be set after DisableAccount")
	}
	// Safety-net cooldown must be at least 23 hours from now.
	minExpected := time.Now().Add(23 * time.Hour)
	if cooldown.Before(minExpected) {
		t.Fatalf("expected cooldown >= 23h in future, got %v", cooldown)
	}
}

func TestGetNextExcludingSkipsExcludedAccount(t *testing.T) {
	p := &AccountPool{
		accounts: []config.Account{
			{ID: "a", Enabled: true},
			{ID: "b", Enabled: true},
		},
		cooldowns:    make(map[string]time.Time),
		errorCounts:  make(map[string]int),
		modelLists:   make(map[string]map[string]bool),
		currentIndex: ^uint64(0),
	}

	acc := p.GetNextExcluding(map[string]bool{"a": true})
	if acc == nil || acc.ID != "b" {
		t.Fatalf("expected account b, got %#v", acc)
	}
}

func TestGetNextForModelExcludingSkipsExcludedAccount(t *testing.T) {
	p := &AccountPool{
		accounts: []config.Account{
			{ID: "a", Enabled: true},
			{ID: "b", Enabled: true},
		},
		cooldowns:    make(map[string]time.Time),
		errorCounts:  make(map[string]int),
		modelLists:   make(map[string]map[string]bool),
		currentIndex: ^uint64(0),
	}
	p.SetModelList("a", []string{"claude-sonnet-4.5"})
	p.SetModelList("b", []string{"claude-sonnet-4.5"})

	acc := p.GetNextForModelExcluding("claude-sonnet-4.5", map[string]bool{"a": true})
	if acc == nil || acc.ID != "b" {
		t.Fatalf("expected account b, got %#v", acc)
	}
}

// ---------------------------------------------------------------------------
// Reload over-usage filtering
// ---------------------------------------------------------------------------

func TestReloadKeepsOverQuotaAccountWhenAllowOverUsage(t *testing.T) {
	cfgFile := filepath.Join(t.TempDir(), "config.json")
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.AddAccount(config.Account{
		ID:           "over",
		Enabled:      true,
		UsageCurrent: 10,
		UsageLimit:   10,
	}); err != nil {
		t.Fatalf("AddAccount: %v", err)
	}
	if err := config.UpdateAllowOverUsage(true); err != nil {
		t.Fatalf("UpdateAllowOverUsage: %v", err)
	}

	p := newTestPool()
	p.Reload()

	if got := p.GetNext(); got == nil || got.ID != "over" {
		t.Fatalf("expected over-quota account to remain routable when allowOverUsage=true, got %#v", got)
	}
}

func TestReloadDropsOverQuotaAccountWhenAllowOverUsageDisabled(t *testing.T) {
	cfgFile := filepath.Join(t.TempDir(), "config.json")
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.AddAccount(config.Account{
		ID:           "over",
		Enabled:      true,
		UsageCurrent: 10,
		UsageLimit:   10,
	}); err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	p := newTestPool()
	p.Reload()

	if got := p.GetNext(); got != nil {
		t.Fatalf("expected over-quota account to be dropped, got %q", got.ID)
	}
}

// --- Regression tests for pool routing fixes ---

// Bug 1: getters must return a copy, not a pointer into the backing array, so
// callers can mutate the result without racing Reload()/UpdateToken.
func TestGetNextReturnsCopyNotBackingPointer(t *testing.T) {
	p := newTestPool(config.Account{ID: "a", Enabled: true, AccessToken: "orig"})

	acc := p.GetNext()
	if acc == nil {
		t.Fatal("expected an account")
	}
	// Mutating the returned account must NOT change the pool's backing array.
	acc.AccessToken = "mutated"

	p.mu.RLock()
	backing := p.accounts[0].AccessToken
	p.mu.RUnlock()
	if backing != "orig" {
		t.Fatalf("mutation leaked into backing array: got %q, want %q", backing, "orig")
	}
}

func TestGetByIDReturnsCopyNotBackingPointer(t *testing.T) {
	p := newTestPool(config.Account{ID: "a", Enabled: true, AccessToken: "orig"})
	acc := p.GetByID("a")
	if acc == nil {
		t.Fatal("expected account a")
	}
	acc.AccessToken = "mutated"
	p.mu.RLock()
	backing := p.accounts[0].AccessToken
	p.mu.RUnlock()
	if backing != "orig" {
		t.Fatalf("GetByID leaked mutation into backing array: got %q", backing)
	}
}

// Bug 2: a non-quota error must never shorten an existing longer (quota) cooldown.
func TestRecordErrorDoesNotShortenQuotaCooldown(t *testing.T) {
	p := newTestPool(config.Account{ID: "a", Enabled: true})

	// Quota error → 1h cooldown.
	p.RecordError("a", true)
	p.mu.RLock()
	afterQuota := p.cooldowns["a"]
	p.mu.RUnlock()

	// Three subsequent non-quota errors would set a 1m cooldown; must not win.
	p.RecordError("a", false)
	p.RecordError("a", false)
	p.RecordError("a", false)

	p.mu.RLock()
	final := p.cooldowns["a"]
	p.mu.RUnlock()

	if final.Before(afterQuota) {
		t.Fatalf("non-quota error shortened quota cooldown: quota=%v final=%v", afterQuota, final)
	}
	// Should still be ~1h out, not ~1m.
	if final.Before(time.Now().Add(30 * time.Minute)) {
		t.Fatalf("expected cooldown to remain ~1h, got %v", final)
	}
}

// Bug 2b: quota errors should not count toward the >=3 transient-error threshold.
func TestRecordErrorQuotaDoesNotCountTowardTransientThreshold(t *testing.T) {
	p := newTestPool(config.Account{ID: "a", Enabled: true})
	// Two quota errors then one non-quota: errorCounts should be 1 (only the
	// non-quota one counts), so no transient cooldown is triggered by count.
	p.RecordError("a", true)
	p.RecordError("a", true)
	p.mu.RLock()
	count := p.errorCounts["a"]
	p.mu.RUnlock()
	if count != 0 {
		t.Fatalf("quota errors must not increment transient counter, got %d", count)
	}
}

// Bug 3: status codes embedded in identifiers must not trigger auth-failure.
func TestHasStatusTokenBoundaries(t *testing.T) {
	cases := []struct {
		msg  string
		want bool
	}{
		{"HTTP 401 from upstream", true},
		{"status (403)", true},
		{"got 401", true},
		{"x-amzn-requestid: request_401abc", false},
		{"status4010", false},
		{"id=x401y", false},
		{"err 4011", false},
	}
	for _, c := range cases {
		if got := IsAuthFailure(errors.New(c.msg)); got != c.want {
			t.Errorf("IsAuthFailure(%q) = %v, want %v", c.msg, got, c.want)
		}
	}
}

// Bug 4: ClearCooldown must remove a lingering cooldown and reset error count so a
// re-enabled account becomes routable again.
func TestClearCooldownReinstatesAccount(t *testing.T) {
	cfgFile := filepath.Join(t.TempDir(), "config.json")
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	p := newTestPool(config.Account{ID: "a", Enabled: true})
	p.RecordError("a", true) // 1h cooldown
	p.errorCounts["a"] = 5

	p.ClearCooldown("a")

	p.mu.RLock()
	_, hasCooldown := p.cooldowns["a"]
	count := p.errorCounts["a"]
	p.mu.RUnlock()
	if hasCooldown {
		t.Fatal("ClearCooldown should remove the cooldown")
	}
	if count != 0 {
		t.Fatalf("ClearCooldown should reset error count, got %d", count)
	}
}

// Bug 5: the degraded fallback path scans from currentIndex (rotation-aware) so
// that, among non-cooling accounts the main loop skipped (e.g. near-expiry tokens),
// it doesn't always hand back the same index-0 account. When EVERY account is
// cooling, however, the fallback intentionally returns the earliest-cooldown
// account — the one soonest to recover — which maximizes availability rather than
// rotating onto an account that is cooling for much longer.
func TestFallbackReturnsEarliestCoolingAccount(t *testing.T) {
	p := newTestPool(
		config.Account{ID: "a", Enabled: true},
		config.Account{ID: "b", Enabled: true},
		config.Account{ID: "c", Enabled: true},
	)
	now := time.Now()
	p.cooldowns["a"] = now.Add(3 * time.Hour)
	p.cooldowns["b"] = now.Add(1 * time.Hour) // earliest → soonest to recover
	p.cooldowns["c"] = now.Add(2 * time.Hour)

	for i := 0; i < 10; i++ {
		acc := p.GetNext()
		if acc == nil {
			t.Fatal("fallback should still return a cooling account, got nil")
		}
		if acc.ID != "b" {
			t.Fatalf("fallback should return the earliest-cooldown account b, got %s", acc.ID)
		}
	}
}

// Bug 5b: in the fallback's non-cooling branch (accounts the main loop skipped due
// to near-expiry tokens, not cooldown), selection rotates via currentIndex instead
// of always returning the first by slice order.
func TestFallbackRotatesAmongNonCoolingSkippedAccounts(t *testing.T) {
	soonExpiry := time.Now().Unix() + tokenRefreshSkewSeconds - 10 // within skew → main loop skips
	p := newTestPool(
		config.Account{ID: "a", Enabled: true, ExpiresAt: soonExpiry},
		config.Account{ID: "b", Enabled: true, ExpiresAt: soonExpiry},
		config.Account{ID: "c", Enabled: true, ExpiresAt: soonExpiry},
	)
	// None are cooling; all are near-expiry so the main loop skips them and the
	// fallback (which ignores expiry) returns them. It should rotate across IDs.
	seen := make(map[string]bool)
	for i := 0; i < 30; i++ {
		acc := p.GetNext()
		if acc == nil {
			t.Fatal("fallback should return a near-expiry account for refresh, got nil")
		}
		seen[acc.ID] = true
	}
	if len(seen) < 2 {
		t.Fatalf("fallback did not rotate among near-expiry accounts: only saw %v", seen)
	}
}
