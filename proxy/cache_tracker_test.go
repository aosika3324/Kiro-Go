package proxy

import (
	"strings"
	"testing"
	"time"
)

// computeUsage runs Compute and converts the returned fractions back to absolute
// token counts against the profile's total input, so tests can assert on
// read/creation token counts the way the handler ultimately reports them.
func computeUsage(t *promptCacheTracker, keyID string, profile *promptCacheProfile) promptCacheUsage {
	fracs := t.Compute(keyID, profile)
	return applyCacheFractions(profile.TotalInputTokens, fracs)
}

func TestPromptCacheTrackerComputeAndUpdate(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	longSystem := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)
	req := &ClaudeRequest{
		Model: "claude-sonnet-4.5",
		System: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": longSystem,
				"cache_control": map[string]interface{}{
					"type": "ephemeral",
				},
			},
		},
		Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
	}

	profile := tracker.BuildClaudeProfile(req, 120)
	if profile == nil {
		t.Fatalf("expected cache profile to be built")
	}

	first := computeUsage(tracker, "key-1", profile)
	if first.CacheCreationInputTokens <= 0 {
		t.Fatalf("expected first request to create cache tokens, got %+v", first)
	}
	if first.CacheReadInputTokens != 0 {
		t.Fatalf("expected first request to have zero cache reads, got %+v", first)
	}

	tracker.Update("key-1", profile)
	second := computeUsage(tracker, "key-1", profile)
	if second.CacheReadInputTokens <= 0 {
		t.Fatalf("expected repeated request to read cache tokens, got %+v", second)
	}
	if second.CacheCreationInputTokens != 0 {
		t.Fatalf("expected repeated request to avoid cache creation, got %+v", second)
	}
}

// TestPromptCacheIsolatedPerKey verifies the cache bucket is keyed by API key,
// not by account: a prefix stored under key-1 must NOT be readable by key-2.
func TestPromptCacheIsolatedPerKey(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	longSystem := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)
	req := &ClaudeRequest{
		Model: "claude-sonnet-4.5",
		System: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": longSystem,
				"cache_control": map[string]interface{}{
					"type": "ephemeral",
				},
			},
		},
		Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
	}
	profile := tracker.BuildClaudeProfile(req, 2048)
	if profile == nil {
		t.Fatalf("expected cache profile to be built")
	}

	tracker.Update("key-1", profile)

	// key-2 has never seen this prefix → must report creation, not a read.
	other := computeUsage(tracker, "key-2", profile)
	if other.CacheReadInputTokens != 0 {
		t.Fatalf("expected no cross-key cache read, got %+v", other)
	}
	if other.CacheCreationInputTokens <= 0 {
		t.Fatalf("expected key-2 to create its own cache entry, got %+v", other)
	}
}

// TestPromptCacheHitAcrossRoundRobinAccounts verifies the simulated cache hits
// regardless of which upstream account served the prior turn: the cache is keyed
// by API key, so account round-robin is irrelevant. Two Update/Compute cycles
// under the same key must produce a read on the second turn.
func TestPromptCacheHitAcrossRoundRobinAccounts(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	systemText := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)
	baseSystem := []interface{}{
		map[string]interface{}{
			"type":          "text",
			"text":          systemText,
			"cache_control": map[string]interface{}{"type": "ephemeral"},
		},
	}

	// Turn 1 served by "account A" — but the tracker only sees the API key.
	req1 := &ClaudeRequest{
		Model:    "claude-sonnet-4.5",
		System:   baseSystem,
		Messages: []ClaudeMessage{{Role: "user", Content: "question one"}},
	}
	profile1 := tracker.BuildClaudeProfile(req1, 2048)
	tracker.Update("shared-key", profile1)

	// Turn 2 served by a DIFFERENT account, same API key and same prefix.
	req2 := &ClaudeRequest{
		Model:  "claude-sonnet-4.5",
		System: baseSystem,
		Messages: []ClaudeMessage{
			{Role: "user", Content: "question one"},
			{Role: "assistant", Content: "answer one"},
			{Role: "user", Content: "follow-up question"},
		},
	}
	profile2 := tracker.BuildClaudeProfile(req2, 4096)
	result := computeUsage(tracker, "shared-key", profile2)
	if result.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read regardless of account round-robin, got %+v", result)
	}
}

func TestBuildClaudeUsageMapIncludesCacheFields(t *testing.T) {
	usage := promptCacheUsage{
		CacheCreationInputTokens:   30,
		CacheReadInputTokens:       20,
		CacheCreation5mInputTokens: 10,
		CacheCreation1hInputTokens: 20,
	}

	m := buildClaudeUsageMap(100, 50, usage, true)

	if got := m["input_tokens"]; got != 50 {
		t.Fatalf("expected billed input tokens 50, got %#v", got)
	}
	if got := m["cache_creation_input_tokens"]; got != 30 {
		t.Fatalf("expected cache creation tokens 30, got %#v", got)
	}
	if got := m["cache_read_input_tokens"]; got != 20 {
		t.Fatalf("expected cache read tokens 20, got %#v", got)
	}
	creation, ok := m["cache_creation"].(map[string]int)
	if !ok {
		t.Fatalf("expected typed cache creation map, got %#v", m["cache_creation"])
	}
	if creation["ephemeral_5m_input_tokens"] != 10 || creation["ephemeral_1h_input_tokens"] != 20 {
		t.Fatalf("unexpected ttl breakdown: %#v", creation)
	}
}

// TestPromptCacheStableAcrossBillingHeaderDrift verifies that Claude Code's
// per-request "x-anthropic-billing-header: cc_version=...; cch=...;" system
// block (whose content drifts on every request) does not break cache hits.
// The tracker should ignore that metadata when fingerprinting cached prefixes.
func TestPromptCacheStableAcrossBillingHeaderDrift(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	mainSystem := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)

	build := func(billingHdr string) *ClaudeRequest {
		return &ClaudeRequest{
			Model: "claude-sonnet-4.5",
			System: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": billingHdr,
				},
				map[string]interface{}{
					"type": "text",
					"text": mainSystem,
					"cache_control": map[string]interface{}{
						"type": "ephemeral",
					},
				},
			},
			Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
		}
	}

	req1 := build("x-anthropic-billing-header: cc_version=2.1.87.1; cch=aaaa;")
	profile1 := tracker.BuildClaudeProfile(req1, 2048)
	if profile1 == nil {
		t.Fatalf("profile1 should be built")
	}
	first := computeUsage(tracker, "key-1", profile1)
	if first.CacheReadInputTokens != 0 {
		t.Fatalf("expected no cache read on first request, got %+v", first)
	}
	tracker.Update("key-1", profile1)

	req2 := build("x-anthropic-billing-header: cc_version=2.1.87.42; cch=bbbb; padding=xxyyzz;")
	profile2 := tracker.BuildClaudeProfile(req2, 2048)
	if profile2 == nil {
		t.Fatalf("profile2 should be built")
	}
	second := computeUsage(tracker, "key-1", profile2)
	if second.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read after billing header drift, got %+v", second)
	}
}

func TestPromptCacheStableWhenBillingHeaderAppearsOrDisappears(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	mainSystem := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)

	build := func(includeBilling bool) *ClaudeRequest {
		system := []interface{}{}
		if includeBilling {
			system = append(system, map[string]interface{}{
				"type": "text",
				"text": "x-anthropic-billing-header: cc_version=2.1.87.1; cch=aaaa;",
			})
		}
		system = append(system, map[string]interface{}{
			"type": "text",
			"text": mainSystem,
			"cache_control": map[string]interface{}{
				"type": "ephemeral",
			},
		})
		return &ClaudeRequest{
			Model:    "claude-sonnet-4.5",
			System:   system,
			Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
		}
	}

	withBilling := tracker.BuildClaudeProfile(build(true), 2048)
	if withBilling == nil {
		t.Fatalf("profile with billing header should be built")
	}
	tracker.Update("key-1", withBilling)

	withoutBilling := tracker.BuildClaudeProfile(build(false), 2048)
	if withoutBilling == nil {
		t.Fatalf("profile without billing header should be built")
	}
	result := computeUsage(tracker, "key-1", withoutBilling)
	if result.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read when billing header disappears, got %+v", result)
	}
}

func TestCanonicalCacheValueIgnoresPositionKeys(t *testing.T) {
	first := canonicalizeCacheValue(stripCachePositionKeys(map[string]interface{}{
		"kind":         "system",
		"system_index": 0,
		"block": map[string]interface{}{
			"type": "text",
			"text": "stable",
		},
	}))
	second := canonicalizeCacheValue(stripCachePositionKeys(map[string]interface{}{
		"kind":         "system",
		"system_index": 1,
		"block": map[string]interface{}{
			"type": "text",
			"text": "stable",
		},
	}))
	if first != second {
		t.Fatalf("expected position keys to be ignored, got %q vs %q", first, second)
	}
}

func TestCanonicalCacheValuePreservesSemanticPositionKeys(t *testing.T) {
	first := canonicalizeCacheValue(map[string]interface{}{
		"kind": "system",
		"block": map[string]interface{}{
			"type":        "text",
			"text":        "stable",
			"block_index": 1,
		},
	})
	second := canonicalizeCacheValue(map[string]interface{}{
		"kind": "system",
		"block": map[string]interface{}{
			"type":        "text",
			"text":        "stable",
			"block_index": 2,
		},
	})
	if first == second {
		t.Fatalf("expected semantic block_index fields to remain fingerprinted")
	}
}

// TestPromptCacheImplicitBreakpointAtMessageEnd verifies that once any
// explicit cache_control breakpoint has been seen, subsequent message-end
// boundaries act as implicit breakpoints. This allows multi-turn conversations
// to hit earlier stored prefix fingerprints even when the newest messages
// lack explicit cache_control.
func TestPromptCacheImplicitBreakpointAtMessageEnd(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	systemText := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)

	baseSystem := []interface{}{
		map[string]interface{}{
			"type": "text",
			"text": systemText,
			"cache_control": map[string]interface{}{
				"type": "ephemeral",
			},
		},
	}

	// Round 1: single user message.
	req1 := &ClaudeRequest{
		Model:    "claude-sonnet-4.5",
		System:   baseSystem,
		Messages: []ClaudeMessage{{Role: "user", Content: "question one"}},
	}
	profile1 := tracker.BuildClaudeProfile(req1, 2048)
	if profile1 == nil {
		t.Fatalf("profile1 should be built")
	}
	tracker.Update("key-1", profile1)

	// Round 2: conversation continues with new messages. The latest user
	// message has no explicit cache_control; it should still hit the stored
	// prefix via the implicit message-end breakpoint.
	req2 := &ClaudeRequest{
		Model:  "claude-sonnet-4.5",
		System: baseSystem,
		Messages: []ClaudeMessage{
			{Role: "user", Content: "question one"},
			{Role: "assistant", Content: "answer one"},
			{Role: "user", Content: "follow-up question"},
		},
	}
	profile2 := tracker.BuildClaudeProfile(req2, 4096)
	if profile2 == nil {
		t.Fatalf("profile2 should be built")
	}
	result := computeUsage(tracker, "key-1", profile2)
	if result.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read via implicit message-end breakpoint, got %+v", result)
	}
}

// TestApplyCacheFractionsConservesTokens verifies cache_read + cache_creation +
// billed always equals the input-token basis, and the TTL tier breakdown sums to
// the creation total.
func TestApplyCacheFractionsConservesTokens(t *testing.T) {
	fracs := cacheFractions{
		ReadFrac:       0.6,
		CreationFrac:   0.2,
		Creation5mFrac: 0.15,
		Creation1hFrac: 0.05,
	}
	usage := applyCacheFractions(1000, fracs)
	billed := billedClaudeInputTokens(1000, usage)
	if billed+usage.CacheReadInputTokens+usage.CacheCreationInputTokens != 1000 {
		t.Fatalf("token conservation violated: billed=%d read=%d creation=%d",
			billed, usage.CacheReadInputTokens, usage.CacheCreationInputTokens)
	}
	if usage.CacheCreation5mInputTokens+usage.CacheCreation1hInputTokens != usage.CacheCreationInputTokens {
		t.Fatalf("ttl breakdown must sum to creation total: 5m=%d 1h=%d creation=%d",
			usage.CacheCreation5mInputTokens, usage.CacheCreation1hInputTokens, usage.CacheCreationInputTokens)
	}
}

// TestApplyCacheFractionsNeverExceedsInput verifies read + creation are clamped
// so the downstream-billed remainder is never negative even with noisy fractions.
func TestApplyCacheFractionsNeverExceedsInput(t *testing.T) {
	fracs := cacheFractions{ReadFrac: 0.9, CreationFrac: 0.9}
	usage := applyCacheFractions(1000, fracs)
	if usage.CacheReadInputTokens+usage.CacheCreationInputTokens > 1000 {
		t.Fatalf("read+creation exceeded input: read=%d creation=%d",
			usage.CacheReadInputTokens, usage.CacheCreationInputTokens)
	}
}

func TestApplyCacheFractionsZeroInputSafe(t *testing.T) {
	usage := applyCacheFractions(0, cacheFractions{ReadFrac: 0.9})
	if usage.CacheReadInputTokens != 0 || usage.CacheCreationInputTokens != 0 {
		t.Fatalf("zero input must report no cache, got %+v", usage)
	}
}

func TestApplyCacheFractionsNoDataReportsNothing(t *testing.T) {
	usage := applyCacheFractions(1000, cacheFractions{})
	if usage.CacheReadInputTokens != 0 || usage.CacheCreationInputTokens != 0 {
		t.Fatalf("empty fractions must report no cache, got %+v", usage)
	}
}

// TestBuildOpenAIProfileMultiTurnHits verifies the OpenAI profile builder
// produces breakpoints that hit across turns: a long stable system message plus
// growing conversation should read cache on the second turn under the same key.
func TestBuildOpenAIProfileMultiTurnHits(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	systemText := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)

	req1 := &OpenAIRequest{
		Model: "gpt-4o",
		Messages: []OpenAIMessage{
			{Role: "system", Content: systemText},
			{Role: "user", Content: "question one"},
		},
	}
	profile1 := tracker.BuildOpenAIProfile(req1, 2048)
	if profile1 == nil {
		t.Fatalf("expected OpenAI profile to be built")
	}
	first := computeUsage(tracker, "key-1", profile1)
	if first.CacheReadInputTokens != 0 {
		t.Fatalf("expected no cache read on first OpenAI turn, got %+v", first)
	}
	tracker.Update("key-1", profile1)

	req2 := &OpenAIRequest{
		Model: "gpt-4o",
		Messages: []OpenAIMessage{
			{Role: "system", Content: systemText},
			{Role: "user", Content: "question one"},
			{Role: "assistant", Content: "answer one"},
			{Role: "user", Content: "follow-up question"},
		},
	}
	profile2 := tracker.BuildOpenAIProfile(req2, 4096)
	if profile2 == nil {
		t.Fatalf("expected second OpenAI profile to be built")
	}
	second := computeUsage(tracker, "key-1", profile2)
	if second.CacheReadInputTokens == 0 {
		t.Fatalf("expected OpenAI cache read on second turn, got %+v", second)
	}
}

// TestMinCacheableTokensForModel pins the model-specific minimum cacheable
// prompt length so the opus-everything-4096 bug doesn't regress.
func TestMinCacheableTokensForModel(t *testing.T) {
	cases := map[string]int{
		"claude-opus-4.8":          1024,
		"claude-sonnet-4.6":        1024,
		"claude-sonnet-4.5":        1024,
		"claude-sonnet-4":          1024,
		"claude-opus-4.7":          2048,
		"claude-opus-4.7-thinking": 2048,
		"claude-opus-4.6":          4096,
		"claude-opus-4.5":          4096,
		"claude-haiku-4.5":         4096,
		"some-unknown-model":       1024,
	}
	for model, want := range cases {
		if got := minCacheableTokensForModel(model); got != want {
			t.Errorf("minCacheableTokensForModel(%q) = %d, want %d", model, got, want)
		}
	}
}
