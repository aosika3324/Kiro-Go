package proxy

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const defaultPromptCacheTTL = 5 * time.Minute

// globalCacheBucket is the sentinel bucket key used when a request has no
// associated API key (legacy single-key mode / unauthenticated path). All such
// requests share one simulated cache bucket.
const globalCacheBucket = "__global__"

// Anthropic requires cached prefixes to reach a minimum token count before
// caching takes effect. Breakpoints below this threshold are excluded from
// matching and storage to avoid reporting unrealistic cache hits on short
// requests. The threshold is model-specific (see minCacheableTokensForModel).
const defaultMinCacheableTokens = 1024

type promptCacheUsage struct {
	CacheCreationInputTokens   int
	CacheReadInputTokens       int
	CacheCreation5mInputTokens int
	CacheCreation1hInputTokens int
}

type promptCacheBreakpoint struct {
	Fingerprint      [32]byte
	CumulativeTokens int
	TTL              time.Duration
}

type promptCacheProfile struct {
	Breakpoints      []promptCacheBreakpoint
	TotalInputTokens int
	Model            string
}

// minCacheableTokensForModel returns the minimum prompt length (in tokens) a
// prefix must reach before Anthropic will cache it. The value is model-specific;
// prefixes shorter than this never produce a cache hit. Matching is done on
// lowercased substrings of the model name so suffixes (e.g. "-thinking") and
// vendor prefixes don't break the lookup.
func minCacheableTokensForModel(model string) int {
	lower := strings.ToLower(model)
	switch {
	// 4,096-token minimum.
	case strings.Contains(lower, "haiku-4.5"),
		strings.Contains(lower, "opus-4.6"),
		strings.Contains(lower, "opus-4.5"):
		return 4096
	// 2,048-token minimum.
	case strings.Contains(lower, "opus-4.7"),
		strings.Contains(lower, "haiku-3.5"):
		return 2048
	// Everything else (opus-4.8, sonnet-4.6/4.5/4, opus-4.1/4, …) uses the
	// 1,024-token default.
	default:
		return defaultMinCacheableTokens
	}
}

type promptCacheEntry struct {
	ExpiresAt time.Time
	TTL       time.Duration
}

// promptCacheTracker simulates Anthropic prompt-cache read/write behavior. The
// cache unit is the API key (one client / one conversation), NOT the upstream
// Kiro account: the account pool round-robins requests, so keying the simulated
// cache by account would never produce a hit across turns. Keying by API key
// mirrors Anthropic's "cache isolated per organization / workspace" model and
// lets a multi-turn conversation hit prefixes stored by its own earlier turns.
type promptCacheTracker struct {
	mu              sync.Mutex
	entriesByKey    map[string]map[[32]byte]promptCacheEntry
	maxSupportedTTL time.Duration
}

func newPromptCacheTracker(maxTTL time.Duration) *promptCacheTracker {
	if maxTTL <= 0 {
		maxTTL = defaultPromptCacheTTL
	}
	return &promptCacheTracker{
		entriesByKey:    make(map[string]map[[32]byte]promptCacheEntry),
		maxSupportedTTL: maxTTL,
	}
}

// bucketKey maps an API key ID to its cache bucket, collapsing the empty key
// (legacy / unauthenticated path) onto the shared global sentinel bucket.
func bucketKey(apiKeyID string) string {
	if apiKeyID == "" {
		return globalCacheBucket
	}
	return apiKeyID
}

func (t *promptCacheTracker) BuildClaudeProfile(req *ClaudeRequest, totalInputTokens int) *promptCacheProfile {
	blocks := flattenClaudeCacheBlocks(req)
	return buildCacheProfileFromBlocks(blocks, req.Model, totalInputTokens)
}

// BuildOpenAIProfile builds a prompt-cache profile from an OpenAI chat request,
// mirroring BuildClaudeProfile so the OpenAI endpoints get the same simulated
// cache read/write behavior as the Claude endpoints. The cache prefix hierarchy
// is tools → messages (system messages are ordinary role="system" messages in
// the OpenAI schema, so they fall into the message stream naturally).
func (t *promptCacheTracker) BuildOpenAIProfile(req *OpenAIRequest, totalInputTokens int) *promptCacheProfile {
	blocks := flattenOpenAICacheBlocks(req)
	return buildCacheProfileFromBlocks(blocks, req.Model, totalInputTokens)
}

// buildCacheProfileFromBlocks turns an ordered list of cacheable prompt blocks
// into a profile of cumulative-prefix breakpoints. Shared by the Claude and
// OpenAI profile builders.
func buildCacheProfileFromBlocks(blocks []cacheablePromptBlock, model string, totalInputTokens int) *promptCacheProfile {
	if len(blocks) == 0 {
		return nil
	}

	hasher := sha256.New()
	breakpoints := make([]promptCacheBreakpoint, 0)
	cumulativeTokens := 0
	var activeTTL time.Duration

	for _, block := range blocks {
		canonical := canonicalizeCacheValue(block.Value)
		writeHashChunk(hasher, canonical)
		cumulativeTokens += block.Tokens

		// Determine whether this block acts as a cache breakpoint:
		//   1) Explicit cache_control on the block itself.
		//   2) Once any explicit breakpoint has been seen, every message-end
		//      boundary becomes an implicit breakpoint so that multi-turn
		//      conversations can hit earlier stored prefixes.
		//   3) For requests that carry no explicit cache_control at all (the
		//      common OpenAI case), every message-end boundary is an implicit
		//      breakpoint so multi-turn conversations still simulate caching.
		breakpointTTL := time.Duration(0)
		if block.TTL > 0 {
			breakpointTTL = block.TTL
			activeTTL = block.TTL
		} else if block.IsMessageEnd && activeTTL > 0 {
			breakpointTTL = activeTTL
		} else if block.IsMessageEnd && block.ImplicitBreakpoint {
			breakpointTTL = defaultPromptCacheTTL
		}

		if breakpointTTL <= 0 {
			continue
		}

		var fingerprint [32]byte
		copy(fingerprint[:], hasher.Sum(nil))
		breakpoints = append(breakpoints, promptCacheBreakpoint{
			Fingerprint:      fingerprint,
			CumulativeTokens: cumulativeTokens,
			TTL:              breakpointTTL,
		})
	}

	if len(breakpoints) == 0 {
		return nil
	}

	if totalInputTokens < cumulativeTokens {
		totalInputTokens = cumulativeTokens
	}

	return &promptCacheProfile{
		Breakpoints:      breakpoints,
		TotalInputTokens: totalInputTokens,
		Model:            model,
	}
}

// cacheFractions expresses simulated cache attribution as proportions of the
// request's total input tokens, decoupled from any specific token count. The
// handler resolves the final input-token basis (the estimator value, or the
// upstream context-usage figure when available) and only then converts these
// fractions to absolute token counts via applyCacheFractions. This avoids the
// pre-rewrite bug where cache_read was computed against the estimator basis but
// reported alongside a different (context-usage) input basis, letting the
// reported hit ratio drift above 100%.
type cacheFractions struct {
	ReadFrac       float64
	CreationFrac   float64
	Creation5mFrac float64
	Creation1hFrac float64
}

// hasData reports whether the simulation produced any read or creation
// attribution. Used to decide whether to report cache fields at all.
func (f cacheFractions) hasData() bool {
	return f.ReadFrac > 0 || f.CreationFrac > 0
}

// Compute simulates the prompt-cache read/write split for a request, returning
// proportions of total input tokens. It is read-only: TTL refresh and entry
// storage happen in Update, called only after a request succeeds.
func (t *promptCacheTracker) Compute(apiKeyID string, profile *promptCacheProfile) cacheFractions {
	if t == nil || profile == nil || len(profile.Breakpoints) == 0 || profile.TotalInputTokens <= 0 {
		return cacheFractions{}
	}

	minTokens := minCacheableTokensForModel(profile.Model)
	total := profile.TotalInputTokens
	last := profile.Breakpoints[len(profile.Breakpoints)-1]
	lastTokens := minInt(last.CumulativeTokens, total)
	now := time.Now()

	t.mu.Lock()
	defer t.mu.Unlock()
	t.pruneExpiredLocked(now)

	entries := t.entriesByKey[bucketKey(apiKeyID)]
	if len(entries) == 0 {
		// First request for this key: the whole cacheable prefix is written
		// (cache_creation), nothing is read. Below the model threshold nothing
		// caches at all.
		if lastTokens < minTokens {
			return cacheFractions{}
		}
		cache5m, cache1h := computePromptCacheTTLBreakdown(profile, 0)
		return fractionsFromTokens(total, 0, lastTokens, cache5m, cache1h)
	}

	// Cap cacheable tokens at 85% of total input so the newest content in a
	// request is never fully served from cache on the current turn.
	maxCacheable := int(float64(total) * 0.85)
	if lastTokens > maxCacheable {
		lastTokens = maxCacheable
	}

	matchedTokens := 0
	for i := len(profile.Breakpoints) - 1; i >= 0; i-- {
		breakpoint := profile.Breakpoints[i]
		if breakpoint.CumulativeTokens < minTokens {
			continue
		}
		entry, ok := entries[breakpoint.Fingerprint]
		if !ok || entry.ExpiresAt.Before(now) {
			continue
		}
		matchedTokens = minInt(breakpoint.CumulativeTokens, total)
		if matchedTokens > lastTokens {
			matchedTokens = lastTokens
		}
		break
	}

	creation := maxInt(lastTokens-matchedTokens, 0)
	cache5m, cache1h := computePromptCacheTTLBreakdown(profile, matchedTokens)
	return fractionsFromTokens(total, matchedTokens, creation, cache5m, cache1h)
}

// fractionsFromTokens converts absolute simulated token counts (computed against
// the profile's total input) into proportions of that total.
func fractionsFromTokens(total, read, creation, creation5m, creation1h int) cacheFractions {
	if total <= 0 {
		return cacheFractions{}
	}
	denom := float64(total)
	return cacheFractions{
		ReadFrac:       float64(read) / denom,
		CreationFrac:   float64(creation) / denom,
		Creation5mFrac: float64(creation5m) / denom,
		Creation1hFrac: float64(creation1h) / denom,
	}
}

// applyCacheFractions converts cache proportions to absolute token counts against
// the final input-token basis the handler settled on. The result always satisfies
// cache_read + cache_creation <= inputTokens, so the downstream-billed remainder
// (inputTokens - read - creation) is never negative.
func applyCacheFractions(inputTokens int, f cacheFractions) promptCacheUsage {
	if inputTokens <= 0 || !f.hasData() {
		return promptCacheUsage{}
	}
	read := int(float64(inputTokens)*f.ReadFrac + 0.5)
	creation := int(float64(inputTokens)*f.CreationFrac + 0.5)
	if read < 0 {
		read = 0
	}
	if creation < 0 {
		creation = 0
	}
	if read > inputTokens {
		read = inputTokens
	}
	if read+creation > inputTokens {
		creation = inputTokens - read
	}
	// Split creation across TTL tiers in the same proportion the profile carried.
	create5m := int(float64(inputTokens)*f.Creation5mFrac + 0.5)
	create1h := int(float64(inputTokens)*f.Creation1hFrac + 0.5)
	// Reconcile the tier breakdown with the (possibly clamped) creation total so
	// ephemeral_5m + ephemeral_1h == cache_creation_input_tokens.
	if create5m+create1h != creation {
		if create1h > creation {
			create1h = creation
		}
		create5m = creation - create1h
		if create5m < 0 {
			create5m = 0
			create1h = creation
		}
	}
	return promptCacheUsage{
		CacheReadInputTokens:       read,
		CacheCreationInputTokens:   creation,
		CacheCreation5mInputTokens: create5m,
		CacheCreation1hInputTokens: create1h,
	}
}

func (t *promptCacheTracker) Update(apiKeyID string, profile *promptCacheProfile) {
	if t == nil || profile == nil || len(profile.Breakpoints) == 0 {
		return
	}

	minTokens := minCacheableTokensForModel(profile.Model)
	now := time.Now()
	t.mu.Lock()
	defer t.mu.Unlock()
	t.pruneExpiredLocked(now)

	key := bucketKey(apiKeyID)
	entries := t.entriesByKey[key]
	if entries == nil {
		entries = make(map[[32]byte]promptCacheEntry)
		t.entriesByKey[key] = entries
	}

	for _, breakpoint := range profile.Breakpoints {
		// Skip breakpoints below the minimum cacheable token threshold.
		if breakpoint.CumulativeTokens < minTokens {
			continue
		}
		// Store new prefixes and refresh the TTL of existing ones (Anthropic
		// refreshes the cache at no cost on each use — a sliding window).
		entries[breakpoint.Fingerprint] = promptCacheEntry{
			ExpiresAt: now.Add(breakpoint.TTL),
			TTL:       breakpoint.TTL,
		}
	}
}

func (t *promptCacheTracker) pruneExpiredLocked(now time.Time) {
	for key, entries := range t.entriesByKey {
		for fingerprint, entry := range entries {
			if !entry.ExpiresAt.After(now) {
				delete(entries, fingerprint)
			}
		}
		if len(entries) == 0 {
			delete(t.entriesByKey, key)
		}
	}
}

type cacheablePromptBlock struct {
	Value        interface{}
	Tokens       int
	TTL          time.Duration
	IsMessageEnd bool
	// ImplicitBreakpoint marks a message-end block that should act as a cache
	// breakpoint even when no explicit cache_control appears anywhere in the
	// request. The Claude path leaves this false (it relies on explicit
	// cache_control + activeTTL propagation); the OpenAI path sets it true on
	// every message end so multi-turn conversations simulate caching despite the
	// OpenAI schema having no cache_control concept.
	ImplicitBreakpoint bool
}

func flattenClaudeCacheBlocks(req *ClaudeRequest) []cacheablePromptBlock {
	blocks := make([]cacheablePromptBlock, 0)
	blocks = append(blocks, buildCachePreludeBlock(req))

	for toolIndex, tool := range req.Tools {
		toolValue := map[string]interface{}{
			"kind":         "tool",
			"tool_index":   toolIndex,
			"name":         tool.Name,
			"description":  tool.Description,
			"input_schema": tool.InputSchema,
		}
		fingerprintValue := stripCachePositionKeys(toolValue)
		blocks = append(blocks, cacheablePromptBlock{
			Value:  fingerprintValue,
			Tokens: estimateApproxTokens(canonicalizeCacheValue(fingerprintValue)),
			TTL:    normalizePromptCacheTTL(extractPromptCacheTTL(tool)),
		})
	}

	appendSystemCacheBlocks(&blocks, req.System)

	for messageIndex, msg := range req.Messages {
		appendMessageCacheBlocks(&blocks, messageIndex, msg)
	}

	return blocks
}

// flattenOpenAICacheBlocks builds the ordered cacheable-prefix block list for an
// OpenAI chat request, mirroring flattenClaudeCacheBlocks. Hierarchy: prelude →
// tools → messages. Every message-end block is marked as an implicit breakpoint
// (OpenAI has no cache_control concept) so multi-turn conversations simulate
// caching against prefixes their earlier turns stored.
func flattenOpenAICacheBlocks(req *OpenAIRequest) []cacheablePromptBlock {
	blocks := make([]cacheablePromptBlock, 0)

	prelude := map[string]interface{}{
		"kind":  "request_prelude",
		"model": req.Model,
	}
	blocks = append(blocks, cacheablePromptBlock{
		Value:  prelude,
		Tokens: estimateApproxTokens(canonicalizeCacheValue(prelude)),
	})

	for toolIndex, tool := range req.Tools {
		toolValue := map[string]interface{}{
			"kind":        "tool",
			"tool_index":  toolIndex,
			"tool_type":   tool.Type,
			"name":        tool.Function.Name,
			"description": tool.Function.Description,
			"parameters":  tool.Function.Parameters,
		}
		fingerprintValue := stripCachePositionKeys(toolValue)
		blocks = append(blocks, cacheablePromptBlock{
			Value:  fingerprintValue,
			Tokens: estimateApproxTokens(canonicalizeCacheValue(fingerprintValue)),
		})
	}

	for messageIndex, msg := range req.Messages {
		wrapper := map[string]interface{}{
			"kind":          "message",
			"message_index": messageIndex,
			"role":          msg.Role,
			"content":       msg.Content,
			"tool_call_id":  msg.ToolCallID,
			"tool_calls":    msg.ToolCalls,
		}
		fingerprintValue := stripCachePositionKeys(wrapper)
		canonical := canonicalizeCacheValue(fingerprintValue)
		blocks = append(blocks, cacheablePromptBlock{
			Value:              fingerprintValue,
			Tokens:             estimateApproxTokens(canonical),
			IsMessageEnd:       true,
			ImplicitBreakpoint: true,
		})
	}

	return blocks
}

func buildCachePreludeBlock(req *ClaudeRequest) cacheablePromptBlock {
	prelude := map[string]interface{}{
		"kind":        "request_prelude",
		"model":       req.Model,
		"tool_choice": req.ToolChoice,
	}
	return cacheablePromptBlock{
		Value:  prelude,
		Tokens: estimateApproxTokens(canonicalizeCacheValue(prelude)),
	}
}

func appendSystemCacheBlocks(blocks *[]cacheablePromptBlock, system interface{}) {
	switch v := system.(type) {
	case string:
		appendPromptBlock(blocks, map[string]interface{}{
			"kind":         "system",
			"system_index": 0,
			"block": map[string]interface{}{
				"type": "text",
				"text": v,
			},
		}, false)
	case []interface{}:
		for i, block := range v {
			appendPromptBlock(blocks, map[string]interface{}{
				"kind":         "system",
				"system_index": i,
				"block":        block,
			}, false)
		}
	case []string:
		for i, block := range v {
			appendPromptBlock(blocks, map[string]interface{}{
				"kind":         "system",
				"system_index": i,
				"block": map[string]interface{}{
					"type": "text",
					"text": block,
				},
			}, false)
		}
	}
}

func appendMessageCacheBlocks(blocks *[]cacheablePromptBlock, messageIndex int, msg ClaudeMessage) {
	role := msg.Role
	switch content := msg.Content.(type) {
	case string:
		appendPromptBlock(blocks, map[string]interface{}{
			"kind":          "message",
			"message_index": messageIndex,
			"role":          role,
			"block_index":   0,
			"block": map[string]interface{}{
				"type": "text",
				"text": content,
			},
		}, true)
	case []interface{}:
		lastIdx := len(content) - 1
		for blockIndex, block := range content {
			appendPromptBlock(blocks, map[string]interface{}{
				"kind":          "message",
				"message_index": messageIndex,
				"role":          role,
				"block_index":   blockIndex,
				"block":         block,
			}, blockIndex == lastIdx)
		}
	default:
		if content != nil {
			appendPromptBlock(blocks, map[string]interface{}{
				"kind":          "message",
				"message_index": messageIndex,
				"role":          role,
				"block_index":   0,
				"block":         content,
			}, true)
		}
	}
}

func appendPromptBlock(blocks *[]cacheablePromptBlock, wrapper map[string]interface{}, isMessageEnd bool) {
	blockValue := wrapper["block"]
	ttl := normalizePromptCacheTTL(extractPromptCacheTTL(blockValue))

	// Drop volatile billing metadata from the cache fingerprint. Claude Code's
	// x-anthropic-billing-header can drift, appear, or disappear across
	// otherwise identical requests, and it does not change model semantics.
	if isAnthropicBillingHeaderBlock(blockValue) {
		return
	}

	fingerprintValue := stripCachePositionKeys(wrapper)
	canonical := canonicalizeCacheValue(fingerprintValue)
	*blocks = append(*blocks, cacheablePromptBlock{
		Value:        fingerprintValue,
		Tokens:       estimateApproxTokens(canonical),
		TTL:          ttl,
		IsMessageEnd: isMessageEnd,
	})
}

func stripCachePositionKeys(value map[string]interface{}) map[string]interface{} {
	cloned := make(map[string]interface{}, len(value))
	for key, item := range value {
		if isCachePositionKey(key) {
			continue
		}
		cloned[key] = item
	}
	return cloned
}

func isAnthropicBillingHeaderBlock(value interface{}) bool {
	blockMap, ok := value.(map[string]interface{})
	if !ok {
		return false
	}

	// Only normalize text blocks (or blocks without an explicit type but containing text).
	if t, ok := blockMap["type"].(string); ok && t != "" && t != "text" {
		return false
	}

	text, ok := blockMap["text"].(string)
	if !ok {
		return false
	}

	trimmed := strings.TrimLeft(text, " \t\r\n")
	return strings.HasPrefix(strings.ToLower(trimmed), "x-anthropic-billing-header:")
}

func extractPromptCacheTTL(value interface{}) time.Duration {
	block, ok := value.(map[string]interface{})
	if !ok {
		if raw, err := json.Marshal(value); err == nil {
			var decoded map[string]interface{}
			if json.Unmarshal(raw, &decoded) == nil {
				block = decoded
				ok = true
			}
		}
	}
	if !ok {
		return 0
	}

	rawCache, ok := block["cache_control"]
	if !ok {
		return 0
	}
	cacheControl, ok := rawCache.(map[string]interface{})
	if !ok {
		return 0
	}
	cacheType, _ := cacheControl["type"].(string)
	if !strings.EqualFold(cacheType, "ephemeral") {
		return 0
	}

	if ttl, ok := parsePromptCacheTTLValue(cacheControl["ttl"]); ok {
		return ttl
	}
	return defaultPromptCacheTTL
}

func parsePromptCacheTTLValue(value interface{}) (time.Duration, bool) {
	switch v := value.(type) {
	case string:
		trimmed := strings.TrimSpace(strings.ToLower(v))
		if trimmed == "" {
			return 0, false
		}
		if d, err := time.ParseDuration(trimmed); err == nil {
			return d, true
		}
		if seconds, err := strconv.Atoi(trimmed); err == nil {
			return time.Duration(seconds) * time.Second, true
		}
	case float64:
		if v > 0 {
			return time.Duration(v) * time.Second, true
		}
	case int:
		if v > 0 {
			return time.Duration(v) * time.Second, true
		}
	case int64:
		if v > 0 {
			return time.Duration(v) * time.Second, true
		}
	}
	return 0, false
}

func normalizePromptCacheTTL(ttl time.Duration) time.Duration {
	if ttl <= 0 {
		return 0
	}
	if ttl > time.Hour {
		return time.Hour
	}
	if ttl > defaultPromptCacheTTL {
		return time.Hour
	}
	return defaultPromptCacheTTL
}

func computePromptCacheTTLBreakdown(profile *promptCacheProfile, matchedTokens int) (int, int) {
	if profile == nil || len(profile.Breakpoints) == 0 {
		return 0, 0
	}

	cache5m := 0
	cache1h := 0
	previous := matchedTokens
	for _, breakpoint := range profile.Breakpoints {
		current := minInt(breakpoint.CumulativeTokens, profile.TotalInputTokens)
		if current <= previous {
			continue
		}
		delta := current - previous
		if breakpoint.TTL >= time.Hour {
			cache1h += delta
		} else {
			cache5m += delta
		}
		previous = current
	}
	return cache5m, cache1h
}

func billedClaudeInputTokens(inputTokens int, usage promptCacheUsage) int {
	return maxInt(inputTokens-usage.CacheCreationInputTokens-usage.CacheReadInputTokens, 0)
}

func buildClaudeUsageMap(inputTokens, outputTokens int, usage promptCacheUsage, includeCache bool) map[string]interface{} {
	result := map[string]interface{}{
		"input_tokens":  billedClaudeInputTokens(inputTokens, usage),
		"output_tokens": outputTokens,
	}
	if !includeCache {
		return result
	}
	result["cache_creation_input_tokens"] = usage.CacheCreationInputTokens
	result["cache_read_input_tokens"] = usage.CacheReadInputTokens
	result["cache_creation"] = map[string]int{
		"ephemeral_5m_input_tokens": usage.CacheCreation5mInputTokens,
		"ephemeral_1h_input_tokens": usage.CacheCreation1hInputTokens,
	}
	return result
}

func canonicalizeCacheValue(value interface{}) string {
	var buf bytes.Buffer
	writeCanonicalJSON(&buf, value)
	return buf.String()
}

func writeCanonicalJSON(buf *bytes.Buffer, value interface{}) {
	switch v := value.(type) {
	case nil:
		buf.WriteString("null")
	case string:
		encoded, _ := json.Marshal(v)
		buf.Write(encoded)
	case bool:
		if v {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	case float64, float32, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, json.Number:
		encoded, _ := json.Marshal(v)
		buf.Write(encoded)
	case []interface{}:
		buf.WriteByte('[')
		for i, item := range v {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeCanonicalJSON(buf, item)
		}
		buf.WriteByte(']')
	case map[string]interface{}:
		buf.WriteByte('{')
		keys := make([]string, 0, len(v))
		for key := range v {
			if key == "cache_control" {
				continue
			}
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for i, key := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			encoded, _ := json.Marshal(key)
			buf.Write(encoded)
			buf.WriteByte(':')
			writeCanonicalJSON(buf, v[key])
		}
		buf.WriteByte('}')
	default:
		encoded, _ := json.Marshal(v)
		buf.Write(encoded)
	}
}

func isCachePositionKey(key string) bool {
	switch key {
	case "tool_index", "system_index", "message_index", "block_index":
		return true
	default:
		return false
	}
}

func writeHashChunk(hasher hashWriter, chunk string) {
	length := strconv.Itoa(len(chunk))
	hasher.Write([]byte(length))
	hasher.Write([]byte{0})
	hasher.Write([]byte(chunk))
	hasher.Write([]byte{0})
}

type hashWriter interface {
	Write([]byte) (int, error)
	Sum([]byte) []byte
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
