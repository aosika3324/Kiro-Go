package proxy

import (
	"os"
	"strings"
	"testing"
)

// TestMain forces the v1.1.2 history-rewrite transforms ON for the proxy test
// suite. Those transforms (sanitizeKiroHistory + truncatePayloadToLimit) are
// gated behind config.HistoryRewrite and default to OFF in production (see
// [[history-rewrite-gate]] / config.HistoryRewrite), but a number of upstream
// tests — translator_compaction_test.go, tool_narration_pollution_test.go,
// translator_truncate_test.go, and the orphan/placeholder cases in
// translator_test.go — assert the rewrite-ON behavior. Forcing the seam ON here
// keeps them green without touching each test. OFF-path tests must opt out
// explicitly via SetHistoryRewriteForTest(false).
func TestMain(m *testing.M) {
	restore := SetHistoryRewriteForTest(true)
	code := m.Run()
	restore()
	os.Exit(code)
}

// TestClaudeToKiroPreservesStructuredToolHistoryWhenRewriteOff is the regression
// guard for the gate: with the history rewrite disabled (the production
// default), historical structured tool_use / tool_result turns must survive
// verbatim — not be flattened into "Tool results:" text — because flattening
// degraded multi-turn output quality.
func TestClaudeToKiroPreservesStructuredToolHistoryWhenRewriteOff(t *testing.T) {
	restore := SetHistoryRewriteForTest(false)
	defer restore()

	req := &ClaudeRequest{
		Model: "claude-opus-4.8",
		Messages: []ClaudeMessage{
			{Role: "user", Content: "run the build"},
			{Role: "assistant", Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "running build"},
				map[string]interface{}{"type": "tool_use", "id": "t1", "name": "exec_command", "input": map[string]interface{}{"cmd": "make"}},
			}},
			{Role: "user", Content: []interface{}{
				map[string]interface{}{"type": "tool_result", "tool_use_id": "t1", "content": "build ok"},
			}},
			{Role: "user", Content: "now summarize"},
		},
	}

	payload := ClaudeToKiro(req, false)

	// The historical assistant turn must keep its structured tool call.
	foundStructuredToolUse := false
	for _, h := range payload.ConversationState.History {
		if a := h.AssistantResponseMessage; a != nil && len(a.ToolUses) > 0 {
			for _, tu := range a.ToolUses {
				if tu.ToolUseID == "t1" && tu.Name == "exec_command" {
					foundStructuredToolUse = true
				}
			}
		}
	}
	if !foundStructuredToolUse {
		t.Fatalf("expected structured tool_use t1 preserved in history when rewrite is off")
	}

	// The historical tool result must remain structured (not narrated into text).
	foundStructuredToolResult := false
	narratedIntoText := false
	for _, h := range payload.ConversationState.History {
		if u := h.UserInputMessage; u != nil {
			if u.UserInputMessageContext != nil {
				for _, tr := range u.UserInputMessageContext.ToolResults {
					if tr.ToolUseID == "t1" {
						foundStructuredToolResult = true
					}
				}
			}
			if strings.Contains(u.Content, toolResultsContinuationPrefix) && strings.Contains(u.Content, "build ok") {
				narratedIntoText = true
			}
		}
	}
	if !foundStructuredToolResult {
		t.Fatalf("expected structured tool_result t1 preserved in history when rewrite is off")
	}
	if narratedIntoText {
		t.Fatalf("tool result should not be narrated into text when rewrite is off")
	}
}

// TestClaudeToKiroDoesNotTruncateWhenRewriteOff verifies the hard payload-size
// cap is inactive by default: an oversized conversation passes through without a
// truncation placeholder.
func TestClaudeToKiroDoesNotTruncateWhenRewriteOff(t *testing.T) {
	restore := SetHistoryRewriteForTest(false)
	defer restore()

	big := strings.Repeat("lorem ipsum dolor sit amet ", 80) // ~2.1KB
	msgs := []ClaudeMessage{{Role: "user", Content: "start"}}
	for i := 0; i < 800; i++ {
		msgs = append(msgs,
			ClaudeMessage{Role: "assistant", Content: "step: " + big},
			ClaudeMessage{Role: "user", Content: "next: " + big},
		)
	}
	msgs = append(msgs, ClaudeMessage{Role: "user", Content: "FINAL"})

	payload := ClaudeToKiro(&ClaudeRequest{Model: "claude-opus-4.8", Messages: msgs}, false)

	for _, h := range payload.ConversationState.History {
		if h.UserInputMessage != nil && strings.Contains(h.UserInputMessage.Content, "truncated to fit") {
			t.Fatalf("payload should not be truncated when rewrite is off")
		}
	}
}

// TestOpenAIToKiroToolResultHistoryHasContentWhenRewriteOff guards the trickiest
// part of the gate. Upstream v1.1.2 stopped pre-filling a history tool-result
// turn's Content (it delegated narration to sanitizeKiroHistory). With the
// rewrite off, nothing narrates later, so the OpenAI builder must restore the
// buildToolResultsContinuation prefill — otherwise the history turn would go
// upstream with empty content while still carrying structured tool results.
func TestOpenAIToKiroToolResultHistoryHasContentWhenRewriteOff(t *testing.T) {
	restore := SetHistoryRewriteForTest(false)
	defer restore()

	req := &OpenAIRequest{
		Model: "claude-opus-4.8",
		Messages: []OpenAIMessage{
			{Role: "user", Content: "run it"},
			{Role: "assistant", Content: "", ToolCalls: []ToolCall{
				newPollToolCall("call_1", "exec_command", `{"cmd":"make"}`),
			}},
			{Role: "tool", ToolCallID: "call_1", Content: "BUILD_OK"},
			{Role: "user", Content: "now summarize"},
		},
	}

	payload := OpenAIToKiro(req, false)

	// Find the history tool-result turn (the one carrying structured ToolResults).
	var toolTurn *KiroUserInputMessage
	for i := range payload.ConversationState.History {
		u := payload.ConversationState.History[i].UserInputMessage
		if u != nil && u.UserInputMessageContext != nil && len(u.UserInputMessageContext.ToolResults) > 0 {
			toolTurn = u
			break
		}
	}
	if toolTurn == nil {
		t.Fatalf("expected a structured tool-result history turn when rewrite is off")
	}
	if strings.TrimSpace(toolTurn.Content) == "" {
		t.Fatalf("tool-result history turn must have non-empty Content when rewrite is off (would go upstream empty)")
	}
	if !strings.Contains(toolTurn.Content, "BUILD_OK") {
		t.Fatalf("expected tool output BUILD_OK in the prefilled continuation, got %q", toolTurn.Content)
	}
}
