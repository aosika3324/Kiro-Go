package proxy

import (
	"context"
	"kiro-go/config"
	"net/http"
	"strings"
)

// apiKeyContextKey is an unexported type used as the context key for the matched ApiKeyEntry
// so it cannot collide with keys defined in other packages.
type apiKeyContextKey struct{}

// clientIPContextKey is the context key for the resolved caller IP.
type clientIPContextKey struct{}

// authError describes why authentication failed. status is the HTTP status code to send.
type authError struct {
	status  int
	code    string
	message string
}

func (e *authError) Error() string { return e.message }

func newAuthError(status int, code, message string) *authError {
	return &authError{status: status, code: code, message: message}
}

// extractProvidedKey reads the API key from Authorization (Bearer ...) or X-Api-Key header.
func extractProvidedKey(r *http.Request) string {
	authHeader := r.Header.Get("Authorization")
	if strings.HasPrefix(authHeader, "Bearer ") {
		return strings.TrimPrefix(authHeader, "Bearer ")
	}
	if v := r.Header.Get("X-Api-Key"); v != "" {
		return v
	}
	return ""
}

// authenticate validates an incoming request against the configured API keys.
//
// Master switch: config.RequireApiKey. When false, requests pass without checking
// any keys, even if entries exist (so the admin UI can hold draft keys without
// affecting public deployments).
//
// When RequireApiKey is true:
//  1. If ApiKeys is non-empty, the provided key MUST match an enabled, in-quota
//     entry. Returns the matched entry (a copy) so callers can attribute usage.
//  2. Else if the legacy single ApiKey field is set, the provided key MUST match it.
//  3. Else (switch on but nothing configured) → fail-closed: every request is rejected.
//     This prevents the prior bug where toggling auth on without keys silently
//     left the service open.
//
// Returns (entry, nil) on success. entry is nil when the legacy single-key path
// is used or when the master switch is off.
func (h *Handler) authenticate(r *http.Request) (*config.ApiKeyEntry, error) {
	if !config.IsApiKeyRequired() {
		return nil, nil
	}

	provided := extractProvidedKey(r)

	if config.HasApiKeys() {
		if provided == "" {
			return nil, newAuthError(http.StatusUnauthorized, "authentication_error", "Invalid or missing API key")
		}
		entry := config.FindApiKeyByValue(provided)
		if entry == nil {
			return nil, newAuthError(http.StatusUnauthorized, "authentication_error", "Invalid or missing API key")
		}
		if !entry.Enabled {
			return nil, newAuthError(http.StatusUnauthorized, "authentication_error", "API key disabled")
		}
		if overToken, overCredit := config.ApiKeyOverLimit(*entry); overToken || overCredit {
			if overToken {
				return nil, newAuthError(http.StatusTooManyRequests, "rate_limit_error", "token limit exceeded")
			}
			return nil, newAuthError(http.StatusTooManyRequests, "rate_limit_error", "credit limit exceeded")
		}
		return entry, nil
	}

	// Legacy single-key path.
	expected := config.GetApiKey()
	if expected == "" {
		// Auth required but nothing configured → fail closed.
		return nil, newAuthError(http.StatusUnauthorized, "authentication_error", "API key authentication is required but no keys are configured")
	}
	if provided == "" || provided != expected {
		return nil, newAuthError(http.StatusUnauthorized, "authentication_error", "Invalid or missing API key")
	}
	return nil, nil
}

// withApiKeyContext attaches the matched entry to the request context so downstream
// handlers (recordSuccess, etc.) can credit usage against the correct key.
func withApiKeyContext(r *http.Request, entry *config.ApiKeyEntry) *http.Request {
	if entry == nil {
		return r
	}
	ctx := context.WithValue(r.Context(), apiKeyContextKey{}, entry.ID)
	return r.WithContext(ctx)
}

// apiKeyIDFromContext returns the matched API key ID stored in ctx, or empty string.
func apiKeyIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if v, ok := ctx.Value(apiKeyContextKey{}).(string); ok {
		return v
	}
	return ""
}

// resolveClientIP extracts the caller IP from the request, honoring the first
// X-Forwarded-For hop and X-Real-IP (set by reverse proxies / load balancers),
// falling back to the TCP RemoteAddr. The port is stripped from RemoteAddr.
func resolveClientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// XFF is a comma-separated list; the left-most entry is the original client.
		if idx := strings.IndexByte(xff, ','); idx >= 0 {
			return strings.TrimSpace(xff[:idx])
		}
		return strings.TrimSpace(xff)
	}
	if xrip := strings.TrimSpace(r.Header.Get("X-Real-IP")); xrip != "" {
		return xrip
	}
	addr := r.RemoteAddr
	// Strip the port: "ip:port" or "[ipv6]:port".
	if idx := strings.LastIndexByte(addr, ':'); idx >= 0 {
		host := addr[:idx]
		host = strings.TrimPrefix(host, "[")
		host = strings.TrimSuffix(host, "]")
		return host
	}
	return addr
}

// withClientIPContext attaches the resolved caller IP to the request context.
func withClientIPContext(r *http.Request) *http.Request {
	ctx := context.WithValue(r.Context(), clientIPContextKey{}, resolveClientIP(r))
	return r.WithContext(ctx)
}

// clientIPFromContext returns the caller IP stored in ctx, or empty string.
func clientIPFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if v, ok := ctx.Value(clientIPContextKey{}).(string); ok {
		return v
	}
	return ""
}
