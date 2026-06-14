package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// defaultRequestLogCapacity is how many recent inference-API request records the
// in-memory ring buffer keeps. The buffer is intentionally volatile: it is a live
// monitoring aid, not an audit log, and is cleared on restart.
const defaultRequestLogCapacity = 500

// requestLogSubscriberBuffer is the per-subscriber channel depth. A slow SSE
// client that falls this far behind is dropped rather than blocking Record.
const requestLogSubscriberBuffer = 64

// RequestLogEntry is a single inference-API request record surfaced in the admin
// live log. It carries only what the monitoring view needs; it never includes
// tokens, message bodies, or full API keys.
type RequestLogEntry struct {
	Seq           uint64  `json:"seq"`           // Monotonic sequence number (for client dedup/ordering)
	Time          int64   `json:"time"`          // Unix milliseconds when the request completed
	AccountEmail  string  `json:"accountEmail"`  // Serving account (email or id fallback)
	AccountID     string  `json:"accountId"`     // Serving account id
	ClientIP      string  `json:"clientIp"`      // Caller IP (X-Forwarded-For / X-Real-IP aware)
	KeyName       string  `json:"keyName"`       // Matched API key name (or masked key / "-")
	Model         string  `json:"model"`         // Requested model (with thinking suffix stripped)
	InputTokens   int     `json:"inputTokens"`   // Reported input tokens (billed + cache)
	OutputTokens  int     `json:"outputTokens"`  // Reported output tokens
	CacheRead     int     `json:"cacheRead"`     // Reported cache_read_input_tokens
	CacheCreation int     `json:"cacheCreation"` // Reported cache_creation_input_tokens
	Credits       float64 `json:"credits"`       // Credits consumed for this request
	Endpoint      string  `json:"endpoint"`      // "claude" | "openai" | "responses"
	Stream        bool    `json:"stream"`        // Whether this was a streaming response
	Status        string  `json:"status"`        // "success" | "error"
}

// requestLogger keeps a fixed-size ring buffer of recent request records and
// fans new records out to any active SSE subscribers. All state is in memory;
// nothing is persisted.
type requestLogger struct {
	mu       sync.Mutex
	buf      []RequestLogEntry
	capacity int
	start    int // index of the oldest entry in buf
	size     int // number of valid entries
	seq      uint64

	subMu sync.Mutex
	subs  map[*logSubscriber]struct{}
}

type logSubscriber struct {
	ch     chan RequestLogEntry
	closed atomic.Bool
}

func newRequestLogger(capacity int) *requestLogger {
	if capacity <= 0 {
		capacity = defaultRequestLogCapacity
	}
	return &requestLogger{
		buf:      make([]RequestLogEntry, capacity),
		capacity: capacity,
		subs:     make(map[*logSubscriber]struct{}),
	}
}

// Record appends an entry to the ring buffer (evicting the oldest when full),
// stamps it with a monotonic sequence number and completion time, then broadcasts
// it to all subscribers. Broadcast is non-blocking: a subscriber whose buffer is
// full is skipped for this entry rather than stalling the request path.
func (l *requestLogger) Record(e RequestLogEntry) {
	if l == nil {
		return
	}
	l.mu.Lock()
	l.seq++
	e.Seq = l.seq
	if e.Time == 0 {
		e.Time = time.Now().UnixMilli()
	}
	// Append into the ring.
	idx := (l.start + l.size) % l.capacity
	if l.size < l.capacity {
		l.buf[idx] = e
		l.size++
	} else {
		// Full: overwrite oldest and advance start.
		l.buf[l.start] = e
		l.start = (l.start + 1) % l.capacity
	}
	l.mu.Unlock()

	l.broadcast(e)
}

// Recent returns up to n most recent entries in chronological order (oldest
// first). n <= 0 or n greater than the stored count returns everything stored.
func (l *requestLogger) Recent(n int) []RequestLogEntry {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.size == 0 {
		return []RequestLogEntry{}
	}
	count := l.size
	if n > 0 && n < count {
		count = n
	}
	out := make([]RequestLogEntry, 0, count)
	// The newest `count` entries: skip the oldest (size-count).
	skip := l.size - count
	for i := skip; i < l.size; i++ {
		out = append(out, l.buf[(l.start+i)%l.capacity])
	}
	return out
}

func (l *requestLogger) broadcast(e RequestLogEntry) {
	l.subMu.Lock()
	defer l.subMu.Unlock()
	for sub := range l.subs {
		if sub.closed.Load() {
			continue
		}
		select {
		case sub.ch <- e:
		default:
			// Subscriber is too slow; drop this entry for it rather than block.
		}
	}
}

// subscribe registers a new SSE subscriber and returns it. Call unsubscribe when
// the client disconnects.
func (l *requestLogger) subscribe() *logSubscriber {
	sub := &logSubscriber{ch: make(chan RequestLogEntry, requestLogSubscriberBuffer)}
	l.subMu.Lock()
	l.subs[sub] = struct{}{}
	l.subMu.Unlock()
	return sub
}

func (l *requestLogger) unsubscribe(sub *logSubscriber) {
	if sub == nil {
		return
	}
	l.subMu.Lock()
	if _, ok := l.subs[sub]; ok {
		delete(l.subs, sub)
		sub.closed.Store(true)
		close(sub.ch)
	}
	l.subMu.Unlock()
}

// apiGetLogs serves the most recent request-log entries as JSON. Optional ?limit=N
// caps the count (default: all stored). Admin auth is enforced by handleAdminAPI
// before this is reached.
func (h *Handler) apiGetLogs(w http.ResponseWriter, r *http.Request) {
	limit := 0
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			limit = n
		}
	}
	entries := h.reqLog.Recent(limit)
	json.NewEncoder(w).Encode(map[string]interface{}{"logs": entries})
}

// apiStreamLogs streams new request-log entries to the client over SSE. The
// caller is already admin-authenticated by handleAdminAPI (which accepts the
// X-Admin-Password header or the admin_password cookie; EventSource can only send
// the cookie, so the cookie path is what makes this work in the browser).
func (h *Handler) apiStreamLogs(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "streaming unsupported"})
		return
	}

	// Override the JSON content-type that handleAdminAPI preset for /admin/api/*.
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // disable proxy buffering (nginx)
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	sub := h.reqLog.subscribe()
	defer h.reqLog.unsubscribe(sub)

	// Initial comment so the client's onopen fires promptly even before traffic.
	fmt.Fprint(w, ": connected\n\n")
	flusher.Flush()

	ctx := r.Context()
	keepalive := time.NewTicker(25 * time.Second)
	defer keepalive.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case entry, ok := <-sub.ch:
			if !ok {
				return
			}
			data, err := json.Marshal(entry)
			if err != nil {
				continue
			}
			fmt.Fprintf(w, "event: log\ndata: %s\n\n", data)
			flusher.Flush()
		case <-keepalive.C:
			// SSE comment line as heartbeat to keep idle connections alive.
			fmt.Fprint(w, ": keepalive\n\n")
			flusher.Flush()
		}
	}
}
