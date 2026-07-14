/**
 * OMLX Research Panel — JavaScript Controller
 * =============================================
 *
 * Bridges the research panel HTML template and the Python API routes.
 * Responsibilities:
 *   - Polls GET /api/research/status every 5 s and updates status cards.
 *   - Handles agent "Run" form submissions as POST /api/research/agents/run.
 *   - Reads / writes TurboQuant+ config via GET/POST /api/research/turboquant/config.
 *   - Updates the UI on each poll cycle with inline animations.
 *
 * Usage
 * -----
 * Include after the research panel HTML has loaded:
 *   <script src="/static/admin-extensions/js/research_controller.js"></script>
 *   <script>ResearchPanelController.init();</script>
 *
 * Dependencies: none (vanilla ES2022).
 */
(function () {
  "use strict";

  const ResearchPanelController = {
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    config: {
      statusEndpoint: "/api/research/status",
      agentsListEndpoint: "/api/research/agents/list",
      agentsRunEndpoint: "/api/research/agents/run",
      turboquantConfigEndpoint: "/api/research/turboquant/config",
      pollIntervalMs: 5000,
      retryDelayMs: 2000,
      maxRetries: 3,
    },

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    _pollTimer: null,
    _retryCount: 0,
    _isPolling: false,

    // -----------------------------------------------------------------------
    // Initialisation
    // -----------------------------------------------------------------------

    /** Initialise the controller — wire DOM, fetch initial data, start poll. */
    init: function () {
      if (document.querySelector(".rp-container") === null) {
        console.warn(
          "[ResearchPanel] No .rp-container found; skipping initialisation."
        );
        return;
      }
      this._wireAgentForms();
      this._wireRefreshButton();
      this._wireTurboquantForms();
      this._fetchAll();
      this._startPolling();
    },

    /** Stop all polling and clean up. */
    destroy: function () {
      if (this._pollTimer) {
        clearTimeout(this._pollTimer);
        this._pollTimer = null;
      }
      this._isPolling = false;
    },

    // -----------------------------------------------------------------------
    // Data fetching
    // -----------------------------------------------------------------------

    /** Fetch status + agent list in parallel, then update the DOM. */
    _fetchAll: function () {
      const container = document.querySelector(".rp-container");
      if (!container) return;

      Promise.all([
        this._fetchJSON(this.config.statusEndpoint),
        this._fetchJSON(this.config.agentsListEndpoint).catch(() => null),
      ])
        .then(([statusData, agentsData]) => {
          this._retryCount = 0;
          if (statusData) this._updateBackendCards(statusData);
          if (agentsData) this._updateAgentCards(agentsData);
          this._updateStatusBar(statusData);
        })
        .catch((err) => {
          console.error("[ResearchPanel] Fetch error:", err);
          this._retryCount++;
          if (this._retryCount > this.config.maxRetries) {
            this._showError(
              "Failed to reach the research API after several attempts."
            );
          }
        });
    },

    /** Thin wrapper around fetch() that returns JSON or null. */
    _fetchJSON: function (url, options) {
      return fetch(url, options || {})
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status} on ${url}`);
          return res.json();
        })
        .catch((err) => {
          console.warn(`[ResearchPanel] GET ${url} failed:`, err.message);
          return null;
        });
    },

    // -----------------------------------------------------------------------
    // Polling
    // -----------------------------------------------------------------------

    _startPolling: function () {
      if (this._isPolling) return;
      this._isPolling = true;
      this._pollLoop();
    },

    _pollLoop: function () {
      if (!this._isPolling) return;
      this._fetchAll();
      this._pollTimer = setTimeout(() => this._pollLoop(), this.config.pollIntervalMs);
    },

    // -----------------------------------------------------------------------
    // DOM updates — Status cards
    // -----------------------------------------------------------------------

    /**
     * Update every backend card from a /api/research/status response.
     * The response shape is:
     *   { backends: [{ id, name, primary, available, cuda, metal,
     *                  supports_batching, supports_streaming,
     *                  supports_turboquant, supports_spec_decode }],
     *     turboquant: { installed, version, path, turbo_kv_cache },
     *     turboquant_config: { enabled, kv_cache_bits, ... },
     *     timestamp }
     */
    _updateBackendCards: function (data) {
      const backends = data && Array.isArray(data.backends) ? data.backends : [];
      for (const b of backends) {
        const card = document.querySelector(`[data-backend-id="${b.id}"]`);
        if (!card) continue;

        // Status badge
        const badge = card.querySelector(".rp-badge");
        if (badge) {
          badge.textContent = b.available ? "Online" : "Offline";
          badge.className = "rp-badge";
          badge.classList.add(b.available ? "rp-badge--online" : "rp-badge--offline");
        }

        // Left-edge indicator
        const indicator = card.querySelector(".rp-card-indicator");
        if (indicator) {
          indicator.className = "rp-card-indicator";
          indicator.classList.add(b.available ? "rp-card-indicator--online" : "rp-card-indicator--offline");
        }

        // Capability tags
        const caps = card.querySelector(".rp-capability-list");
        if (caps) {
          const enabled = b.available ? b : {};
          caps.querySelectorAll(".rp-cap").forEach((el) => {
            const key = el.getAttribute("data-cap");
            const supported = enabled[key] === true;
            el.classList.toggle("rp-cap--disabled", !supported);
            el.title = supported ? "Available" : "Not available";
          });
        }
      }

      // TurboQuant+ dashboard status
      this._updateTurboquantStatus(data);
    },

    /** Update the TurboQuant+ indicator in the status bar / header. */
    _updateTurboquantStatus: function (data) {
      const el = document.querySelector("[data-tq-status]");
      if (!el || !data) return;
      const tq = data.turboquant || {};
      const ok = tq.installed || tq.turbo_kv_cache;
      el.textContent = ok ? "TurboQuant+: OK" : "TurboQuant+: N/A";
      el.className = ok ? "rp-badge rp-badge--online" : "rp-badge rp-badge--offline";

      // Update config section if present
      const cfg = data.turboquant_config;
      if (cfg) this._updateTurboquantConfigUI(cfg);
    },

    // -----------------------------------------------------------------------
    // DOM updates — Agent cards
    // -----------------------------------------------------------------------

    /** Populate agent cards with description text. */
    _updateAgentCards: function (data) {
      const agents = Array.isArray(data) ? data : data && Array.isArray(data.agents) ? data.agents : [];
      for (const a of agents) {
        const card = document.querySelector(`[data-agent-id="${a.id}"]`);
        if (!card) continue;
        const desc = card.querySelector(".rp-agent-description");
        if (desc && a.description) {
          desc.textContent = a.description;
        }
      }
    },

    // -----------------------------------------------------------------------
    // Agent form handling
    // -----------------------------------------------------------------------

    _wireAgentForms: function () {
      const forms = document.querySelectorAll(".rp-agent-form");
      for (const form of forms) {
        // Remove stale listener if re-wired
        const clone = form.cloneNode(true);
        form.parentNode.replaceChild(clone, form);
        clone.addEventListener("submit", (e) => this._onAgentSubmit(e));
      }
    },

    _onAgentSubmit: function (e) {
      e.preventDefault();
      const form = e.currentTarget;
      const agentId = form.getAttribute("data-agent-id") || "";
      const textarea = form.querySelector("textarea");
      const prompt = textarea ? textarea.value.trim() : "";
      const outputArea = form.parentNode.querySelector(".rp-output-area");

      if (!prompt) {
        this._appendOutput(outputArea, "Please enter a prompt.", "error");
        return;
      }

      const submitBtn = form.querySelector('button[type="submit"]');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Running…";
      }

      this._appendOutput(outputArea, `⏳ Running ${agentId}…`, "info");

      this._fetchJSON(this.config.agentsRunEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent: agentId, prompt }),
      })
        .then((result) => {
          if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = "Run";
          }
          if (!result) {
            this._appendOutput(outputArea, "✗ No response from server.", "error");
            return;
          }
          if (result.ok) {
            const elapsed = result.elapsed_ms ? ` (${result.elapsed_ms} ms)` : "";
            this._appendOutput(
              outputArea,
              `✓ ${result.agent} finished${elapsed}:\n${result.output}`,
              "ok"
            );
          } else {
            this._appendOutput(
              outputArea,
              `✗ ${result.agent} failed: ${result.error || "unknown error"}`,
              "error"
            );
          }
        })
        .catch((err) => {
          if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = "Run";
          }
          this._appendOutput(outputArea, `✗ Network error: ${err.message}`, "error");
        });
    },

    _appendOutput: function (area, text, className) {
      if (!area) return;
      const line = document.createElement("div");
      line.className = `rp-output-line rp-output-line--${className || "info"}`;
      line.textContent = text;
      area.appendChild(line);
      area.scrollTop = area.scrollHeight;
    },

    // -----------------------------------------------------------------------
    // TurboQuant+ Config
    // -----------------------------------------------------------------------

    _wireTurboquantForms: function () {
      // Toggle switch for enabling TurboQuant+
      const toggle = document.querySelector("#tq-enabled");
      if (toggle) {
        toggle.addEventListener("change", () => this._applyTqConfig());
      }

      // Slider / number inputs
      const inputs = document.querySelectorAll(
        ".rp-config-section input[type='range'], .rp-config-section input[type='number']"
      );
      for (const input of inputs) {
        input.addEventListener("input", () => {
          const display = document.querySelector(`[data-display-for="${input.id}"]`);
          if (display) display.textContent = input.value;
        });
        input.addEventListener("change", () => this._applyTqConfig());
      }
    },

    /** Read all config controls and POST to the config endpoint. */
    _applyTqConfig: function () {
      const payload = {
        enabled: document.querySelector("#tq-enabled")?.checked || false,
        kv_cache_bits: parseInt(document.querySelector("#tq-kv-cache-bits")?.value, 10) || 4,
        weight_bits: parseInt(document.querySelector("#tq-weight-bits")?.value, 10) || 4,
        block_size: parseInt(document.querySelector("#tq-block-size")?.value, 10) || 64,
        rotation_enabled: document.querySelector("#tq-rotation")?.checked || false,
        outlier_channel_threshold:
          parseFloat(document.querySelector("#tq-outlier-threshold")?.value) || 2.0,
        codebook_size: parseInt(document.querySelector("#tq-codebook-size")?.value, 10) || 65536,
      };

      this._fetchJSON(this.config.turboquantConfigEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).then((result) => {
        if (result && result.ok) {
          this._updateTurboquantConfigUI(result.config);
        }
      });
    },

    /** Update the read-only config display fields from server data. */
    _updateTurboquantConfigUI: function (cfg) {
      if (!cfg) return;
      // Update toggle
      const toggle = document.querySelector("#tq-enabled");
      if (toggle && toggle.checked !== !!cfg.enabled) {
        toggle.checked = !!cfg.enabled;
      }
      // Update slider displays
      for (const [key, value] of Object.entries(cfg)) {
        const display = document.querySelector(`[data-display-for="tq-${key}"]`);
        if (display) display.textContent = value;
      }
    },

    // -----------------------------------------------------------------------
    // Status bar
    // -----------------------------------------------------------------------

    _updateStatusBar: function (data) {
      const tsEl = document.querySelector("[data-last-update]");
      if (tsEl && data && data.timestamp) {
        const d = new Date(data.timestamp * 1000);
        tsEl.textContent = d.toLocaleTimeString();
      }
    },

    _wireRefreshButton: function () {
      const btn = document.querySelector(".rp-refresh-btn");
      if (!btn) return;
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        btn.disabled = true;
        this._fetchAll();
        setTimeout(() => {
          btn.disabled = false;
        }, 1000);
      });
    },

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    _showError: function (msg) {
      const container = document.querySelector(".rp-container");
      if (!container) return;
      const banner = document.createElement("div");
      banner.style.cssText = `
        background: rgba(248,81,73,0.12);
        border: 1px solid var(--rp-accent-red, #f85149);
        border-radius: 8px;
        padding: 12px 16px;
        color: var(--rp-accent-red, #f85149);
        margin-bottom: 16px;
        font-size: 0.9rem;
      `;
      banner.textContent = "⚠ " + msg;
      container.prepend(banner);
      setTimeout(() => banner.remove(), 8000);
    },
  };

  // -------------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------------
  // Attach to `window` so the HTML template can call ResearchPanelController.
  window.ResearchPanelController = ResearchPanelController;

  // Auto-init if the DOM is already loaded; otherwise wait for DOMContentLoaded.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () =>
      ResearchPanelController.init()
    );
  } else {
    ResearchPanelController.init();
  }
})();
