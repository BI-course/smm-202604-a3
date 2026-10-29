/* Shared helpers for the Group A3BI frontend.
   Load this file before the page-specific JavaScript file. */
(() => {
  "use strict";

  const DEFAULT_API_ORIGIN = (() => {
    const host = window.location.hostname || "127.0.0.1";
    const isHttps = window.location.protocol === "https:";
    // Local Flask development server normally runs on port 5000.
    if (!isHttps && (window.location.protocol === "file:" || !window.location.port)) {
      return `http://${host}:5000`;
    }
    return `${window.location.protocol}//${host}${window.location.port ? `:${window.location.port}` : ""}`;
  })();

  function getApiOrigin() {
    try {
      const saved = window.localStorage.getItem("apiOrigin");
      return (saved || DEFAULT_API_ORIGIN).replace(/\/+$/, "");
    } catch (_) {
      return DEFAULT_API_ORIGIN.replace(/\/+$/, "");
    }
  }

  async function callApi(path, payload, options = {}) {
    const controller = new AbortController();
    const timeoutMs = options.timeoutMs || 15000;
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${getApiOrigin()}${path}`, {
        method: options.method || "POST",
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {})
        },
        body: payload === undefined ? undefined : JSON.stringify(payload),
        signal: controller.signal
      });

      const text = await response.text();
      let data = {};
      if (text) {
        try {
          data = JSON.parse(text);
        } catch (_) {
          data = { message: text };
        }
      }

      if (!response.ok) {
        const message = data.error || data.message || `Request failed (${response.status})`;
        throw new Error(message);
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("The request timed out. Check that the Flask API is running and try again.");
      }
      if (error instanceof TypeError) {
        throw new Error(`Could not connect to the API at ${getApiOrigin()}. Check that the Flask server is running and that the API origin is correct.`);
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);
  }

  function getResultElement(form) {
    const candidates = [
      form.dataset.resultTarget,
      "result", "results", "output", "prediction-result", "resultBox"
    ].filter(Boolean);

    for (const id of candidates) {
      const element = document.getElementById(id);
      if (element) return element;
    }

    const element = document.createElement("div");
    element.id = "result";
    element.setAttribute("aria-live", "polite");
    element.style.marginTop = "1rem";
    form.insertAdjacentElement("afterend", element);
    return element;
  }

  function showMessage(element, message, type = "info") {
    const colors = {
      error: "#b42318",
      success: "#067647",
      info: "#344054"
    };
    element.textContent = message;
    element.style.color = colors[type] || colors.info;
    element.style.whiteSpace = "pre-wrap";
    element.style.marginTop = "1rem";
  }

  function readForm(form) {
    const payload = {};
    const fields = form.querySelectorAll("input[name], select[name], textarea[name]");

    for (const field of fields) {
      if (field.disabled || field.type === "submit" || field.type === "button" ||
          field.type === "reset" || field.type === "checkbox" || field.type === "radio") {
        continue;
      }

      const raw = field.value.trim();
      const isOptional = field.dataset.optional === "true" || !field.required;

      if (raw === "") {
        if (isOptional) {
          payload[field.name] = null;
          continue;
        }
        throw new Error(`Please complete the "${field.labels?.[0]?.textContent?.trim() || field.name}" field.`);
      }

      if (field.type === "number" || field.dataset.int !== undefined || field.dataset.number !== undefined) {
        const value = Number(raw);
        if (!Number.isFinite(value)) {
          throw new Error(`Enter a valid number for "${field.name}".`);
        }
        if (field.dataset.int !== undefined && !Number.isInteger(value)) {
          throw new Error(`"${field.name}" must be a whole number.`);
        }
        if (field.min !== "" && value < Number(field.min)) {
          throw new Error(`"${field.name}" must be at least ${field.min}.`);
        }
        if (field.max !== "" && value > Number(field.max)) {
          throw new Error(`"${field.name}" must be no more than ${field.max}.`);
        }
        payload[field.name] = value;
      } else {
        payload[field.name] = raw;
      }
    }

    return payload;
  }

  function setupForm(formId, endpoint, renderResult) {
    const form = document.getElementById(formId);
    if (!form) {
      console.warn(`Form #${formId} was not found.`);
      return;
    }

    const resultElement = getResultElement(form);
    let isSubmitting = false;

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (isSubmitting) return;

      isSubmitting = true;
      const submitButton = form.querySelector('button[type="submit"], input[type="submit"]');
      const originalButtonText = submitButton
        ? (submitButton.tagName === "INPUT" ? submitButton.value : submitButton.textContent)
        : "";

      if (submitButton) {
        submitButton.disabled = true;
        if (submitButton.tagName === "INPUT") submitButton.value = "Sending...";
        else submitButton.textContent = "Sending...";
      }
      showMessage(resultElement, "Sending request to the model API…", "info");

      try {
        const payload = readForm(form);
        const data = await callApi(endpoint, payload);
        if (typeof renderResult === "function") {
          renderResult(data, resultElement);
        } else {
          resultElement.textContent = JSON.stringify(data, null, 2);
        }
      } catch (error) {
        showMessage(resultElement, error.message || "Something went wrong.", "error");
      } finally {
        isSubmitting = false;
        if (submitButton) {
          submitButton.disabled = false;
          if (submitButton.tagName === "INPUT") submitButton.value = originalButtonText;
          else submitButton.textContent = originalButtonText;
        }
      }
    });
  }

  window.App = {
    callApi,
    getApiOrigin,
    readForm,
    setupForm,
    escapeHtml,
    showMessage
  };
  // Preserve compatibility with the earlier inline setupForm(...) usage.
  window.callApi = callApi;
  window.setupForm = setupForm;
})();
