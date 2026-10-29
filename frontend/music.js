/* Music recommendation page. Requires app.js.
   The API expects track names in "basket", not track IDs. */
(() => {
  "use strict";

  const form = document.getElementById("form");
  const tracksContainer = document.getElementById("tracks");
  const clearButton = document.getElementById("clear");
  if (!form || !tracksContainer) return;

  const fallbackTracks = [
    ["M01", "Deep Sleep"],
    ["M02", "Weightless"],
    ["M03", "Rain Sounds"],
    ["M04", "Piano for Relaxation"],
    ["M05", "Nature Sounds"],
    ["M06", "Focus Music"],
    ["M07", "Deep Sleep"]
  ];

  const resultElement = (() => {
    let element = document.getElementById("result") ||
      document.getElementById("results") ||
      document.getElementById("output");
    if (!element) {
      element = document.createElement("div");
      element.id = "result";
      element.setAttribute("aria-live", "polite");
      element.style.marginTop = "1rem";
      form.insertAdjacentElement("afterend", element);
    }
    return element;
  })();

  function normalizeTrack(item) {
    const id = item.track_id ?? item.id ?? "";
    const name = item.track_name ?? item.item_name ?? item.product_name ?? item.name ?? "";
    return { id: String(id), name: String(name) };
  }

  function drawTracks(items) {
    tracksContainer.innerHTML = "";
    const seen = new Set();

    for (const item of items) {
      const track = normalizeTrack(Array.isArray(item)
        ? { track_id: item[0], track_name: item[1] }
        : item);
      if (!track.name || seen.has(track.name)) continue;
      seen.add(track.name);

      const label = document.createElement("label");
      label.style.display = "block";
      label.style.margin = "0.35rem 0";

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = "basket";
      checkbox.value = track.name;
      checkbox.dataset.trackId = track.id;
      checkbox.style.marginRight = "0.5rem";

      label.append(checkbox, document.createTextNode(track.name));
      tracksContainer.appendChild(label);
    }
  }

  async function loadTracks() {
    try {
      // This endpoint must be implemented by the backend. If it is unavailable,
      // the fallback list below will still let the page render.
      const data = await window.App.callApi("/api/v1/tracks", undefined, { method: "GET" });
      const items = Array.isArray(data) ? data : (data.tracks || data.items || []);
      if (items.length) {
        drawTracks(items);
        return;
      }
    } catch (error) {
      console.info("Could not load track list from API; using fallback list.", error.message);
    }
    drawTracks(fallbackTracks);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const basket = Array.from(tracksContainer.querySelectorAll('input[name="basket"]:checked'))
      .map((checkbox) => checkbox.value);

    if (!basket.length) {
      window.App.showMessage(resultElement, "Select at least one track before requesting recommendations.", "error");
      return;
    }

    const topNField = form.querySelector('[name="top_n"]');
    const topN = topNField && topNField.value.trim() ? Number(topNField.value) : 5;
    if (!Number.isInteger(topN) || topN < 1) {
      window.App.showMessage(resultElement, "Number of recommendations must be a positive whole number.", "error");
      return;
    }

    const submitButton = form.querySelector('button[type="submit"], input[type="submit"]');
    const originalText = submitButton
      ? (submitButton.tagName === "INPUT" ? submitButton.value : submitButton.textContent)
      : "";
    if (submitButton) {
      submitButton.disabled = true;
      if (submitButton.tagName === "INPUT") submitButton.value = "Finding…";
      else submitButton.textContent = "Finding…";
    }

    window.App.showMessage(resultElement, "Finding tracks…", "info");
    try {
      const data = await window.App.callApi(
        "/api/v1/models/music-recommender/predictions",
        { basket, top_n: topN }
      );

      const recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
      if (!recommendations.length) {
        window.App.showMessage(
          resultElement,
          data.message || "No recommendations were found for this selection.",
          "info"
        );
        return;
      }

      const rows = recommendations.map((item) => `
        <li>
          <strong>${window.App.escapeHtml(item.product_name || item.track_name || item.track_id || "Recommended track")}</strong>
          <div>Confidence: ${Number(item.confidence).toFixed(3)} · Lift: ${Number(item.lift).toFixed(2)} · Support: ${Number(item.support).toFixed(4)}</div>
          ${Array.isArray(item.because_you_played) && item.because_you_played.length
            ? `<div>Because you played: ${item.because_you_played.map(window.App.escapeHtml).join(", ")}</div>`
            : ""}
        </li>`).join("");

      resultElement.innerHTML = `
        <section class="prediction-result">
          <h2>Recommended Tracks</h2>
          <p>Based on: ${basket.map(window.App.escapeHtml).join(", ")}</p>
          <ol>${rows}</ol>
          <p>Model: ${window.App.escapeHtml(data.model || "Apriori association rules")}</p>
        </section>`;
      resultElement.style.color = "";
    } catch (error) {
      window.App.showMessage(resultElement, error.message || "Could not get recommendations.", "error");
    } finally {
      if (submitButton) {
        submitButton.disabled = false;
        if (submitButton.tagName === "INPUT") submitButton.value = originalText;
        else submitButton.textContent = originalText;
      }
    }
  });

  if (clearButton) {
    clearButton.addEventListener("click", () => {
      tracksContainer.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
        checkbox.checked = false;
      });
      resultElement.textContent = "";
    });
  }

  loadTracks();
})();
