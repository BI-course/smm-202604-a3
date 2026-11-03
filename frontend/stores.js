/* Existing store-clustering page only.
   This is NOT the additional store classification model.
   The endpoint/response below must match the backend route you already have. */
(() => {
  "use strict";

  const endpoint = "/api/v1/models/store-segmenter/predictions";

  window.App.setupForm("form", endpoint, (data, resultElement) => {
    const cluster = data.assigned_cluster;
    const segment = data.segment_name;
    const model = data.model || "Store clustering model";

    if (cluster === undefined || cluster === null) {
      window.App.showMessage(
        resultElement,
        "The API response did not include assigned_cluster. Check that this endpoint returns the fields expected by stores.html.",
        "error"
      );
      return;
    }

    resultElement.innerHTML = `
      <section class="prediction-result">
        <h2>Store Segment Result</h2>
        <p>Assigned cluster: <strong>${window.App.escapeHtml(cluster)}</strong></p>
        ${segment ? `<p>Segment: <strong>${window.App.escapeHtml(segment)}</strong></p>` : ""}
        <p>Model: ${window.App.escapeHtml(model)}</p>
      </section>`;
    resultElement.style.color = "";
  });
})();
