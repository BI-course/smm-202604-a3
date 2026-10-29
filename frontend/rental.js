/* Rent prediction page. Requires app.js. */
(() => {
  "use strict";

  const endpoint = "/api/v1/models/rent-prediction-regressor/predictions";

  window.App.setupForm("form", endpoint, (data, resultElement) => {
    const rent = Number(data.predicted_monthly_rent_kes);
    if (!Number.isFinite(rent)) {
      window.App.showMessage(resultElement, "The API response did not include a valid predicted rent.", "error");
      return;
    }

    resultElement.innerHTML = `
      <section class="prediction-result">
        <h2>Estimated Monthly Rent</h2>
        <p><strong>KES ${rent.toLocaleString("en-KE", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        })}</strong></p>
        <p>Model: ${window.App.escapeHtml(data.model || "Random Forest")}</p>
      </section>`;
    resultElement.style.color = "";
  });
})();
