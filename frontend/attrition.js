/* Employee attrition page. Requires app.js. */
(() => {
  "use strict";

  const endpoint = "/api/v1/models/employee-attrition-classifier/predictions";

  window.App.setupForm("form", endpoint, (data, resultElement) => {
    const predicted = data.predicted_employee_attrition_category;
    const probabilities = data.class_probabilities || {};

    if (predicted === undefined || predicted === null) {
      window.App.showMessage(resultElement, "The API response did not include the predicted attrition category.", "error");
      return;
    }

    const probabilityRows = Object.entries(probabilities)
      .map(([label, probability]) => {
        const pct = Number(probability) * 100;
        return `<li><span>${window.App.escapeHtml(label)}</span>: <strong>${Number.isFinite(pct) ? `${pct.toFixed(1)}%` : window.App.escapeHtml(probability)}</strong></li>`;
      })
      .join("");

    resultElement.innerHTML = `
      <section class="prediction-result">
        <h2>Employee Attrition Prediction</h2>
        <p>Predicted category: <strong>${window.App.escapeHtml(predicted)}</strong></p>
        ${probabilityRows ? `<h3>Class probabilities</h3><ul>${probabilityRows}</ul>` : ""}
        <p>Model: ${window.App.escapeHtml(data.model || "Support Vector Classifier")}</p>
      </section>`;
    resultElement.style.color = "";
  });
})();
