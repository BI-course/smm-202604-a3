/* music.js -- logic for music.html (end-user view).
   Load app.js first: this file uses its helpers through window.App (callApi, escapeHtml).

   The listener ticks tracks and gets a recommendation. Rules, confidence and lift stay
   inside the API; they are not shown here.
   Fails gracefully: no matching rule -> a friendly "no recommendation yet" card;
   server unreachable or bad request -> a plain error message. */
(() => {
  "use strict";

  const { callApi, escapeHtml: esc } = window.App;

  const PREDICT_PATH = "/api/v1/models/music-recommender/predictions";

  // Must match the item names the rules were mined on (lookup_table["item_name"]).
  // A name the API does not know is rejected with a clear 400 error, shown below.
  const TRACKS = ["Workout Anthem", "Pump It Up", "Cardio Beats", "Study Focus 1", "Study Focus 2", "Rain Sounds",
    "Deep Sleep", "Sunday Worship 1", "Sunday Worship 2", "Praise Medley", "Heartbreak Ballad", "Love Story",
    "Hip Hop Anthem", "Street Beats", "Amapiano Vibes", "Gengetone Hit"];

  const $ = (id) => document.getElementById(id);
  const errorBox = (message) => `<div class="error" role="alert">${esc(message)}</div>`;

  function drawTracks() {
    $("tracks").innerHTML = TRACKS.map((name) =>
      `<label class="track"><input type="checkbox" value="${esc(name)}"> ${esc(name)}</label>`).join("");
  }

  function selectedTracks() {
    return [...document.querySelectorAll("#tracks input:checked")].map((box) => box.value);
  }

  function resultHtml(data) {
    if (!data.recommendations || !data.recommendations.length) {
      return `<div class="result empty"><b>No recommendation yet</b>
        <div class="muted">We don't have a suggestion for this mix of tracks. Try adding another track or choosing different ones.</div></div>`;
    }
    return `<div class="result"><div class="muted">You might like next</div>` + data.recommendations.map((r) =>
      `<div class="rec"><div class="big">${esc(r.product_name)}</div>` +
      (r.because_you_played && r.because_you_played.length
        ? `<div class="muted">Because you played ${r.because_you_played.map(esc).join(" + ")}</div>` : "") +
      `</div>`).join("") + `</div>`;
  }

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const out = $("result"), btn = $("form").querySelector("button[type=submit]"), basket = selectedTracks();
    if (!basket.length) { out.innerHTML = errorBox("Pick at least one track."); return; }
    btn.disabled = true; btn.textContent = "Working...";
    try {
      const data = await callApi(PREDICT_PATH, { basket, top_n: 3 });
      out.innerHTML = resultHtml(data);
    } catch (err) {
      out.innerHTML = errorBox(err.message);
    } finally {
      btn.disabled = false; btn.textContent = "Recommend";
    }
  });

  $("clear").addEventListener("click", () => {
    document.querySelectorAll("#tracks input").forEach((box) => { box.checked = false; });
    $("result").innerHTML = "";
  });

  drawTracks();
})();