/* ── Score cell gradient colouring ──────────────────────────── */
function scoreHue(val) {
  var c = Math.max(0, Math.min(1, val));
  return Math.round(c * 120); // 0=red, 60=yellow, 120=green
}

document.querySelectorAll('td.score-cell').forEach(function (cell) {
  var val = parseFloat(cell.textContent);
  if (isNaN(val)) return;
  cell.style.color = 'hsl(' + scoreHue(val) + ', 72%, 62%)';
});

/* ── Score pill dynamic colour ───────────────────────────────── */
document.querySelectorAll('.score-pill[data-score]').forEach(function (pill) {
  var val = parseFloat(pill.dataset.score);
  if (isNaN(val)) return;
  var h = scoreHue(val);
  pill.style.background   = 'hsla(' + h + ', 60%, 30%, 0.32)';
  pill.style.color         = 'hsl(' + h + ', 72%, 65%)';
  pill.style.borderColor   = 'hsla(' + h + ', 60%, 50%, 0.35)';
});

/* ── Inner toggle arrow ──────────────────────────────────────── */
document.querySelectorAll('details.inner-toggle').forEach(function (el) {
  var arrow = el.querySelector('.inner-arrow');
  function sync() {
    if (arrow) arrow.style.transform = el.open ? 'rotate(90deg)' : 'rotate(0deg)';
  }
  sync();
  el.addEventListener('toggle', sync);
});
